from __future__ import annotations

import abc
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolName

import numpy as np
import psutil
import torch
import torch.distributed as dist

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer as jit_transfer_hicache_one_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.glm.hugepage_utils import (
    GLM_HICACHE_SHM_DIR,
    _align_up,
    _hugepage_enabled,
    _hugepage_size,
)
from sglang.srt.mem_cache.memory_pool import (
    _get_layer_shard_range,
    KVCache,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.utils import is_cuda, is_dcu, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_dcu = is_dcu()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if not (_is_npu or _is_xpu or _is_mps):
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_lf_pf,
        transfer_kv_all_layer_lf_ph,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
        transfer_kv_per_layer_pf_lf,
        transfer_kv_per_layer_ph_lf,
    )
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

logger = logging.getLogger(__name__)

# Host RAM to leave free when sizing HiCache pools (OS, other processes).
HICACHE_HOST_MEMORY_RESERVE_BYTES: int = 10 * (1024**3)


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostTensorAllocator(abc.ABC):
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Allocate a tensor of given dims and dtype on the memory."""
        self.dtype = dtype
        self.dims = dims
        tensor = torch.empty(dims, dtype=dtype, device=device)
        return tensor


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    else:
        return HostTensorAllocator()


def alloc_with_host_register(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        # On DCU/ROCm the "kernel" hicache io backend runs a GPU transfer kernel
        # that dereferences the host buffer pointer directly. For the GPU to be
        # able to access this host memory, it must be registered as *mapped*
        # (cudaHostRegisterMapped == 2); the transfer kernel then obtains a
        # GPU-accessible pointer via cudaHostGetDevicePointer (see
        # get_rocm_kernel_accessible_ptr in sgl-kernel/csrc/kvcacheio/transfer.cu).
        # With the default flag (0) the host pages are not mapped into the GPU
        # address space and the transfer kernel faults (VMFault). On CUDA, UVA
        # makes flag 0 sufficient, so keep the original behavior there.
        flags = 2 if _is_dcu else 0
        torch.cuda.cudart().cudaHostRegister(
            buffer.data_ptr(), buffer.numel() * buffer.element_size(), flags
        )
    return buffer


def register_host_tensor_for_kernel_access(tensor: torch.Tensor, nbytes: int) -> int:
    """Register host tensor pages for GPU-kernel access.

    DCU/ROCm needs cudaHostRegisterMapped because kernel hicache translates
    host pointers with hipHostGetDevicePointer. CUDA keeps the existing flag 0
    behavior because UVA makes raw host pointers usable there.
    """
    flags = 2 if _is_dcu else 0
    return torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), nbytes, flags)


def checked_register_host_tensor_for_kernel_access(
    tensor: torch.Tensor, nbytes: int, name: str
) -> None:
    err = register_host_tensor_for_kernel_access(tensor, nbytes)
    if err != 0:
        raise RuntimeError(
            f"Failed to register host tensor for kernel access: {name}, "
            f"nbytes={nbytes}, error={err}."
        )


def alloc_with_pin_memory(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


def kernel_accessible_host_ptr(tensor: torch.Tensor) -> int:
    """Return a pointer to ``tensor``'s storage that a GPU transfer kernel can
    dereference.

    The "kernel" hicache io backend passes a table of per-layer base addresses
    (``k_data_ptrs`` / ``v_data_ptrs``) to a GPU kernel that dereferences those
    addresses directly (see ``get_global_offset_lf_tbl`` in
    ``sgl-kernel/csrc/kvcacheio/transfer.cu``). For host buffers, the raw host
    ``data_ptr()`` is only GPU-accessible on CUDA (thanks to UVA). On DCU/ROCm
    the host pages must be registered as mapped (see ``alloc_with_host_register``)
    and the GPU-side address obtained via ``hipHostGetDevicePointer``; otherwise
    the kernel faults (VMFault) when it dereferences an unmapped host address.

    Device tensors and non-DCU platforms return the plain ``data_ptr()``.
    """
    if not _is_dcu or tensor.is_cuda:
        return tensor.data_ptr()
    # DCU host tensor: translate the host pointer to a GPU-accessible one.
    # torch's cudart python binding does not expose HostGetDevicePointer, so we
    # call the HIP runtime directly via ctypes.
    try:
        return _hip_host_get_device_pointer(tensor.data_ptr())
    except RuntimeError as first_err:
        nbytes = tensor.numel() * tensor.element_size()
        reg_err = register_host_tensor_for_kernel_access(tensor, nbytes)
        if reg_err != 0:
            raise RuntimeError(
                "hipHostGetDevicePointer failed and retry registration failed: "
                f"shape={tuple(tensor.shape)}, nbytes={nbytes}, "
                f"host_ptr=0x{tensor.data_ptr():x}, register_error={reg_err}"
            ) from first_err
        try:
            return _hip_host_get_device_pointer(tensor.data_ptr())
        except RuntimeError as second_err:
            raise RuntimeError(
                "hipHostGetDevicePointer failed after mapped retry registration: "
                f"shape={tuple(tensor.shape)}, nbytes={nbytes}, "
                f"host_ptr=0x{tensor.data_ptr():x}"
            ) from second_err


_HIP_RT = None


def _hip_host_get_device_pointer(host_ptr: int) -> int:
    import ctypes

    global _HIP_RT
    if _HIP_RT is None:
        last_err = None
        for lib in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                _HIP_RT = ctypes.CDLL(lib)
                break
            except OSError as e:  # noqa: PERF203
                last_err = e
        if _HIP_RT is None:
            raise RuntimeError(
                f"Failed to load the HIP runtime for hipHostGetDevicePointer: {last_err}"
            )

    dev_ptr = ctypes.c_void_p()
    # hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags)
    err = _HIP_RT.hipHostGetDevicePointer(
        ctypes.byref(dev_ptr), ctypes.c_void_p(host_ptr), ctypes.c_uint(0)
    )
    if err != 0:
        raise RuntimeError(
            f"hipHostGetDevicePointer failed (hipError={err}) while building the "
            f"kernel hicache device-pointer table; the host buffer must be "
            f"registered with hipHostRegisterMapped (see alloc_with_host_register)."
        )
    return int(dev_ptr.value)


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self._get_physical_allocation_bytes()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if (
            requested_bytes > 0
            and requested_bytes > available_bytes
            and not _hugepage_enabled()
        ):
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    def _get_physical_allocation_bytes(self) -> int:
        """Bytes this rank will physically allocate for the host pool."""
        return self.size * self.size_per_token

    def _is_device_layer_sharded(self, device_pool=None) -> bool:
        device_pool = device_pool or self.device_pool
        return bool(getattr(device_pool, "layer_shard_enabled", False))

    def _is_device_layer_owned(self, device_pool, layer_id: int) -> bool:
        if not self._is_device_layer_sharded(device_pool):
            return True
        return device_pool._is_layer_owned(
            getattr(device_pool, "start_layer", 0) + layer_id
        )

    def _owned_device_layer_ids(self, device_pool) -> list[int]:
        layer_num = getattr(device_pool, "layer_num", self.layer_num)
        if not self._is_device_layer_sharded(device_pool):
            return list(range(layer_num))
        return [
            layer_id
            for layer_id in range(layer_num)
            if self._is_device_layer_owned(device_pool, layer_id)
        ]

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """
        Load KV data from the host memory pool to the device memory pool for a specific layer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        raise NotImplementedError()

    @synchronized
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])
        return len(indices)


class MHATokenToKVPoolHost(HostKVCache):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        self.element_dim = self.device_pool.head_num * self.device_pool.head_dim
        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(
            element_size=self.element_dim * self.dtype.itemsize
        )

        if self.layout == "page_first":
            # Transpose [page, layer, ...] -> [layer, page, ...] to get per-layer views
            # This swaps strides without copying data
            k_transposed = self.k_buffer.transpose(0, 1)
            v_transposed = self.v_buffer.transpose(0, 1)
            self.k_data_refs = [k_transposed[i] for i in range(self.layer_num)]
            self.v_data_refs = [v_transposed[i] for i in range(self.layer_num)]
        else:
            self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
            self.v_data_refs = [self.v_buffer[i] for i in range(self.layer_num)]
        self.k_data_ptrs = torch.tensor(
            [kernel_accessible_host_ptr(x) for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [kernel_accessible_host_ptr(x) for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self):
        return self.get_size_per_token() // 2

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                self.page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_buffer[layer_id],
                        v_cache_src=self.v_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer(
                        src_k=self.k_buffer[layer_id],
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer[layer_id],
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    # Transpose [page, layer, ...] -> [layer, page, ...] then
                    # index by layer_id to get a per-layer view with strided layout.
                    # The kernel handles different src/dst strides automatically.
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_data_refs[layer_id],
                        v_cache_src=self.v_data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer_pf_lf(
                        src_k=self.k_buffer,
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer,
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            elif self.layout == "page_head":
                transfer_kv_per_layer_ph_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],
                    dst_layers=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer, self.v_buffer],
                    dst_ptrs=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_from_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=self.k_buffer[layer_id],
                        v_cache_dst=self.v_buffer[layer_id],
                        k_cache_src=device_pool.k_buffer[layer_id],
                        v_cache_src=device_pool.v_buffer[layer_id],
                        indices_dst=host_indices,
                        indices_src=device_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer(
                        src_k=device_pool.k_buffer[layer_id],
                        dst_k=self.k_buffer[layer_id],
                        src_v=device_pool.v_buffer[layer_id],
                        dst_v=self.v_buffer[layer_id],
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=self.k_data_refs[layer_id],
                        v_cache_dst=self.v_data_refs[layer_id],
                        k_cache_src=device_pool.k_buffer[layer_id],
                        v_cache_src=device_pool.v_buffer[layer_id],
                        indices_dst=host_indices,
                        indices_src=device_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    raise ValueError(
                        "Layer-sharded MHA HiCache backup with page_first layout "
                        "requires the JIT one-layer kernel."
                    )
            else:
                raise ValueError(
                    f"Layer-sharded HiCache backup does not support layout: {self.layout}"
                )
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    dst_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(
                    f"Layer-sharded direct HiCache backup does not support layout: {self.layout}"
                )
        else:
            raise ValueError(
                f"Layer-sharded HiCache backup does not support IO backend: {io_backend}"
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if self._is_device_layer_sharded(device_pool):
            for layer_id in self._owned_device_layer_ids(device_pool):
                self._backup_from_device_per_layer(
                    device_pool, host_indices, device_indices, layer_id, io_backend
                )
            return

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_dst_stride_bytes=self.token_stride_size,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k_layers=self.k_data_ptrs,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v_layers=self.v_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    # Use transposed data ptrs so the kernel writes to
                    # [layer, page, item] view with stride layout_dim per token.
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        kv_cache_dst_stride_bytes=self.layout_dim,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_lf_pf(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k=self.k_buffer,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v=self.v_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_head":
                transfer_kv_all_layer_lf_ph(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k=self.k_buffer,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,
                    dst_layers=self.k_data_refs + self.v_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.k_buffer + device_pool.v_buffer,
                    dst_ptrs=[self.k_buffer, self.v_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :, :]
        elif self.layout in ["page_first_direct", "page_head"]:
            real_index = index // self.page_size
            data_page = self.kv_buffer[:, real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, :, index : index + self.page_size, :, :] = (
                data_page.reshape(
                    2,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            )
        elif self.layout == "page_first":
            self.kv_buffer[:, index : index + self.page_size, :, :, :] = (
                data_page.reshape(
                    2, self.page_size, self.layer_num, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.layer_num, self.page_size, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_head":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.head_num, self.page_size, self.layer_num, self.head_dim
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        """
        get meta data for zero copy of heterogeneous ranks' KVCache
        """
        assert self.layout == "page_head"
        assert len(indices) % self.page_size == 0
        assert self.head_num % split_factor == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        for index in range(0, len(indices), self.page_size):
            for head_id in range(0, self.head_num, self.head_num // split_factor):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                    + head_id
                    * self.page_size
                    * self.layer_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
            // split_factor
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                        + layer_id
                        * self.size
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                    )
                    v_ptr = k_ptr + v_offset
                    ptr_list.append(k_ptr)
                    ptr_list.append(v_ptr)
            element_size = (
                self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct", "page_head"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.head_num
                * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


class MLATokenToKVPoolHost(HostKVCache):
    device_pool: MLATokenToKVPool

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        override_kv_cache_dim: Optional[int] = None,
    ):
        self.override_kv_cache_dim = override_kv_cache_dim
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(
            element_size=self.kv_cache_dim * self.dtype.itemsize
        )

        if self.layout == "page_first" and self.can_use_jit:
            # Transpose [page, layer, ...] -> [layer, page, ...] to get per-layer views
            # This swaps strides without copying data
            transposed = self.kv_buffer.transpose(0, 1)
            self.data_refs = [transposed[i] for i in range(self.layer_num)]
        else:
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            self._build_data_pointer_values(self.data_refs),
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def _build_data_pointer_values(self, data_refs) -> list[int]:
        return [kernel_accessible_host_ptr(x) for x in data_refs]

    def get_contiguous_buf_infos(self):
        """Return (data_ptrs, data_lens, item_lens) in the same format as device pool,
        for registering host memory with the disaggregation transfer engine."""
        data_ptrs = [int(self.data_ptrs[i].item()) for i in range(self.layer_num)]
        data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.token_stride_size] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        self.kv_cache_dim = self.override_kv_cache_dim or (
            self.kv_lora_rank + self.qk_rope_head_dim
        )
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (
                self.layer_num,
                self.size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            dims = (
                self.size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        # Ascend-specific: Aligns with NPUMLATokenToKVPool layout
        # Separately allocate k_buffer and v_buffer for easier data transfer.
        elif self.layout == "page_first_kv_split":
            base_dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
            )
            alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
            self.k_buffer = alloc_func(
                (*base_dims, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.v_buffer = alloc_func(
                (*base_dims, self.qk_rope_head_dim),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_buffer = None
            if self.device_pool.index_head_dim is not None:
                self.index_k_buffer = alloc_func(
                    (*base_dims, self.device_pool.index_head_dim),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
            # Return k_buffer to preserve original kv_buffer and data_refs init logic,
            # though Ascend doesn't use these parameters.
            return self.k_buffer
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[layer_id],
                        cache_src=self.kv_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla(
                        src=self.kv_buffer[layer_id],
                        dst=device_pool.kv_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[layer_id],
                        cache_src=self.data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla_pf_lf(
                        src=self.kv_buffer,
                        dst=device_pool.kv_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.kv_buffer[layer_id]],
                    dst_layers=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        device_index_k=device_pool.index_k_buffer,
                        host_index_k=self.index_k_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_from_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=self.kv_buffer[layer_id],
                        cache_src=device_pool.kv_buffer[layer_id],
                        indices_dst=host_indices,
                        indices_src=device_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla(
                        src=device_pool.kv_buffer[layer_id],
                        dst=self.kv_buffer[layer_id],
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=self.data_refs[layer_id],
                        cache_src=device_pool.kv_buffer[layer_id],
                        indices_dst=host_indices,
                        indices_src=device_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    raise ValueError(
                        "Layer-sharded MLA HiCache backup with page_first layout "
                        "requires the JIT one-layer kernel."
                    )
            else:
                raise ValueError(
                    f"Layer-sharded HiCache backup does not support layout: {self.layout}"
                )
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[device_pool.kv_buffer[layer_id]],
                    dst_layers=[self.kv_buffer[layer_id]],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(
                    f"Layer-sharded direct HiCache backup does not support layout: {self.layout}"
                )
        else:
            raise ValueError(
                f"Layer-sharded HiCache backup does not support IO backend: {io_backend}"
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if self._is_device_layer_sharded(device_pool):
            for layer_id in self._owned_device_layer_ids(device_pool):
                self._backup_from_device_per_layer(
                    device_pool, host_indices, device_indices, layer_id, io_backend
                )
            return

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_pool.data_ptrs,
                        indices_src=device_indices,
                        cache_dst_stride_bytes=self.token_stride_size,
                        cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_mla(
                        src_layers=device_pool.data_ptrs,
                        dst_layers=self.data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_pool.data_ptrs,
                        indices_src=device_indices,
                        cache_src_stride_bytes=self.token_stride_size,
                        cache_dst_stride_bytes=self.layout_dim,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=device_pool.data_ptrs,
                        dst=self.kv_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.kv_buffer,
                    dst_layers=self.data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.kv_buffer,
                    dst_ptrs=[self.kv_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    device_index_k=device_pool.index_k_buffer,
                    host_index_k=self.index_k_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.kv_buffer[real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            ),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, index : index + self.page_size, :, :] = data_page.reshape(
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            self.kv_buffer[index : index + self.page_size, :, :, :] = data_page.reshape(
                self.page_size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                1,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index] * self.kv_cache_dim * self.dtype.itemsize
                        + layer_id * self.size * self.kv_cache_dim * self.dtype.itemsize
                    )
                    ptr_list.append(k_ptr)
            element_size = self.dtype.itemsize * self.page_size * self.kv_cache_dim
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.kv_cache_dim
                    * self.dtype.itemsize
                )
                ptr_list.append(k_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.kv_cache_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


class MambaPoolHost(HostKVCache):

    def __init__(
        self,
        device_pool: MambaPool,
        host_to_device_ratio: float,
        host_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        layout: str = "layer_first",
    ):
        self.device_pool = device_pool
        self.page_size = 1
        assert layout in [
            "page_first",
            "page_first_direct",
            "layer_first",
        ], f"Unsupported layout: {layout}"

        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.num_mamba_layers = device_pool.num_mamba_layers

        self.conv_state_shapes = [
            conv_state.shape[2:] for conv_state in device_pool.mamba_cache.conv
        ]
        self.temporal_state_shape = device_pool.mamba_cache.temporal.shape[2:]
        self.temporal_state_elem_size = int(np.prod(self.temporal_state_shape))
        self.conv_state_elem_sizes = [
            int(np.prod(conv_shape)) for conv_shape in self.conv_state_shapes
        ]
        self.conv_dtype = device_pool.mamba_cache.conv[0].dtype
        self.temporal_dtype = device_pool.mamba_cache.temporal.dtype
        self.dtype = self.conv_dtype
        self.size_per_token = self.get_size_per_token()

        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)

        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        logger.info(
            "Allocating %.2f GB host memory for hierarchical Mamba cache (layout=%s).",
            requested_bytes / 1e9,
            self.layout,
        )

        self.init_kv_buffer()
        self.lock = threading.RLock()
        self.clear()

    def init_kv_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]

        if self.layout in ["page_first", "page_first_direct"]:
            # page-first: (page_num, num_layers, 1, *shape) — per-page data is contiguous
            temporal_dims = (
                self.size,
                self.num_mamba_layers,
                1,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.size, self.num_mamba_layers, 1) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )
        else:
            # layer-first: (num_layers, size, *shape)
            temporal_dims = (
                self.num_mamba_layers,
                self.size,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.num_mamba_layers, self.size) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )

    def get_hybrid_pool_buffer(self):
        # Expose all mamba host tensors that need Mooncake buffer registration.
        return [self.temporal_buffer, *self.conv_buffer]

    def _iter_page_tensors(self, index: int):
        if self.layout in ["page_first", "page_first_direct"]:
            yield self.temporal_buffer[index]
            for conv_buf in self.conv_buffer:
                yield conv_buf[index]
        else:
            yield self.temporal_buffer[:, index : index + self.page_size]
            for conv_buf in self.conv_buffer:
                yield conv_buf[:, index : index + self.page_size]

    @staticmethod
    def _flatten_tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(torch.uint8).reshape(-1)

    @synchronized
    def clear(self):
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)

    def get_size_per_token(self):
        conv_total_size = sum(
            conv_elem_size * self.conv_dtype.itemsize
            for conv_elem_size in self.conv_state_elem_sizes
        )
        temporal_size = self.temporal_state_elem_size * self.temporal_dtype.itemsize
        return (conv_total_size + temporal_size) * self.num_mamba_layers

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    @staticmethod
    def _item_size_per_index(tensor: torch.Tensor) -> int:
        if tensor.shape[0] == 0:
            return 0
        return int(tensor[0].numel() * tensor.element_size())

    @staticmethod
    def _copy_tensor(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            # TODO: Rename the interface for clarity.
            # Here, transfer_kv_per_layer_mla is reused to transfer the Mamba state.
            # This has nothing to do with MLA; it's only reused because this interface happens to transfer a single Pool.
            transfer_kv_per_layer_mla(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=MambaPoolHost._item_size_per_index(src),
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=[src],
                dst_layers=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_pf_lf(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        layer_id: int,
        num_layers: int,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(dst)
            transfer_kv_per_layer_mla_pf_lf(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                item_size=item_size,
                src_layout_dim=item_size * num_layers,
            )
        elif io_backend == "direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[src],
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_all_layers_lf_pf(
        src_layers: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        num_layers: int,
        device: str,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(src_layers[0])
            src_ptrs = torch.tensor(
                [kernel_accessible_host_ptr(src_layers[i]) for i in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=src_ptrs,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=item_size,
                dst_layout_dim=item_size * num_layers,
                num_layers=num_layers,
            )
        elif io_backend == "direct":
            src_ptrs = [src_layers[i] for i in range(num_layers)]
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=src_ptrs,
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend="kernel",
    ):
        if self.layout in ["page_first", "page_first_direct"]:
            self._copy_tensor_pf_lf(
                src=self.temporal_buffer,
                dst=device_pool.mamba_cache.temporal[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                num_layers=self.num_mamba_layers,
                io_backend=io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_pf_lf(
                    src=self.conv_buffer[conv_idx],
                    dst=device_pool.mamba_cache.conv[conv_idx][layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                )
        else:
            self._copy_tensor(
                self.temporal_buffer[layer_id],
                device_pool.mamba_cache.temporal[layer_id],
                host_indices,
                device_indices,
                io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor(
                    self.conv_buffer[conv_idx][layer_id],
                    device_pool.mamba_cache.conv[conv_idx][layer_id],
                    host_indices,
                    device_indices,
                    io_backend,
                )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        if self.layout in ["page_first", "page_first_direct"]:
            self._copy_tensor_all_layers_lf_pf(
                src_layers=device_pool.mamba_cache.temporal,
                dst=self.temporal_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                num_layers=self.num_mamba_layers,
                device=self.device_pool.device,
                io_backend=io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_all_layers_lf_pf(
                    src_layers=device_pool.mamba_cache.conv[conv_idx],
                    dst=self.conv_buffer[conv_idx],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    num_layers=self.num_mamba_layers,
                    device=self.device_pool.device,
                    io_backend=io_backend,
                )
        else:
            for layer_id in range(self.num_mamba_layers):
                self._copy_tensor(
                    device_pool.mamba_cache.temporal[layer_id],
                    self.temporal_buffer[layer_id],
                    device_indices,
                    host_indices,
                    io_backend,
                )
                for conv_idx in range(len(self.conv_state_shapes)):
                    self._copy_tensor(
                        device_pool.mamba_cache.conv[conv_idx][layer_id],
                        self.conv_buffer[conv_idx][layer_id],
                        device_indices,
                        host_indices,
                        io_backend,
                    )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        data_page = torch.cat(
            [
                self._flatten_tensor_bytes(tensor)
                for tensor in self._iter_page_tensors(index)
            ]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.page_size * self.size_per_token,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(
        self,
        index: int,
        data_page: torch.Tensor,
    ) -> None:
        flat_bytes = data_page.contiguous().view(torch.uint8).reshape(-1)
        start = 0
        for tensor in self._iter_page_tensors(index):
            num_bytes = tensor.numel() * tensor.element_size()
            tensor_bytes = flat_bytes[start : start + num_bytes]
            start += num_bytes
            restored = tensor_bytes.view(dtype=tensor.dtype).reshape(tensor.shape)
            tensor.copy_(restored)

    def get_page_buffer_meta(self, indices):
        """Meta data for zero-copy storage I/O.

        Only page-first layouts are supported for mamba storage zero-copy because
        each page slot in temporal/conv buffers is directly addressable.
        """
        assert len(indices) % self.page_size == 0
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(
                f"Mamba storage zero-copy requires page_first layout, got {self.layout}"
            )
        indices = indices.tolist()
        ptr_list = []
        element_size_list = []

        # Compute base pointers once; each page pointer is offset from these bases.
        temporal_base_ptr = self.temporal_buffer.data_ptr()
        conv_base_ptrs = [buf.data_ptr() for buf in self.conv_buffer]
        # Component sizes are constant across pages, so precompute once as well.
        temporal_element_size = (
            self.page_size
            * self.num_mamba_layers
            * self.temporal_dtype.itemsize
            * self.temporal_state_elem_size
        )
        conv_element_sizes = [
            (
                self.page_size
                * self.num_mamba_layers
                * self.conv_dtype.itemsize
                * self.conv_state_elem_sizes[i]
            )
            for i in range(len(self.conv_state_shapes))
        ]

        for i in range(0, len(indices), self.page_size):
            # Emit component pointers in stable order:
            # temporal first, then conv_0..conv_n for this page.
            temporal_ptr = (
                temporal_base_ptr
                + indices[i]
                * self.num_mamba_layers
                * self.temporal_state_elem_size
                * self.temporal_dtype.itemsize
            )
            ptr_list.append(temporal_ptr)
            element_size_list.append(temporal_element_size)
            for j in range(len(self.conv_buffer)):
                conv_ptr = (
                    conv_base_ptrs[j]
                    + indices[i]
                    * self.num_mamba_layers
                    * self.conv_state_elem_sizes[j]
                    * self.conv_dtype.itemsize
                )
                ptr_list.append(conv_ptr)
                element_size_list.append(conv_element_sizes[j])
        return ptr_list, element_size_list


# ---- V4 Compressed KV Host Pools ----


class LogicalHostPool:
    """Pure-logical anchor pool for V4 HiCache.

    The pool manages page-aligned token slots but holds no KV tensor. V4
    compressed side pools use these logical FULL indices as stable page anchors.
    """

    def __init__(self, size: int, page_size: int):
        if size % page_size != 0:
            raise ValueError(
                "LogicalHostPool size must be page-aligned, "
                f"got size={size}, page_size={page_size}"
            )
        self.size = size
        self.page_size = page_size
        self.device = "cpu"
        self.layout = "layer_first"
        self.dtype = torch.uint8
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.kv_buffer = None
        self.size_per_token = 0
        self.allocator = None
        self.lock = threading.RLock()
        self.clear()

    @synchronized
    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool allocation must be page-aligned, "
                f"got need_size={need_size}, page_size={self.page_size}"
            )
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        if len(indices) % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool free must be page-aligned, "
                f"got len(indices)={len(indices)}, page_size={self.page_size}"
            )
        self.free_slots = torch.cat(
            [self.free_slots, indices.to(dtype=torch.int64, device="cpu").flatten()]
        )
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        pass

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        pass

    def get_data_page(self, index, flat=True):
        return torch.empty(0, dtype=torch.uint8)

    def get_dummy_flat_data_page(self):
        return torch.empty(0, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, data_page):
        pass

    def get_page_buffer_meta(self, indices):
        return None

    def get_ksize_per_token(self):
        return 0


class DeepSeekV4PagedHostPool(HostKVCache):
    """Host mirror for a DeepSeek V4 paged KV/indexer sub-pool."""

    def __init__(
        self,
        pool_name: str,
        device_buffers: list[torch.Tensor],
        item_bytes: int,
        num_host_pages: int,
        slot_page_size: int,
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        self.pool_name = pool_name
        self.layer_num = len(device_buffers)
        self.item_bytes = item_bytes
        self.num_host_pages = num_host_pages
        self.slot_page_size = slot_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = slot_page_size
        self.size = num_host_pages * slot_page_size
        self.layout = "layer_first"
        self.size_per_token = item_bytes
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.device_buffers = device_buffers
        self.gpu_device = device_buffers[0].device if device_buffers else device

        requested_bytes = self.layer_num * num_host_pages * self.item_bytes
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 paged pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.kv_buffer = [
            alloc_func(
                (num_host_pages, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            for _ in range(self.layer_num)
        ]
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]

        logger.info(
            "Allocating %.2f GB host memory for V4 paged pool '%s' "
            "(layers=%d, pages=%d, item_bytes=%d).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.item_bytes,
        )
        self.clear()

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() % self.slot_page_size != 0:
            raise ValueError(
                f"{self.pool_name} transfer indices must be page-aligned, "
                f"got numel={indices.numel()}, slot_page_size={self.slot_page_size}"
            )
        return indices.reshape(-1, self.slot_page_size)[:, 0] // self.slot_page_size

    def _check_io_backend(self, io_backend: str) -> None:
        if io_backend != "direct":
            raise NotImplementedError(
                f"{self.pool_name} supports only direct io_backend, got {io_backend}"
            )

    def get_size_per_token(self):
        return self.item_bytes

    def get_ksize_per_token(self):
        return self.item_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer

    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        need_size = (
            (need_size + self.slot_page_size - 1) // self.slot_page_size
        ) * self.slot_page_size
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat(
            [self.free_slots, indices.to(dtype=torch.int64, device="cpu").flatten()]
        )
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        self._check_io_backend(io_backend)
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        transfer_kv_direct(
            src_layers=self.device_buffers,
            dst_layers=self.data_refs,
            src_indices=device_rows,
            dst_indices=host_rows,
            page_size=1,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        self._check_io_backend(io_backend)
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        transfer_kv_direct(
            src_layers=[self.kv_buffer[layer_id]],
            dst_layers=[self.device_buffers[layer_id]],
            src_indices=host_rows,
            dst_indices=device_rows,
            page_size=1,
        )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.slot_page_size
        data_page = torch.stack(
            [self.kv_buffer[i][index] for i in range(self.layer_num)]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.slot_page_size
        data = data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
        for i in range(self.layer_num):
            self.kv_buffer[i][index].copy_(data[i])

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        for row in rows:
            for layer_id in range(self.layer_num):
                ptr = (
                    self.kv_buffer[layer_id].data_ptr()
                    + int(row) * self.item_bytes * self.dtype.itemsize
                )
                ptr_list.append(ptr)
        element_size = self.item_bytes * self.dtype.itemsize
        return ptr_list, [element_size] * len(ptr_list)


class DeepSeekV4StateHostPool(HostKVCache):
    """Host pool for V4 CompressStatePool page rows."""

    def __init__(
        self,
        pool_name: str,
        state_pools: list,
        num_host_pages: int,
        swa_page_size: int,
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        if any(pool is None for pool in state_pools):
            raise ValueError(f"{pool_name} state_pools must not contain None")

        self.pool_name = pool_name
        self.state_pools = state_pools
        self.layer_num = len(state_pools)
        self.num_host_pages = num_host_pages
        self.swa_page_size = swa_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = swa_page_size
        self.size = num_host_pages * swa_page_size
        self.layout = "layer_first"
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.ring_size = 0
        self.state_page_bytes = 0
        self.device_page_views = []
        self.gpu_device = device
        self._init_device_page_views()
        self.size_per_token = self.state_page_bytes

        requested_bytes = self.layer_num * num_host_pages * self.state_page_bytes
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 state pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.kv_buffer = [
            alloc_func(
                (num_host_pages, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            for _ in range(self.layer_num)
        ]
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        logger.info(
            "Allocating %.2f GB host memory for V4 state pool '%s' "
            "(layers=%d, pages=%d, state_page_bytes=%d).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.state_page_bytes,
        )

    def _init_device_page_views(self) -> None:
        expected_ring_size = None
        expected_state_page_bytes = None
        for pool in self.state_pools:
            state_tensor = pool.kv_score_buffer.kv_score
            if not state_tensor.is_contiguous():
                raise ValueError(f"{self.pool_name} state tensor must be contiguous")
            ring_size = pool.ring_size
            slot_bytes = state_tensor[0].nbytes
            state_page_bytes = ring_size * slot_bytes
            if expected_ring_size is None:
                expected_ring_size = ring_size
                expected_state_page_bytes = state_page_bytes
                self.gpu_device = state_tensor.device
            elif (
                expected_ring_size != ring_size
                or expected_state_page_bytes != state_page_bytes
            ):
                raise ValueError(
                    f"{self.pool_name} state pools must share ring size and slot bytes"
                )

            state_bytes = state_tensor.view(torch.uint8).reshape(
                state_tensor.shape[0], -1
            )
            usable_slots = (state_tensor.shape[0] // ring_size) * ring_size
            self.device_page_views.append(
                state_bytes[:usable_slots].reshape(-1, state_page_bytes)
            )

        self.ring_size = expected_ring_size or 0
        self.state_page_bytes = expected_state_page_bytes or 0

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() % self.swa_page_size != 0:
            raise ValueError(
                f"{self.pool_name} transfer indices must be SWA-page-aligned, "
                f"got numel={indices.numel()}, swa_page_size={self.swa_page_size}"
            )
        return indices.reshape(-1, self.swa_page_size)[:, 0] // self.swa_page_size

    def _check_io_backend(self, io_backend: str) -> None:
        if io_backend != "direct":
            raise NotImplementedError(
                f"{self.pool_name} supports only direct io_backend, got {io_backend}"
            )

    def get_size_per_token(self):
        return self.state_page_bytes

    def get_ksize_per_token(self):
        return self.state_page_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer

    def clear(self):
        pass

    def available_size(self):
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no free list"
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        self._check_io_backend(io_backend)
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        transfer_kv_direct(
            src_layers=self.device_page_views,
            dst_layers=self.data_refs,
            src_indices=device_rows,
            dst_indices=host_rows,
            page_size=1,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        self._check_io_backend(io_backend)
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        transfer_kv_direct(
            src_layers=[self.kv_buffer[layer_id]],
            dst_layers=[self.device_page_views[layer_id]],
            src_indices=host_rows,
            dst_indices=device_rows,
            page_size=1,
        )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.swa_page_size
        data_page = torch.stack(
            [self.kv_buffer[i][index] for i in range(self.layer_num)]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.swa_page_size
        data = data_page.view(self.dtype).reshape(self.layer_num, self.state_page_bytes)
        for i in range(self.layer_num):
            self.kv_buffer[i][index].copy_(data[i])

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        for row in rows:
            for layer_id in range(self.layer_num):
                ptr = (
                    self.kv_buffer[layer_id].data_ptr()
                    + int(row) * self.state_page_bytes * self.dtype.itemsize
                )
                ptr_list.append(ptr)
        element_size = self.state_page_bytes * self.dtype.itemsize
        return ptr_list, [element_size] * len(ptr_list)


@dataclass
class PoolEntry:
    name: PoolName
    host_pool: Any
    device_pool: Any
    layer_mapper: Callable[[int], Optional[int]]
    is_primary_index_anchor: bool = False
    # Optional eviction callbacks for auto-alloc in HybridCacheController.
    # host_evict_fn(n): evict n slots from the host pool (used by write()).
    # device_evict_fn(n): evict n slots from the device pool (used by load()).
    host_evict_fn: Optional[Callable] = None
    device_evict_fn: Optional[Callable] = None
    # Optional alloc/free overrides for the device side, used by
    # _resolve_pool_transfers_allocation. Set when entry.device_pool is the
    # raw KV pool (layout) rather than an allocator (e.g. SWA, where alloc
    # lives on a separate sub-allocator inside SWATokenToKVPoolAllocator).
    # When None, fall back to entry.device_pool.alloc/free.
    device_alloc_fn: Optional[Callable] = None
    device_free_fn: Optional[Callable] = None


class HostPoolGroup:
    def __init__(self, entries: list[PoolEntry]):
        if not entries:
            raise ValueError("HostPoolGroup requires at least one pool entry.")
        self.entries = entries
        self.entry_map = {entry.name: entry for entry in entries}
        self.anchor_entry = next(
            (entry for entry in entries if entry.is_primary_index_anchor),
            entries[0],
        )

        self.layout = self.anchor_entry.host_pool.layout
        self.page_size = self.anchor_entry.host_pool.page_size
        self.device = self.anchor_entry.host_pool.device
        self.size = self.anchor_entry.host_pool.size

    @property
    def kv_buffer(self):
        return self.anchor_entry.host_pool.kv_buffer

    @property
    def size_per_token(self):
        return self.anchor_entry.host_pool.size_per_token

    @property
    def allocator(self):
        return self.anchor_entry.host_pool.allocator

    @property
    def dtype(self):
        return self.anchor_entry.host_pool.dtype

    @property
    def start_layer(self):
        return self.anchor_entry.host_pool.start_layer

    @property
    def end_layer(self):
        return self.anchor_entry.host_pool.end_layer

    def get_ksize_per_token(self):
        return self.anchor_entry.host_pool.get_ksize_per_token()

    def get_pool(self, name: PoolName):
        return self.entry_map[name].host_pool

    def get_page_buffer_meta(self, indices):
        return self.anchor_entry.host_pool.get_page_buffer_meta(indices)

    def clear(self) -> None:
        for entry in self.entries:
            entry.host_pool.clear()

    def available_size(self):
        return self.anchor_entry.host_pool.available_size()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        return self.anchor_entry.host_pool.alloc(need_size)

    def free(self, indices: torch.Tensor) -> int:
        return self.anchor_entry.host_pool.free(indices)

    def get_data_page(self, index, flat: bool = True):
        return self.anchor_entry.host_pool.get_data_page(index, flat)

    def get_dummy_flat_data_page(self):
        return self.anchor_entry.host_pool.get_dummy_flat_data_page()

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        return self.anchor_entry.host_pool.set_from_flat_data_page(index, data_page)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 1. Anchor (KV) transfer
        anchor = self.anchor_entry
        local_layer_id = anchor.layer_mapper(layer_id)
        if local_layer_id is not None and host_indices.numel() > 0:
            anchor.host_pool.load_to_device_per_layer(
                anchor.device_pool,
                host_indices,
                device_indices,
                local_layer_id,
                io_backend,
            )

        # 2. Extra pool transfers
        for transfer in pool_transfers or []:
            entry = self.entry_map.get(transfer.name)
            if entry is None or transfer.host_indices is None:
                continue
            local_layer_id = entry.layer_mapper(layer_id)
            if local_layer_id is None:
                continue
            entry.host_pool.load_to_device_per_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                local_layer_id,
                io_backend,
            )

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 1. Anchor (KV) backup
        self.anchor_entry.host_pool.backup_from_device_all_layer(
            self.anchor_entry.device_pool,
            host_indices,
            device_indices,
            io_backend,
        )
        # 2. Extra pool backup
        for transfer in pool_transfers or []:
            entry = self.entry_map.get(transfer.name)
            if entry is None or transfer.host_indices is None:
                continue
            entry.host_pool.backup_from_device_all_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                io_backend,
            )


class NSAIndexerPoolHost(HostKVCache):
    """Host-side NSA index buffers only. Slot layout matches the anchor MLA host pool."""

    device_pool: NSATokenToKVPool

    def __init__(
        self,
        device_pool: NSATokenToKVPool,
        anchor_host: MLATokenToKVPoolHost,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = anchor_host.page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = device_pool.layer_num
        self.use_fp8 = device_pool.use_fp8_index_k_cache
        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.indexer_dtype = NSATokenToKVPool.index_k_with_scale_buffer_dtype
        if self.use_fp8:
            self.indexer_size_per_token = (
                self.index_head_dim
                + self.index_head_dim // self.indexer_quant_block_size * 4
            )
        else:
            self.indexer_size_per_token = self._infer_bf16_indexer_size_per_token(
                device_pool
            )

        self.size = anchor_host.size
        self.page_num = anchor_host.page_num

        self.indexer_page_stride_size = (
            self.indexer_size_per_token
            * self.page_size
            * self.indexer_dtype.itemsize
        )
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self.size_per_token = (
            self.indexer_size_per_token
            * self.layer_num
            * self.indexer_dtype.itemsize
        )

        buf_elem_size = self.page_num * self.layer_num * self.indexer_page_stride_size
        requested_bytes = buf_elem_size * self.indexer_dtype.itemsize
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for NSA indexer hierarchical cache. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        logger.info(
            "Allocating %.2f GB host memory for NSA indexer (layout=%s).",
            requested_bytes / 1e9,
            layout,
        )
        self.init_kv_buffer()
        self.lock = threading.RLock()
        self.clear()

    def _infer_bf16_indexer_size_per_token(self, device_pool: NSATokenToKVPool) -> int:
        for buf in device_pool.index_k_buffer:
            if buf is not None and buf.shape[0] > 0:
                return buf[0].nbytes // self.page_size

        dtype = getattr(device_pool, "index_k_buffer_dtype", torch.bfloat16)
        elem_size = torch.empty((), dtype=dtype).element_size()
        return self.index_head_dim * elem_size

    def _get_device_index_k_cache_for_transfer(self, device_pool):
        if self.use_fp8:
            return device_pool.index_k_with_scale_buffer
        row_width = self.page_size * self.indexer_size_per_token
        return [
            buf.view(torch.uint8).view(buf.shape[0], row_width)
            for buf in device_pool.index_k_buffer
        ]

    def get_size_per_token(self):
        return (
            self.indexer_size_per_token
            * self.layer_num
            * self.indexer_dtype.itemsize
        )

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        device_index_k_cache = self._get_device_index_k_cache_for_transfer(
            self.device_pool
        )
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in device_index_k_cache],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer = alloc_func(
                (self.layer_num, self.indexer_page_num, self.indexer_page_stride_size),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_data_refs = [
                self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
            ]
            self.index_k_data_ptrs = torch.tensor(
                [kernel_accessible_host_ptr(x) for x in self.index_k_data_refs],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer = alloc_func(
                (
                    self.indexer_page_num,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                ),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_hybrid_pool_buffer(self):
        return [self.index_k_with_scale_buffer]

    def _get_indexer_page_indices(self, host_indices, device_indices):
        if host_indices.numel() == 0:
            return host_indices, device_indices
        if host_indices.numel() % self.page_size != 0:
            raise ValueError(
                "Index buffer transfer expects page-aligned indices for NSA."
            )
        host_page_indices = (
            host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        device_page_indices = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        return host_page_indices, device_page_indices

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        device_index_k_cache = self._get_device_index_k_cache_for_transfer(device_pool)
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[layer_id],
                    dst=device_index_k_cache[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=device_index_k_cache[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    item_size=self.indexer_page_stride_size,
                    src_layout_dim=self.indexer_layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.index_k_with_scale_buffer[layer_id]],
                    dst_layers=[device_index_k_cache[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[device_index_k_cache[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_indexer_from_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        device_index_k_cache = self._get_device_index_k_cache_for_transfer(device_pool)
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=device_index_k_cache[layer_id],
                    dst=self.index_k_with_scale_buffer[layer_id],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                raise ValueError(
                    "Layer-sharded NSA indexer HiCache backup with page_first "
                    "layout is not supported without a per-layer LF->PF kernel."
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[device_index_k_cache[layer_id]],
                    dst_layers=[self.index_k_with_scale_buffer[layer_id]],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(
                    f"Layer-sharded direct NSA indexer backup does not support layout: {self.layout}"
                )
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if self._is_device_layer_sharded(device_pool):
            for layer_id in self._owned_device_layer_ids(device_pool):
                self._backup_indexer_from_device_per_layer(
                    device_pool, host_indices, device_indices, layer_id, io_backend
                )
            return

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        device_index_k_cache = self._get_device_index_k_cache_for_transfer(device_pool)
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=self.index_k_device_ptrs,
                    dst_layers=self.index_k_data_ptrs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.index_k_device_ptrs,
                    dst=self.index_k_with_scale_buffer,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    dst_layout_dim=self.indexer_layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_index_k_cache,
                    dst_layers=self.index_k_data_refs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_index_k_cache,
                    dst_ptrs=[self.index_k_with_scale_buffer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        page_idx = int(index) // self.page_size
        if self.layout == "layer_first":
            data_page = self.index_k_with_scale_buffer[:, page_idx : page_idx + 1, :]
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.index_k_with_scale_buffer[page_idx : page_idx + 1, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (self.layer_num, self.indexer_page_stride_size),
            dtype=self.indexer_dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page_idx = int(index) // self.page_size
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer[:, page_idx : page_idx + 1, :] = (
                data_page.reshape(
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                )
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer[page_idx : page_idx + 1, :, :, :] = (
                data_page.reshape(
                    1,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """Meta data for zero-copy storage I/O."""
        assert len(indices) % self.page_size == 0
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(f"Unsupported layout: {self.layout}")
        ptr_list = []
        indices = indices.tolist()
        page_stride_bytes = (
            self.layer_num * self.indexer_page_stride_size * self.indexer_dtype.itemsize
        )
        base_ptr = self.index_k_with_scale_buffer.data_ptr()
        for i in range(0, len(indices), self.page_size):
            page_index = int(indices[i]) // self.page_size
            ptr_list.append(base_ptr + page_index * page_stride_bytes)
        return ptr_list, [page_stride_bytes] * len(ptr_list)


class NSATokenToKVPoolHost(MLATokenToKVPoolHost):
    """Host-side NSA/DSA KV pool: MLA latent KV **plus** the NSA indexer.

    GLM5-Next (DSA + KDA) uses an NSA (MLA-family) full-attention KV pool whose
    device buffer is ``kv_cache_dim`` wide (on DCU/fp8 this is
    kv_lora_rank + fp8 scale + FlashMLA rope padding, e.g. 656) and additionally
    carries a separate NSA indexer buffer per token. Its HiCache full-attention
    host pool must therefore back up / load *both* the latent KV and the indexer
    for every token; otherwise:
      * a plain MLATokenToKVPoolHost sizes host slots at kv_lora_rank +
        qk_rope_head_dim (e.g. 512) and the device<->host transfer op fails with
        a shape mismatch ("size of tensor a (512) must match tensor b (656)"),
      * and even if sizes matched, KV restored from host would be missing its
        indexer, so NSA attention would read stale/zero indexer state.

    We fold the indexer into a single host pool (mirroring NV's validated
    GLM5-Next path) but reuse the already-validated DCU ``NSAIndexerPoolHost``
    (which handles the DCU-specific fp8/bf16 index-K storage) as the indexer
    engine via composition. The main latent KV is handled entirely by the
    MLATokenToKVPoolHost base class, sized with override_kv_cache_dim taken
    directly from the device pool so it always matches (incl. FlashMLA padding).
    """

    device_pool: NSATokenToKVPool

    def _init_indexer_geometry(
        self, device_pool: NSATokenToKVPool, page_size: int
    ) -> None:
        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.use_fp8_index_k_cache = device_pool.use_fp8_index_k_cache
        self.indexer_dtype = NSATokenToKVPool.index_k_with_scale_buffer_dtype
        if self.use_fp8_index_k_cache:
            self.indexer_size_per_token = (
                self.index_head_dim
                + self.index_head_dim // self.indexer_quant_block_size * 4
            )
            self.indexer_page_slots = page_size
        else:
            elem_size = torch.empty(
                (), dtype=device_pool.index_k_buffer_dtype
            ).element_size()
            self.indexer_size_per_token = self.index_head_dim * elem_size
            self.indexer_page_slots = page_size

    def __init__(
        self,
        device_pool: NSATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        # HostKVCache.__init__ asks get_size_per_token() before allocating any
        # buffers. Account for the NSA index cache in --hicache-size up front.
        self._init_indexer_geometry(device_pool, page_size)
        # Main latent KV host buffer. device_pool.kv_cache_dim already accounts
        # for DCU FlashMLA rope padding + fp8 scale, so forward it as override so
        # the host buffer width always matches the device buffer width.
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            override_kv_cache_dim=device_pool.kv_cache_dim,
        )
        # Indexer host buffer + transfer logic. The sidecar shares this pool's
        # host slot layout (same size / page_num), so it consumes the same
        # host_indices we allocate for the main KV.
        if getattr(self, "_skip_nsa_indexer_host", False):
            self.indexer_host = None
        else:
            self.indexer_host = NSAIndexerPoolHost(
                device_pool,
                self,
                layout,
                pin_memory=pin_memory,
                device=device,
                allocator_type=allocator_type,
            )

    # `pool_transfers` is forwarded by `HybridCacheController` to keep the call
    # signature uniform with `HostPoolGroup`, which dispatches per-pool
    # transfers for sibling pools. This pool only handles its own KV /
    # indexer, so the kwarg is accepted-but-ignored here.
    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )
        if self.indexer_host is not None:
            self.indexer_host.load_to_device_per_layer(
                device_pool, host_indices, device_indices, layer_id, io_backend
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend,
    ):
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        if self.indexer_host is not None:
            self.indexer_host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend
            )

    def get_size_per_token(self):
        base = super().get_size_per_token()
        return (
            base
            + self.indexer_size_per_token
            * self.layer_num
            * self.indexer_dtype.itemsize
            * self.indexer_page_slots
            // self.page_size
        )

    def clear(self):
        super().clear()
        # indexer_host is created after super().__init__ (which calls clear()),
        # so guard against the first call during base construction.
        if getattr(self, "indexer_host", None) is not None:
            self.indexer_host.clear()


class NSATokenToKVPoolHostShared(NSATokenToKVPoolHost):
    """NSA host cache backed by shared memory across tensor-parallel ranks."""

    def __init__(
        self,
        device_pool: NSATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        logger.info(
            "Using NSATokenToKVPoolHostShared for zero-copy shared host cache (NSA)."
        )

        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.use_fp8_index_k_cache = device_pool.use_fp8_index_k_cache
        self.indexer_dtype = NSATokenToKVPool.index_k_with_scale_buffer_dtype
        if self.use_fp8_index_k_cache:
            self.indexer_size_per_token = (
                self.index_head_dim
                + self.index_head_dim // self.indexer_quant_block_size * 4
            )
            self.indexer_page_slots = page_size
        else:
            elem_size = torch.empty(
                (), dtype=device_pool.index_k_buffer_dtype
            ).element_size()
            self.indexer_size_per_token = self.index_head_dim * elem_size
            self.indexer_page_slots = page_size

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = tp_group
        self._shared_mmap_refs = []
        self._skip_nsa_indexer_host = True

        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )

        self._init_indexer_buffers()

        self.index_data_refs = [
            self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
        ]
        self.index_data_ptrs = torch.tensor(
            [kernel_accessible_host_ptr(x) for x in self.index_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [kernel_accessible_host_ptr(x) for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def _init_indexer_buffers(self):
        index_buffer_second_dim = (
            self.indexer_page_slots * self.indexer_size_per_token
        )
        self.index_stride_size = (
            self.indexer_size_per_token
        ) * self.indexer_dtype.itemsize

        index_dims = (self.layer_num, self.page_num, index_buffer_second_dim)

        full_index_buffer = self._allocate_shared_buffer(
            "index", index_dims, self.indexer_dtype
        )
        self._shared_mmap_refs.append(full_index_buffer)

        self.index_k_with_scale_buffer = [
            full_index_buffer[i] for i in range(self.layer_num)
        ]

        self.index_k_data_refs = [
            self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
        ]
        self.index_k_data_ptrs = torch.tensor(
            [kernel_accessible_host_ptr(x) for x in self.index_k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.index_k_device_ptrs = torch.tensor(
            self.device_pool.index_k_device_ptrs_for_host_transfer(),
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def _get_physical_allocation_bytes(self) -> int:
        # Every rank maps the complete pool, but rank 0 alone creates the two
        # backing files. Non-zero ranks must not compare the machine-total pool
        # size against memory that rank 0 may already have allocated.
        return super()._get_physical_allocation_bytes() if self.tp_rank == 0 else 0

    def _get_index_device_cache_for_transfer(self, device_pool):
        if self.use_fp8_index_k_cache:
            return device_pool.index_k_with_scale_buffer
        row_width = self.page_size * self.indexer_size_per_token
        return [
            buf.view(torch.uint8).view(buf.shape[0], row_width)
            for buf in device_pool.index_k_buffer
        ]

    def _allocate_shared_buffer(self, name_suffix: str, shape: tuple, dtype: torch.dtype):
        numel = 1
        for d in shape:
            numel *= d
        total_bytes = numel * dtype.itemsize
        size_gb = total_bytes / (1024**3)

        t_start_all = time.perf_counter()

        if self.tp_rank == 0:
            logger.info(
                f"[{name_suffix}] Allocating shared memory buffer: {size_gb:.2f} GB"
            )

        shared_filename = None
        if self.tp_rank == 0:
            t_file_start = time.perf_counter()
            unique_id = str(uuid.uuid4())
            shared_filename = f"/dev/shm/sglang_nsa_{name_suffix}_{unique_id}.bin"
            with open(shared_filename, "wb") as f:
                try:
                    os.posix_fallocate(f.fileno(), 0, total_bytes)
                except (AttributeError, OSError):
                    f.truncate(total_bytes)
            logger.info(
                f"[{name_suffix}] Rank 0 created shared file in "
                f"{time.perf_counter() - t_file_start:.3f}s"
            )

        object_list = [shared_filename]
        dist.broadcast_object_list(object_list, src=0, group=self.tp_group)
        shared_filename = object_list[0]

        dist.barrier(group=self.tp_group)

        try:
            t_map_start = time.perf_counter()
            flat_tensor = torch.from_file(
                shared_filename,
                shared=True,
                size=numel,
                dtype=dtype,
                device="cpu",
            )
            if self.tp_rank == 0:
                logger.info(
                    f"[{name_suffix}] Memory mapping took "
                    f"{time.perf_counter() - t_map_start:.3f}s"
                )

            t_zero_start = time.perf_counter()

            chunk_size = numel // self.tp_size
            start_idx = self.tp_rank * chunk_size
            end_idx = (
                numel if self.tp_rank == self.tp_size - 1 else start_idx + chunk_size
            )

            flat_tensor[start_idx:end_idx].zero_()

            t_zero_end = time.perf_counter()
            logger.info(
                f"[{name_suffix}] Rank {self.tp_rank} finished zeroing chunk in "
                f"{t_zero_end - t_zero_start:.3f}s"
            )

            dist.barrier(group=self.tp_group)
            if self.tp_rank == 0:
                logger.info(
                    f"[{name_suffix}] Parallel page faulting (all ranks) completed "
                    f"in {time.perf_counter() - t_zero_start:.3f}s"
                )

            buffer = flat_tensor.view(shape)

            if self.pin_memory and (_is_cuda or _is_dcu):
                t_pin_start = time.perf_counter()
                err = register_host_tensor_for_kernel_access(buffer, total_bytes)
                if err != 0:
                    logger.warning(
                        f"Failed to pin shared memory for {name_suffix}. "
                        f"Error code: {err}"
                    )
                t_pin_end = time.perf_counter()
                if self.tp_rank == 0:
                    logger.info(
                        f"[{name_suffix}] cudaHostRegister took "
                        f"{t_pin_end - t_pin_start:.3f}s"
                    )

            if self.tp_rank == 0:
                logger.info(
                    f"[{name_suffix}] Total allocation pipeline took "
                    f"{time.perf_counter() - t_start_all:.3f}s"
                )

            return buffer
        finally:
            dist.barrier(group=self.tp_group)
            if self.tp_rank == 0 and os.path.exists(shared_filename):
                os.remove(shared_filename)

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            kv_dims = (
                self.layer_num,
                self.size,
                1,
                self.kv_cache_dim,
            )
        else:
            raise ValueError(
                f"Shared pool currently only supports layer_first layout, got {self.layout}"
            )

        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        self.kv_buffer = self._allocate_shared_buffer("kv", kv_dims, self.dtype)

        return self.kv_buffer

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        pool_transfers=None,
    ):
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )
        page_indices_host = host_indices[:: self.page_size] // self.page_size
        page_indices_device = device_indices[:: self.page_size] // self.page_size
        item_size = self.index_stride_size * self.indexer_page_slots
        device_index_k_cache = self._get_index_device_cache_for_transfer(device_pool)

        if io_backend == "kernel":
            transfer_kv_per_layer_mla(
                src=self.index_k_with_scale_buffer[layer_id],
                dst=device_index_k_cache[layer_id],
                src_indices=page_indices_host,
                dst_indices=page_indices_device,
                item_size=item_size,
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=[self.index_k_with_scale_buffer[layer_id]],
                dst_layers=[device_index_k_cache[layer_id]],
                src_indices=page_indices_host,
                dst_indices=page_indices_device,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported IO backend for NSA indexer: {io_backend}")

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers=None,
    ) -> None:
        if self.tp_rank == 0:
            super().backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend
            )
            page_indices_host = host_indices[:: self.page_size] // self.page_size
            page_indices_device = device_indices[:: self.page_size] // self.page_size
            item_size = self.index_stride_size * self.indexer_page_slots

            if io_backend == "kernel":
                transfer_kv_all_layer_mla(
                    src_layers=self.index_k_device_ptrs,
                    dst_layers=self.index_data_ptrs,
                    src_indices=page_indices_device,
                    dst_indices=page_indices_host,
                    item_size=item_size,
                    num_layers=self.layer_num,
                )
            elif io_backend == "direct":
                device_index_k_cache = self._get_index_device_cache_for_transfer(
                    device_pool
                )
                transfer_kv_direct(
                    src_layers=device_index_k_cache,
                    dst_layers=self.index_k_with_scale_buffer,
                    src_indices=page_indices_device,
                    dst_indices=page_indices_host,
                    page_size=1,
                )
            else:
                raise ValueError(
                    f"Unsupported IO backend for NSA indexer: {io_backend}"
                )


class NSATokenToKVPoolHostSharedLayerGroup(NSATokenToKVPoolHost):
    """NSA host cache backed by per-rank shared-memory layer shards."""

    def __init__(
        self,
        device_pool: NSATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        logger.info(
            "Using NSATokenToKVPoolHostSharedLayerGroup (Per-Rank Distributed Shards) "
            "for zero-copy host cache (NSA)."
        )

        if allocator_type not in (None, "default"):
            raise ValueError(
                "NSA shared layer-group HiCache does not support an L3/storage "
                f"backend yet, got allocator_type={allocator_type!r}."
            )

        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.use_fp8_index_k_cache = device_pool.use_fp8_index_k_cache
        self.indexer_dtype = NSATokenToKVPool.index_k_with_scale_buffer_dtype
        if self.use_fp8_index_k_cache:
            self.indexer_size_per_token = (
                self.index_head_dim
                + self.index_head_dim // self.indexer_quant_block_size * 4
            )
            self.indexer_page_slots = page_size
        else:
            elem_size = torch.empty(
                (), dtype=device_pool.index_k_buffer_dtype
            ).element_size()
            self.indexer_size_per_token = self.index_head_dim * elem_size
            self.indexer_page_slots = page_size

        if tp_group is None:
            raise ValueError(
                "NSA shared layer-group HiCache requires an attention CP group."
            )
        self.tp_group = tp_group
        self.tp_rank = dist.get_rank(group=self.tp_group)
        self.tp_size = dist.get_world_size(group=self.tp_group)

        if getattr(device_pool, "layer_shard_enabled", False) and (
            device_pool.layer_shard_size != self.tp_size
            or device_pool.layer_shard_rank != self.tp_rank
        ):
            raise ValueError(
                "NSA shared layer-group HiCache requires the device layer "
                "split rank space to match the shared host cache group: "
                f"layer_shard_rank={device_pool.layer_shard_rank}, "
                f"layer_shard_size={device_pool.layer_shard_size} vs "
                f"cache_rank={self.tp_rank}, cache_size={self.tp_size}."
            )

        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        local_layer_num = device_pool.layer_num

        self.my_rel_start, self.my_rel_end = _get_layer_shard_range(
            self.tp_rank, self.tp_size, local_layer_num
        )
        self.my_num_layers = self.my_rel_end - self.my_rel_start

        self.my_abs_start = self.start_layer + self.my_rel_start
        self.my_abs_end = self.start_layer + self.my_rel_end

        logger.info(
            f"Host Cache Sharding: Rank {self.tp_rank}/{self.tp_size} owns relative layers "
            f"[{self.my_rel_start}, {self.my_rel_end}) -> absolute layers "
            f"[{self.my_abs_start}, {self.my_abs_end})."
        )

        # Keep the root mmap tensors alive. The layer entries below are views
        # into these mappings; retaining only views is not enough for every
        # backend/runtime combination when querying mapped device pointers.
        self._shared_mmap_refs = []
        self._shared_my_files = None
        self._owned_kv_data_ptr_values = None
        self._kv_root_tensor = None
        self._index_root_tensor = None
        self._skip_nsa_indexer_host = True

        try:
            super().__init__(
                device_pool,
                host_to_device_ratio,
                host_size,
                page_size,
                layout,
                pin_memory,
                device,
                allocator_type,
            )

            self._init_indexer_buffers()
        finally:
            self._unlink_owned_cache_files()

        self.index_data_refs = [
            self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
        ]
        self.index_data_ptrs = torch.tensor(
            self._build_owned_root_pointer_values(
                self._index_root_tensor, self.index_data_refs
            ),
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            self._build_data_pointer_values(self.data_refs),
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def _build_data_pointer_values(self, data_refs) -> list[int]:
        if self._owned_kv_data_ptr_values is not None:
            return self._owned_kv_data_ptr_values
        return super()._build_data_pointer_values(data_refs)

    def _build_owned_root_pointer_values(
        self, root_tensor: Optional[torch.Tensor], layer_refs
    ) -> list[int]:
        ptrs = [0] * self.layer_num
        if root_tensor is None:
            return ptrs

        if _is_dcu:
            root_device_ptr = _hip_host_get_device_pointer(root_tensor.data_ptr())
        else:
            root_device_ptr = kernel_accessible_host_ptr(root_tensor)
        root_host_ptr = root_tensor.data_ptr()
        for layer_idx in range(self.my_rel_start, self.my_rel_end):
            layer_ref = layer_refs[layer_idx]
            if layer_ref is None:
                raise RuntimeError(
                    f"Missing owned host layer {layer_idx} on rank {self.tp_rank}."
                )
            offset = layer_ref.data_ptr() - root_host_ptr
            if offset < 0 or offset >= root_tensor.nbytes:
                raise RuntimeError(
                    "Owned host layer view is outside its mapped root: "
                    f"rank={self.tp_rank}, layer={layer_idx}, offset={offset}, "
                    f"root_nbytes={root_tensor.nbytes}."
                )
            ptrs[layer_idx] = root_device_ptr + offset
        return ptrs

    def _unlink_owned_cache_files(self) -> None:
        for path in (self._shared_my_files or {}).values():
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Failed to unlink NSA HiCache file %s: %s", path, exc)

    def _init_indexer_buffers(self):
        my_files = self._shared_my_files

        index_buffer_second_dim = (
            self.indexer_page_slots * self.indexer_size_per_token
        )
        self.index_stride_size = (
            self.indexer_size_per_token * self.indexer_dtype.itemsize
        )

        self.index_k_with_scale_buffer = [None] * self.layer_num

        file_idx = self.tp_rank
        r_rel_start, r_rel_end = self.my_rel_start, self.my_rel_end
        r_num = r_rel_end - r_rel_start
        if r_num > 0:
            if not my_files or "index" not in my_files:
                raise RuntimeError(
                    f"Rank {self.tp_rank} has owned layers but no index cache file."
                )
            idx_shape = (r_num, self.page_num, index_buffer_second_dim)
            idx_numel = r_num * self.page_num * index_buffer_second_dim
            idx_mapped_numel = (
                _align_up(idx_numel * self.indexer_dtype.itemsize)
                // self.indexer_dtype.itemsize
                if _hugepage_enabled()
                else idx_numel
            )
            idx_tensor = torch.from_file(
                my_files["index"],
                shared=True,
                size=idx_mapped_numel,
                dtype=self.indexer_dtype,
                device="cpu",
            )[:idx_numel].view(idx_shape)
            self._index_root_tensor = idx_tensor
            self._shared_mmap_refs.append(idx_tensor)
            if self.pin_memory and (_is_cuda or _is_dcu):
                checked_register_host_tensor_for_kernel_access(
                    idx_tensor,
                    idx_tensor.numel() * idx_tensor.element_size(),
                    f"nsa_layer_group_index_rank{file_idx}_root",
                )
            logger.info(
                f"Rank {self.tp_rank} mapped and registered only its own Indexer "
                f"root for relative layers [{r_rel_start}, {r_rel_end})."
            )

            for i in range(r_num):
                global_layer_idx = r_rel_start + i
                self.index_k_with_scale_buffer[global_layer_idx] = idx_tensor[i]

        dist.barrier(group=self.tp_group)

        self.index_k_data_refs = [
            self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
        ]
        self.index_k_data_ptrs = torch.tensor(
            self._build_owned_root_pointer_values(
                self._index_root_tensor, self.index_k_data_refs
            ),
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        device_index_k_cache = self._get_index_device_cache_for_transfer(
            self.device_pool
        )
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in device_index_k_cache],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def _get_physical_allocation_bytes(self) -> int:
        # Capacity is still computed from all NSA layers because the rank-local
        # files form one logical shared pool. The startup availability check,
        # however, must use only the files this rank actually creates.
        main_bytes = (
            self.my_num_layers
            * self.size
            * self.kv_cache_dim
            * self.dtype.itemsize
        )
        index_bytes = (
            self.my_num_layers
            * self.page_num
            * self.indexer_page_slots
            * self.indexer_size_per_token
            * self.indexer_dtype.itemsize
        )
        return main_bytes + index_bytes

    def _get_index_device_cache_for_transfer(self, device_pool):
        if self.use_fp8_index_k_cache:
            return device_pool.index_k_with_scale_buffer
        row_width = self.page_size * self.indexer_size_per_token
        return [
            buf.view(torch.uint8).view(buf.shape[0], row_width)
            for buf in device_pool.index_k_buffer
        ]

    def init_kv_buffer(self):
        if self.layout != "layer_first":
            raise ValueError(
                f"Shared pool currently only supports layer_first layout, got {self.layout}"
            )

        if _hugepage_enabled():
            logger.info(
                f"HugePage enabled. Using shared host cache directory: "
                f"{GLM_HICACHE_SHM_DIR}, PageSize = {_hugepage_size()}."
            )

        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        kv_element_dim = self.kv_cache_dim

        index_buffer_second_dim = (
            self.indexer_page_slots * self.indexer_size_per_token
        )

        my_files = None
        if self.my_num_layers > 0:
            uid = str(uuid.uuid4())
            kv_name = (
                f"{GLM_HICACHE_SHM_DIR}/sglang_nsa_kv_"
                f"{self.my_abs_start}_{self.my_abs_end}_{uid}.bin"
            )
            idx_name = (
                f"{GLM_HICACHE_SHM_DIR}/sglang_nsa_idx_"
                f"{self.my_abs_start}_{self.my_abs_end}_{uid}.bin"
            )
            my_files = {"kv": kv_name, "index": idx_name}
            self._shared_my_files = my_files

            kv_bytes = (
                self.my_num_layers
                * self.size
                * kv_element_dim
                * self.dtype.itemsize
            )
            idx_bytes = (
                self.my_num_layers
                * self.page_num
                * index_buffer_second_dim
                * self.indexer_dtype.itemsize
            )
            kv_alloc_bytes = _align_up(kv_bytes) if _hugepage_enabled() else kv_bytes
            idx_alloc_bytes = (
                _align_up(idx_bytes) if _hugepage_enabled() else idx_bytes
            )

            t_zero_alloc = time.perf_counter()
            with open(kv_name, "wb") as f:
                try:
                    os.posix_fallocate(f.fileno(), 0, kv_alloc_bytes)
                except (AttributeError, OSError):
                    f.truncate(kv_alloc_bytes)

            with open(idx_name, "wb") as f:
                try:
                    os.posix_fallocate(f.fileno(), 0, idx_alloc_bytes)
                except (AttributeError, OSError):
                    f.truncate(idx_alloc_bytes)
            logger.info(
                f"Rank {self.tp_rank} cache file creation finished in "
                f"{time.perf_counter() - t_zero_alloc:.3f}s"
            )

            t_zero_alloc = time.perf_counter()
            my_kv_numel = self.my_num_layers * self.size * 1 * kv_element_dim
            tmp_kv_numel = kv_alloc_bytes // self.dtype.itemsize
            tmp_kv = torch.from_file(
                kv_name,
                shared=True,
                size=tmp_kv_numel,
                dtype=self.dtype,
                device="cpu",
            )
            tmp_kv[:my_kv_numel].zero_()
            del tmp_kv

            my_idx_numel = self.my_num_layers * self.page_num * index_buffer_second_dim
            tmp_idx_numel = idx_alloc_bytes // self.indexer_dtype.itemsize
            tmp_idx = torch.from_file(
                idx_name,
                shared=True,
                size=tmp_idx_numel,
                dtype=self.indexer_dtype,
                device="cpu",
            )
            tmp_idx[:my_idx_numel].zero_()
            del tmp_idx
            logger.info(
                f"Rank {self.tp_rank} allocated and zeroed its own physical memory "
                f"locally in {time.perf_counter() - t_zero_alloc:.3f}s"
            )

        self.kv_buffer = [None] * self.layer_num

        file_idx = self.tp_rank
        r_rel_start, r_rel_end = self.my_rel_start, self.my_rel_end
        r_num = r_rel_end - r_rel_start
        if r_num > 0:
            if not my_files or "kv" not in my_files:
                raise RuntimeError(
                    f"Rank {self.tp_rank} has owned layers but no KV cache file."
                )
            kv_shape = (r_num, self.size, 1, kv_element_dim)
            kv_numel = r_num * self.size * 1 * kv_element_dim
            kv_mapped_numel = (
                _align_up(kv_numel * self.dtype.itemsize) // self.dtype.itemsize
                if _hugepage_enabled()
                else kv_numel
            )
            kv_tensor = torch.from_file(
                my_files["kv"],
                shared=True,
                size=kv_mapped_numel,
                dtype=self.dtype,
                device="cpu",
            )[:kv_numel].view(kv_shape)
            self._kv_root_tensor = kv_tensor
            self._shared_mmap_refs.append(kv_tensor)
            if self.pin_memory and (_is_cuda or _is_dcu):
                checked_register_host_tensor_for_kernel_access(
                    kv_tensor,
                    kv_tensor.numel() * kv_tensor.element_size(),
                    f"nsa_layer_group_kv_rank{file_idx}_root",
                )
            logger.info(
                f"Rank {self.tp_rank} mapped and registered only its own KV root "
                f"for relative layers [{r_rel_start}, {r_rel_end})."
            )

            for i in range(r_num):
                global_layer_idx = r_rel_start + i
                self.kv_buffer[global_layer_idx] = kv_tensor[i]

            self._owned_kv_data_ptr_values = self._build_owned_root_pointer_values(
                self._kv_root_tensor, self.kv_buffer
            )

        dist.barrier(group=self.tp_group)

        return self.kv_buffer

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        pool_transfers=None,
    ):
        if getattr(device_pool, "layer_shard_enabled", False):
            absolute_layer_id = device_pool.start_layer + layer_id
            device_pool.invalidate_remote_kv_buffer_for_layer(absolute_layer_id)
            device_pool.invalidate_index_buffer_for_layer(absolute_layer_id)
        if (
            getattr(device_pool, "layer_shard_enabled", False)
            and (layer_id < self.my_rel_start or layer_id >= self.my_rel_end)
        ):
            return
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )
        page_indices_host = host_indices[:: self.page_size] // self.page_size
        page_indices_device = device_indices[:: self.page_size] // self.page_size
        item_size = self.index_stride_size * self.indexer_page_slots
        device_index_k_cache = self._get_index_device_cache_for_transfer(device_pool)

        if io_backend == "kernel":
            transfer_kv_per_layer_mla(
                src=self.index_k_with_scale_buffer[layer_id],
                dst=device_index_k_cache[layer_id],
                src_indices=page_indices_host,
                dst_indices=page_indices_device,
                item_size=item_size,
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=[self.index_k_with_scale_buffer[layer_id]],
                dst_layers=[device_index_k_cache[layer_id]],
                src_indices=page_indices_host,
                dst_indices=page_indices_device,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported IO backend for NSA indexer: {io_backend}")

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers=None,
    ) -> None:
        if self.my_num_layers == 0:
            return

        if io_backend == "kernel":
            src_ptrs = device_pool.data_ptrs[
                self.my_rel_start : self.my_rel_end
            ].contiguous()
            dst_ptrs = self.data_ptrs[
                self.my_rel_start : self.my_rel_end
            ].contiguous()

            transfer_kv_all_layer_mla(
                src_layers=src_ptrs,
                dst_layers=dst_ptrs,
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self.token_stride_size,
                num_layers=self.my_num_layers,
            )
        elif io_backend == "direct":
            src_layers = [
                device_pool.kv_buffer[i]
                for i in range(self.my_rel_start, self.my_rel_end)
            ]
            dst_layers = [
                self.data_refs[i] for i in range(self.my_rel_start, self.my_rel_end)
            ]
            transfer_kv_direct(
                src_layers=src_layers,
                dst_layers=dst_layers,
                src_indices=device_indices,
                dst_indices=host_indices,
                page_size=self.page_size,
            )
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

        page_indices_host = host_indices[:: self.page_size] // self.page_size
        page_indices_device = device_indices[:: self.page_size] // self.page_size
        item_size = self.index_stride_size * self.indexer_page_slots

        if io_backend == "kernel":
            src_ptrs = self.index_k_device_ptrs[
                self.my_rel_start : self.my_rel_end
            ].contiguous()
            dst_ptrs = self.index_data_ptrs[
                self.my_rel_start : self.my_rel_end
            ].contiguous()

            transfer_kv_all_layer_mla(
                src_layers=src_ptrs,
                dst_layers=dst_ptrs,
                src_indices=page_indices_device,
                dst_indices=page_indices_host,
                item_size=item_size,
                num_layers=self.my_num_layers,
            )
        elif io_backend == "direct":
            device_index_k_cache = self._get_index_device_cache_for_transfer(
                device_pool
            )
            src_layers = [
                device_index_k_cache[i]
                for i in range(self.my_rel_start, self.my_rel_end)
            ]
            dst_layers = [
                self.index_k_with_scale_buffer[i]
                for i in range(self.my_rel_start, self.my_rel_end)
            ]
            transfer_kv_direct(
                src_layers=src_layers,
                dst_layers=dst_layers,
                src_indices=page_indices_device,
                dst_indices=page_indices_host,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported IO backend for NSA indexer: {io_backend}")
