from __future__ import annotations

import abc
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolName

import os
import time
import uuid

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
from sglang.srt.environ import envs
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
    KVCache,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.utils import is_cuda, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
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
        torch.cuda.cudart().cudaHostRegister(
            buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
        )
    return buffer


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


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
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

        if self.size <= device_pool.size:
            logger.warning(
                "The host kv memory should be larger than the device memory with the current protocol, "
                "but host_size=%s and device_size=%s.",
                self.size,
                device_pool.size,
            )

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes and not _hugepage_enabled():
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
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
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
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers=None,
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
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

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
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        pool_transfers=None,
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
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers=None,
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
        ], "Unsupported layout: {layout}"

        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.num_mamba_layers = device_pool.num_mamba_layers

        self.conv_state_shapes = [
            conv_state.shape[2:] for conv_state in device_pool.mamba_cache.conv
        ]
        self.temporal_state_shape = device_pool.mamba_cache.temporal.shape[2:]
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

        if self.size <= device_pool.size:
            logger.warning(
                "The host mamba memory should be larger than the device memory with the current protocol, "
                "but host_size=%s and device_size=%s.",
                self.size,
                device_pool.size,
            )

        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
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
        conv_total_size = 0
        for conv_shape in self.conv_state_shapes:
            conv_total_size += int(np.prod(conv_shape)) * self.conv_dtype.itemsize
        temporal_size = (
            int(np.prod(self.temporal_state_shape)) * self.temporal_dtype.itemsize
        )
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
                [src_layers[i].data_ptr() for i in range(num_layers)],
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


def prepare_mamba_h2d_direct_indices(
    host_indices: torch.Tensor, device_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU indices sorted as source/destination pairs for direct H2D."""
    if host_indices.ndim != 1 or device_indices.ndim != 1:
        raise ValueError("Mamba H2D indices must be one-dimensional")
    if host_indices.numel() != device_indices.numel():
        raise ValueError("Mamba H2D indices must have equal length")

    host_indices_cpu = (
        host_indices if host_indices.device.type == "cpu" else host_indices.cpu()
    )
    device_indices_cpu = (
        device_indices if device_indices.device.type == "cpu" else device_indices.cpu()
    )
    sorted_host_indices, order = host_indices_cpu.sort()
    return sorted_host_indices, device_indices_cpu.index_select(0, order)


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
        self.mamba_h2d_direct_enabled = (
            not envs.SGLANG_DISABLE_HICACHE_MAMBA_H2D_DIRECT.get()
        )

    def clear(self) -> None:
        for entry in self.entries:
            entry.host_pool.clear()

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
            effective_backend = io_backend
            if (
                self.mamba_h2d_direct_enabled
                and io_backend == "kernel"
                and isinstance(entry.host_pool, MambaPoolHost)
                and entry.host_pool.layout == "layer_first"
                and entry.host_pool.pin_memory
            ):
                effective_backend = "direct"
            entry.host_pool.load_to_device_per_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                local_layer_id,
                effective_backend,
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


class NSATokenToKVPoolHost(MLATokenToKVPoolHost):
    device_pool: NSATokenToKVPool

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
        # Initialize indexer metadata before HostKVCache.__init__ calls get_size_per_token.
        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.indexer_dtype = NSATokenToKVPool.index_k_with_scale_buffer_dtype
        self.indexer_size_per_token = (
            self.index_head_dim
            + self.index_head_dim // self.indexer_quant_block_size * 4
        )
        # Mirror device-side dense kpool-compress packing (slots_per_page
        # = page_size for anchor; page_size // index_kpool for dense).
        self.slots_per_page = getattr(device_pool, "slots_per_page", page_size)
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
        self.indexer_page_stride_size = (
            self.indexer_size_per_token
            * self.slots_per_page
            * self.indexer_dtype.itemsize
        )
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self._init_indexer_buffers()
        logger.info(
            f"NSATokenToKVPoolHost initialized with indexer page stride size: {self.indexer_page_stride_size}, page num: {self.indexer_page_num}"
        )

    def get_size_per_token(self):
        base = super().get_size_per_token()
        # Anchor: slots_per_page=64 -> 132*L*64/64 = 132 L bytes/token
        # Dense : slots_per_page=4  -> 132*L*4/64  = 8.25 L bytes/token
        return (
            base
            + self.indexer_size_per_token
            * self.layer_num
            * self.indexer_dtype.itemsize
            * self.slots_per_page
            // self.page_size
        )

    def _init_indexer_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.index_k_with_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer = [
                alloc_func(
                    (self.indexer_page_num, self.indexer_page_stride_size),
                    dtype=self.indexer_dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.index_k_data_refs = [
                self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
            ]
            self.index_k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in self.index_k_data_refs],
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

    def _load_indexer_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[layer_id],
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
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
                    dst_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[device_pool.index_k_with_scale_buffer[layer_id]],
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
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=device_pool.index_k_with_scale_buffer[layer_id],
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
                    src_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
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

    def _backup_indexer_from_device_all_layer(
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
                    src_layers=device_pool.index_k_with_scale_buffer,
                    dst_layers=self.index_k_with_scale_buffer,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.index_k_with_scale_buffer,
                    dst_ptrs=[self.index_k_with_scale_buffer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

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
        self._load_indexer_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend,
    ):
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        self._backup_indexer_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )


class NSATokenToKVPoolHostShared(NSATokenToKVPoolHost):
    """
    基于 Shared Memory 的 NSA Host Cache 管理。

    特性：
    1. 同时支持 NSA 的 KV Cache 和 Indexer Cache (index_k_with_scale_buffer) 的共享内存管理。
    2. Rank 0 创建 /dev/shm 映射文件，组内其他 Rank 映射同一物理内存。
    3. Load 时：所有 Rank 并行读取。
    4. Backup 时：仅 Rank 0 写回，保证数据一致性并减少总线竞争。
    """

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

        # 初始化 TP/DP 信息
        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = tp_group

        # 初始化父类
        # 注意：父类 __init__ 会调用 init_kv_buffer，所以在此之前 tp_rank 等必须已设置
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

        # 重新初始化 NSA 特有的 Index buffer 引用 (父类中已经做过，但这里确保 shared memory 设置正确后引用也是对的)
        self.index_data_refs = [
            self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
        ]
        self.index_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.index_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

        # 重新初始化 MLA 部分的引用
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def _init_indexer_buffers(self):
        """
        初始化共享内存 NSA Index Buffer。
        """
        # 计算 Shape
        index_buffer_second_dim = self.slots_per_page * (
            self.index_head_dim
            + self.index_head_dim // self.indexer_quant_block_size * 4
        )
        self.index_stride_size = (
            self.index_head_dim
            + self.index_head_dim // self.indexer_quant_block_size * 4
        ) * self.indexer_dtype.itemsize

        # Shape: (LayerNum, PageNum, ElementDim)
        index_dims = (self.layer_num, self.page_num, index_buffer_second_dim)

        full_index_buffer = self._allocate_shared_buffer(
            "index", index_dims, self.indexer_dtype
        )

        # 切分为 list 以兼容父类接口: [tensor(layer_0), tensor(layer_1), ...]
        self.index_k_with_scale_buffer = [
            full_index_buffer[i] for i in range(self.layer_num)
        ]

        # Set up pointer tensors
        self.index_k_data_refs = [
            self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
        ]
        self.index_k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.index_k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.index_k_with_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def _allocate_shared_buffer(
        self, name_suffix: str, shape: tuple, dtype: torch.dtype
    ):
        """
        通用的共享内存分配辅助函数（已针对 cudaHostRegister 的 Page Fault 瓶颈进行优化，包含耗时打点）。
        """
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

        # 1. 协商文件名
        shared_filename = None
        if self.tp_rank == 0:
            t_file_start = time.perf_counter()
            unique_id = str(uuid.uuid4())
            # 区分 KV 和 Indexer 的文件名
            shared_filename = f"/dev/shm/sglang_nsa_{name_suffix}_{unique_id}.bin"
            with open(shared_filename, "wb") as f:
                try:
                    os.posix_fallocate(f.fileno(), 0, total_bytes)
                except (AttributeError, OSError):
                    f.truncate(total_bytes)
            logger.info(
                f"[{name_suffix}] Rank 0 created shared file in {time.perf_counter() - t_file_start:.3f}s"
            )

        object_list = [shared_filename]
        dist.broadcast_object_list(object_list, src=0, group=self.tp_group)
        shared_filename = object_list[0]

        dist.barrier(group=self.tp_group)

        try:
            # 2. 映射内存
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
                    f"[{name_suffix}] Memory mapping took {time.perf_counter() - t_map_start:.3f}s"
                )

            # =================================================================
            # 3. 并行缺页中断 (Parallel Page Faulting)
            # =================================================================
            t_zero_start = time.perf_counter()

            chunk_size = numel // self.tp_size
            start_idx = self.tp_rank * chunk_size
            end_idx = (
                numel if self.tp_rank == self.tp_size - 1 else start_idx + chunk_size
            )

            # 强制操作系统真正分配物理内存页
            flat_tensor[start_idx:end_idx].zero_()

            t_zero_end = time.perf_counter()
            logger.info(
                f"[{name_suffix}] Rank {self.tp_rank} finished zeroing chunk in {t_zero_end - t_zero_start:.3f}s"
            )

            # 等待所有进程都完成
            dist.barrier(group=self.tp_group)
            if self.tp_rank == 0:
                logger.info(
                    f"[{name_suffix}] Parallel page faulting (all ranks) completed in {time.perf_counter() - t_zero_start:.3f}s"
                )

            buffer = flat_tensor.view(shape)

            # 4. 注册 Pinned Memory
            if self.pin_memory and is_cuda():
                t_pin_start = time.perf_counter()
                err = torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), total_bytes, 0
                )
                if err != 0:
                    logger.warning(
                        f"Failed to pin shared memory for {name_suffix}. Error code: {err}"
                    )
                t_pin_end = time.perf_counter()
                if self.tp_rank == 0:
                    logger.info(
                        f"[{name_suffix}] cudaHostRegister took {t_pin_end - t_pin_start:.3f}s"
                    )

            if self.tp_rank == 0:
                logger.info(
                    f"[{name_suffix}] Total allocation pipeline took {time.perf_counter() - t_start_all:.3f}s"
                )

            return buffer
        finally:
            # 清理文件名
            dist.barrier(group=self.tp_group)
            if self.tp_rank == 0 and os.path.exists(shared_filename):
                os.remove(shared_filename)

    def init_kv_buffer(self):
        """
        初始化共享内存 KV Buffer。Indexer Buffer 由 _init_indexer_buffers 负责。
        """
        # --- 1. 初始化 KV Buffer (Base MLA Logic) ---
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

        # 返回 kv_buffer 以满足 HostKVCache 的接口约定
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
        """
        所有 Rank 并行从共享 Host 内存加载数据 (KV + Indexer)。
        """
        # 直接调用父类的逻辑即可。
        # 父类 NSATokenToKVPoolHost.load_to_device_per_layer 会依次调用:
        # 1. super().load... (即 MLATokenToKVPoolHost 的 load，负责 KV)
        # 2. self._load_indexer_to_device_per_layer (负责 Indexer)
        # 因为所有 Rank 都映射了共享内存，所以 standard load 逻辑 = 并行读取 = 正确。
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers=None,
    ) -> None:
        """
        Backup 数据。仅 Rank 0 写回共享内存。
        同时处理 KV Buffer 和 Index Buffer。
        """
        if self._is_device_layer_sharded(device_pool) or self.tp_rank == 0:
            # 调用父类 backup，父类会依次处理 KV 和 Indexer 的写回。
            # 非 layer-split 时只有 Rank 0 执行此操作，避免写入冲突；
            # layer-split 时每个 rank 只写自己 owned layers。
            super().backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend
            )


class NSATokenToKVPoolHostSharedLayerGroup(NSATokenToKVPoolHost):
    """
    基于分布式 Shared Memory 的 NSA Host Cache 管理 (兼顾 Pipeline Parallel)。

    特性：
    1. 每个 Rank 创建并初始化属于自己的 cache 空间（按相对层划分），并通过 shared memory 映射给组内所有 Rank。
    2. 支持 Pipeline Parallel：文件名与分块逻辑明确使用 absolute layer 进行唯一化隔离。
    3. Load 时：所有 Rank 可以并行地按照 layer id 从组装好的全局 List[Tensor] 中并行读取。
    4. Backup 时：每个 Rank 仅将自己负责的 layer 范围通过内核写回到自己初始化的那段 cache，消除总线竞争。
    """

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
            "Using NSATokenToKVPoolHostShared (Per-Rank Distributed Shards) for zero-copy host cache (NSA)."
        )

        if getattr(device_pool, "slots_per_page", page_size) != page_size:
            raise NotImplementedError(
                "NSATokenToKVPoolHostSharedLayerGroup does not support dense "
                "kpool layout (slots_per_page != page_size); use "
                "NSATokenToKVPoolHost or NSATokenToKVPoolHostShared instead."
            )

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = tp_group

        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        local_layer_num = device_pool.layer_num

        chunk = local_layer_num // self.tp_size
        rem = local_layer_num % self.tp_size

        self.my_rel_start = self.tp_rank * chunk + min(self.tp_rank, rem)
        self.my_rel_end = self.my_rel_start + chunk + (1 if self.tp_rank < rem else 0)
        self.my_num_layers = self.my_rel_end - self.my_rel_start

        self.my_abs_start = self.start_layer + self.my_rel_start
        self.my_abs_end = self.start_layer + self.my_rel_end

        logger.info(
            f"Host Cache Sharding: Rank {self.tp_rank}/{self.tp_size} owns relative layers "
            f"[{self.my_rel_start}, {self.my_rel_end}) -> absolute layers [{self.my_abs_start}, {self.my_abs_end})."
        )

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

        self.index_data_refs = [
            self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
        ]
        self.index_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.index_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def _init_indexer_buffers(self):
        """
        映射共享内存中的 NSA Indexer Buffer 并注册 CUDA pinned memory。
        使用 init_kv_buffer 中保存的 self._shared_all_files。
        """
        all_files = self._shared_all_files
        my_files = self._shared_my_files

        index_buffer_second_dim = self.page_size * self.indexer_size_per_token
        self.index_stride_size = (
            self.indexer_size_per_token * self.indexer_dtype.itemsize
        )

        self.index_k_with_scale_buffer = [None] * self.layer_num

        for step in range(self.tp_size):
            # Rotate stagger 错开锁竞争
            file_idx = (step + self.tp_rank) % self.tp_size

            chunk = self.layer_num // self.tp_size
            rem = self.layer_num % self.tp_size
            r_rel_start = file_idx * chunk + min(file_idx, rem)
            r_rel_end = r_rel_start + chunk + (1 if file_idx < rem else 0)
            r_num = r_rel_end - r_rel_start

            if r_num == 0:
                continue

            files = all_files[file_idx]

            t_pin = time.perf_counter()

            idx_shape = (r_num, self.page_num, index_buffer_second_dim)
            idx_numel = r_num * self.page_num * index_buffer_second_dim
            idx_mapped_numel = (
                _align_up(idx_numel * self.indexer_dtype.itemsize)
                // self.indexer_dtype.itemsize
                if _hugepage_enabled()
                else idx_numel
            )
            idx_tensor = torch.from_file(
                files["index"],
                shared=True,
                size=idx_mapped_numel,
                dtype=self.indexer_dtype,
                device="cpu",
            )[:idx_numel].view(idx_shape)
            if self.pin_memory and is_cuda():
                torch.cuda.cudart().cudaHostRegister(
                    idx_tensor.data_ptr(), idx_numel * self.indexer_dtype.itemsize, 0
                )

            logger.info(
                f"Rank {self.tp_rank} finish Indexer cudaHostRegister for Rank {file_idx}'s file in {time.perf_counter()-t_pin:.3f}s"
            )

            for i in range(r_num):
                global_layer_idx = r_rel_start + i
                self.index_k_with_scale_buffer[global_layer_idx] = idx_tensor[i]

        # 等待所有 Rank 完成 Indexer 映射后，清理 idx 文件
        dist.barrier(group=self.tp_group)
        if my_files:
            idx_file = my_files.get("index")
            if idx_file and os.path.exists(idx_file):
                os.remove(idx_file)

        # 清理临时属性
        del self._shared_all_files
        del self._shared_my_files

        # 设置指针 Tensor
        self.index_k_data_refs = [
            self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
        ]
        self.index_k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.index_k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.index_k_with_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def init_kv_buffer(self):
        if self.layout != "layer_first":
            raise ValueError(
                f"Shared pool currently only supports layer_first layout, got {self.layout}"
            )

        if _hugepage_enabled():
            logger.info(
                f"HugePage enabled. Using shared host cache directory: {GLM_HICACHE_SHM_DIR}, PageSize = {_hugepage_size()}."
            )

        # 计算 Element Dimensions
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        kv_element_dim = self.kv_cache_dim

        index_buffer_second_dim = self.page_size * (
            self.index_head_dim
            + self.index_head_dim // self.indexer_quant_block_size * 4
        )

        # === Step 1: 当前 Rank 在本地 /dev/shm 创建并分配自己负责的那块 Cache 空间 ===
        my_files = None
        t_zero = time.perf_counter()
        if self.my_num_layers > 0:
            uid = str(uuid.uuid4())
            # 文件名带上 absolute layer 以隔离同节点上运行的其他 PP Stage
            kv_name = f"{GLM_HICACHE_SHM_DIR}/sglang_nsa_kv_{self.my_abs_start}_{self.my_abs_end}_{uid}.bin"
            idx_name = f"{GLM_HICACHE_SHM_DIR}/sglang_nsa_idx_{self.my_abs_start}_{self.my_abs_end}_{uid}.bin"

            kv_bytes = (
                self.my_num_layers * self.size * kv_element_dim * self.dtype.itemsize
            )
            idx_bytes = (
                self.my_num_layers
                * self.page_num
                * index_buffer_second_dim
                * self.indexer_dtype.itemsize
            )
            kv_alloc_bytes = _align_up(kv_bytes) if _hugepage_enabled() else kv_bytes
            idx_alloc_bytes = _align_up(idx_bytes) if _hugepage_enabled() else idx_bytes

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
                f"Rank {self.tp_rank} cache file creation finished in {time.perf_counter()-t_zero_alloc:.3f}s"
            )

            # 在 Step 1 就原地将文件映射到内存并执行 zero_()。
            # 此时各个 Rank 之间完全并行，物理内存分配（触发缺页中断）会落在此 Rank 绑定的 NUMA 节点上。
            # 提前写满 0，消除后续 Step 3 中 cudaHostRegister 时因等待物理页分配产生的锁竞争等待。
            t_zero_alloc = time.perf_counter()
            my_kv_numel = self.my_num_layers * self.size * 1 * kv_element_dim
            tmp_kv_numel = kv_alloc_bytes // self.dtype.itemsize
            tmp_kv = torch.from_file(
                kv_name, shared=True, size=tmp_kv_numel, dtype=self.dtype, device="cpu"
            )
            tmp_kv[:my_kv_numel].zero_()
            del tmp_kv  # 释放临时映射句柄

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
            del tmp_idx  # 释放临时映射句柄
            logger.info(
                f"Rank {self.tp_rank} allocated and zeroed its own physical memory locally in {time.perf_counter()-t_zero_alloc:.3f}s"
            )

            my_files = {"kv": kv_name, "index": idx_name}

        # === Step 2: 组内全局收集当前 PP Stage 所有 Rank 创建的文件名 ===
        all_files = [None] * self.tp_size
        dist.all_gather_object(all_files, my_files, group=self.tp_group)

        # === Step 3: 映射 KV Buffer（Indexer 由 _init_indexer_buffers 负责映射） ===
        self.kv_buffer = [None] * self.layer_num

        for step in range(self.tp_size):
            # Rotate stagger 错开锁竞争
            file_idx = (step + self.tp_rank) % self.tp_size

            chunk = self.layer_num // self.tp_size
            rem = self.layer_num % self.tp_size
            r_rel_start = file_idx * chunk + min(file_idx, rem)
            r_rel_end = r_rel_start + chunk + (1 if file_idx < rem else 0)
            r_num = r_rel_end - r_rel_start

            if r_num == 0:
                continue

            files = all_files[file_idx]

            t_pin = time.perf_counter()

            # --- 映射 KV ---
            kv_shape = (r_num, self.size, 1, kv_element_dim)
            kv_numel = r_num * self.size * 1 * kv_element_dim
            kv_mapped_numel = (
                _align_up(kv_numel * self.dtype.itemsize) // self.dtype.itemsize
                if _hugepage_enabled()
                else kv_numel
            )
            kv_tensor = torch.from_file(
                files["kv"],
                shared=True,
                size=kv_mapped_numel,
                dtype=self.dtype,
                device="cpu",
            )[:kv_numel].view(kv_shape)
            if self.pin_memory and is_cuda():
                torch.cuda.cudart().cudaHostRegister(
                    kv_tensor.data_ptr(), kv_numel * self.dtype.itemsize, 0
                )

            logger.info(
                f"Rank {self.tp_rank} finish KV cudaHostRegister for Rank {file_idx}'s file in {time.perf_counter()-t_pin:.3f}s"
            )

            for i in range(r_num):
                global_layer_idx = r_rel_start + i
                self.kv_buffer[global_layer_idx] = kv_tensor[i]

        # 等待所有 Rank 完成 KV 映射后，清理 KV 文件（映射保持有效）
        dist.barrier(group=self.tp_group)
        if my_files:
            kv_file = my_files.get("kv")
            if kv_file and os.path.exists(kv_file):
                os.remove(kv_file)

        # 存储文件信息供 _init_indexer_buffers 使用
        self._shared_all_files = all_files
        self._shared_my_files = my_files

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

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers=None,
    ) -> None:
        layer_ids = [
            layer_id
            for layer_id in range(self.my_rel_start, self.my_rel_end)
            if self._is_device_layer_owned(device_pool, layer_id)
        ]
        if not layer_ids:
            return
        layer_index = torch.tensor(
            layer_ids, dtype=torch.long, device=device_pool.data_ptrs.device
        )
        transfer_layer_num = len(layer_ids)

        # 1. Backup KV Buffer
        if io_backend == "kernel":
            src_ptrs = device_pool.data_ptrs.index_select(0, layer_index).contiguous()
            dst_ptrs = self.data_ptrs.index_select(0, layer_index).contiguous()

            transfer_kv_all_layer_mla(
                src_layers=src_ptrs,
                dst_layers=dst_ptrs,
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self.token_stride_size,
                num_layers=transfer_layer_num,
            )
        elif io_backend == "direct":
            src_layers = [
                device_pool.kv_buffer[i] for i in layer_ids
            ]
            dst_layers = [
                self.data_refs[i] for i in layer_ids
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

        # 2. Backup Index Buffer (NSA)
        page_indices_host = host_indices[:: self.page_size] // self.page_size
        page_indices_device = device_indices[:: self.page_size] // self.page_size

        if io_backend == "kernel":
            src_ptrs = self.index_k_device_ptrs.index_select(
                0, layer_index
            ).contiguous()
            dst_ptrs = self.index_data_ptrs.index_select(0, layer_index).contiguous()

            transfer_kv_all_layer_mla(
                src_layers=src_ptrs,
                dst_layers=dst_ptrs,
                src_indices=page_indices_device,
                dst_indices=page_indices_host,
                item_size=self.index_stride_size * self.page_size,
                num_layers=transfer_layer_num,
            )
        elif io_backend == "direct":
            src_layers = [
                device_pool.index_k_with_scale_buffer[i]
                for i in layer_ids
            ]
            dst_layers = [
                self.index_k_with_scale_buffer[i]
                for i in layer_ids
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
