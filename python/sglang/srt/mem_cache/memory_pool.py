"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import dataclasses
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa import index_buf_accessor
from sglang.srt.layers.attention.nsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.dp_attention import get_attention_cp_group
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    row_bytes = row_dim * store_dtype.itemsize
    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            indices,
            row_bytes=row_bytes,
        )

    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        k_cache[indices] = k
        v_cache[indices] = v


def _get_layer_shard_range(
    rank: int, shard_size: int, total_layers: int
) -> tuple[int, int]:
    base = total_layers // shard_size
    rem = total_layers % shard_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def _get_layer_owner(local_layer_idx: int, shard_size: int, total_layers: int) -> int:
    for rank in range(shard_size):
        start, end = _get_layer_shard_range(rank, shard_size, total_layers)
        if start <= local_layer_idx < end:
            return rank
    raise ValueError(
        f"Invalid local_layer_idx={local_layer_idx} for "
        f"shard_size={shard_size}, total_layers={total_layers}"
    )


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        # NOTE: this check is relaxed temporarily
        # https://github.com/sgl-project/sglang/pull/20476
        # if not any(r.is_dllm() for r in reqs):
        #     assert (
        #         sum(1 for i in reusing if reqs[i].is_chunked > 0) <= 1
        #     ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(self.size))

    def mamba_pool_has_space_for_reqs(self, num_reqs: int = 1):
        return True


class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            kwargs = {}
            # Use fields instead of vars to avoid torch.compile graph break
            for f in fields(self):
                name = f.name
                v = getattr(self, name)
                if name in ("conv", "intermediate_conv_window"):
                    kwargs[name] = [conv[layer] for conv in v]
                else:
                    kwargs[name] = v[layer]

            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(mamba_layer_ids)

        self.size = size
        self.device = device

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE), (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ):
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                conv_state = _init_amx_conv_state(conv_state)

            temporal_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )
            if speculative_num_draft_tokens is not None:
                # Cache intermediate SSM states per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device="cuda",
                )
                # Cache intermediate conv windows (last K-1 inputs) per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, dim, K-1]
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.free_slots = torch.arange(
                1, self.size + 1, dtype=torch.int64, device=self.device
            )
            self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB
            self.num_mamba_layers = num_mamba_layers

            # Store packed component sizes (along the sliceable dim) for PD
            # cross-TP transfer.
            shape = cache_params.shape
            self._state_dim_components_by_type = []
            for _ in conv_state_shape:
                self._state_dim_components_by_type.append(list(shape.conv_dims))
            self._state_dim_components_by_type.append([shape.num_heads])
            if speculative_num_draft_tokens is not None:
                self._state_dim_components_by_type.append(
                    [speculative_num_draft_tokens]
                )
                for _ in conv_state_shape:
                    self._state_dim_components_by_type.append(
                        [speculative_num_draft_tokens]
                    )

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def clear_slots(self, indices: torch.Tensor):
        """Zero out mamba state at the given pool indices. Must run on forward stream."""
        need_size = len(indices)
        for i in range(len(self.mamba_cache.conv)):
            t = self.mamba_cache.conv[i]
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, indices] = z
        t = self.mamba_cache.temporal
        z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
            t.shape[0], need_size, *t.shape[2:]
        )
        t[:, indices] = z

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )

    def copy_from(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_index] = self.mamba_cache.conv[i][
                :, src_index
            ]
        self.mamba_cache.temporal[:, dst_index] = self.mamba_cache.temporal[
            :, src_index
        ]
        return

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        conv_cpu = [
            conv[:, indices].to("cpu", non_blocking=True)
            for conv in self.mamba_cache.conv
        ]
        temporal_cpu = self.mamba_cache.temporal[:, indices].to(
            "cpu", non_blocking=True
        )
        torch.cuda.synchronize()
        return conv_cpu, temporal_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        conv_cpu, temporal_cpu = mamba_cache_cpu
        torch.cuda.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
            self.mamba_cache.temporal.device, non_blocking=True
        )
        torch.cuda.synchronize()

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Skip intermediate buffers used only for speculative decoding
            # These buffers have different size (spec_state_size + 1) and should not be transferred
            if field in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)
        data_ptrs, data_lens, item_lens = [], [], []

        for _, state_tensor in enumerate(state_tensors):
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each state tensor.

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension (index 2) is the one that gets sliced by TP.
        Returns the size of this dimension for each tensor (repeated for each layer).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)

        dim_per_tensor = []
        for state_tensor in state_tensors:
            # state_tensor shape: [num_layers, size+1, sliceable_dim, ...]
            # The sliceable dimension is at index 2 (after num_layers and size)
            sliceable_dim = state_tensor.shape[2]
            # Repeat for each layer since we have per-layer data_ptrs
            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor

    def get_state_dim_components_per_tensor(self):
        """Get the packed sub-component sizes for each state tensor, aligned
        with get_state_dim_per_tensor().

        Each entry is a list of sub-dimension sizes at full (non-TP) scale.
        A single-entry list means no interleaving is needed (the sliceable
        dim is treated as one component).
        """
        splits_per_tensor = []
        for components in self._state_dim_components_by_type:
            splits_per_tensor += [components] * self.num_mamba_layers
        return splits_per_tensor


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        slots_per_req = 1 + (
            self.mamba_ping_pong_track_buffer_size if enable_mamba_extra_buffer else 0
        )
        self.slots_per_req = slots_per_req

        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
    ):
        self.mamba_pool = MambaPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}

        self.device = device
        # Indexed by req_pool_idx, so size from the req pool buffer
        # (self.req_to_token.shape[0]), not from the mamba state pool size.
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def register_layer_transfer_counter(
        self, layer_transfer_counter: "LayerDoneCounter"
    ):
        self.layer_transfer_counter = layer_transfer_counter

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        for req in reqs:
            if req.mamba_pool_idx is not None:  # for radix cache / continuing chunked
                pass
            else:
                mid = self.mamba_pool.alloc(1)
                assert (
                    mid is not None
                ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_pool.available_size()=}, {len(reqs)=}"
                req.mamba_pool_idx = mid[0]
                req.mamba_needs_clear = True
            mamba_indices.append(req.mamba_pool_idx)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    req.mamba_ping_pong_track_buffer = self.mamba_pool.alloc(
                        self.mamba_ping_pong_track_buffer_size
                    )
                    assert (
                        req.mamba_ping_pong_track_buffer is not None
                    ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
                    req.mamba_next_track_idx = 0
                mamba_ping_pong_track_buffers.append(req.mamba_ping_pong_track_buffer)
        assert len(select_index) == len(
            mamba_indices
        ), "Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffers
            ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        select_index_gpu = torch.tensor(
            select_index, dtype=torch.long, device="cpu", pin_memory=True
        ).to(device=self.device, non_blocking=True)
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping.index_copy_(
            0, select_index_gpu, mamba_index_tensor
        )
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers)
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.index_copy_(
                0,
                select_index_gpu,
                ping_pong_tensor
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def copy_mamba_state(
        self, src_index: torch.Tensor, dst_index: torch.Tensor
    ) -> None:
        """Copy all Mamba layers after any active HiCache load is ready."""
        if src_index.numel() == 0:
            return
        if (
            self.layer_transfer_counter is not None
            and self.layer_transfer_counter.consumer_index >= 0
        ):
            last_mamba_layer = max(self.mamba_map)
            self.layer_transfer_counter.wait_until(
                last_mamba_layer - self.start_layer
            )
        self.mamba_pool.copy_from(src_index, dst_index)

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def free_mamba_cache(
        self, req: "Req", mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_pool.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None
        req.mamba_cow_src_index = None
        req.mamba_needs_clear = False

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                # Avoid Python-list advanced indexing on a device tensor.
                # The ping-pong buffer size is either 2 (normal) or 1 (spec decode).
                if self.mamba_ping_pong_track_buffer_size == 2:
                    idx_to_free = 1 - mamba_ping_pong_track_buffer_to_keep
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[
                            idx_to_free : idx_to_free + 1
                        ]
                    )
                else:
                    assert self.mamba_ping_pong_track_buffer_size == 1, (
                        f"Unexpected mamba_ping_pong_track_buffer_size="
                        f"{self.mamba_ping_pong_track_buffer_size}"
                    )
                    assert mamba_ping_pong_track_buffer_to_keep == 0, (
                        "mamba_ping_pong_track_buffer_to_keep must be 0 when "
                        "mamba_ping_pong_track_buffer_size is 1"
                    )
                    # Keep the only slot, so free nothing.
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[0:0]
                    )
            self.mamba_pool.free(mamba_ping_pong_track_buffer_to_free)
            # Mirror the req.mamba_pool_idx = None clear above so the
            # alloc()/free() pair stays symmetric. alloc() decides whether
            # to (re)allocate a ping-pong buffer by checking
            # `req.mamba_ping_pong_track_buffer is None`; leaving a stale
            # reference here would let a future alloc reuse a freed slot.
            # Harmless today (non-streaming reqs are not re-alloc'd after
            # free), defensive against streaming-session / retract paths.
            req.mamba_ping_pong_track_buffer = None
            req.mamba_next_track_idx = None

    def mamba_pool_has_space_for_reqs(self, num_reqs: int = 1):
        need_size = num_reqs * self.slots_per_req
        return need_size <= self.mamba_pool.available_size()

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        layer_shard_rank: Optional[int] = None,
        layer_shard_size: int = 1,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_enabled = (
            layer_shard_rank is not None and layer_shard_size > 1
        )
        self.layer_shard_start = self.start_layer
        if self.layer_shard_enabled:
            self._log_layer_shard_plan()
            chunk = self.layer_num // self.layer_shard_size
            rem = self.layer_num % self.layer_shard_size
            self.layer_shard_start = self.layer_shard_rank * chunk + min(
                self.layer_shard_rank, rem
            )
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None
        self.layer_broadcast_comm = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _local_layer_idx(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def _owned_local_layer_range(self) -> tuple[int, int]:
        assert self.layer_shard_rank is not None
        return _get_layer_shard_range(
            self.layer_shard_rank, self.layer_shard_size, self.layer_num
        )

    def _log_layer_shard_plan(self):
        assert self.layer_shard_rank is not None
        partitions = []
        for rank in range(self.layer_shard_size):
            st, ed = _get_layer_shard_range(rank, self.layer_shard_size, self.layer_num)
            partitions.append(f"r{rank}:[{st},{ed})")
        my_start, my_end = self._owned_local_layer_range()
        logger.info(
            "Layer shard plan (continuous): "
            f"layer_num={self.layer_num}, shard_size={self.layer_shard_size}, "
            f"rank={self.layer_shard_rank}, local=[{my_start},{my_end}), "
            f"global=[{self.start_layer + my_start},{self.start_layer + my_end}), "
            f"partitions={'; '.join(partitions)}"
        )

    def _is_layer_owned(self, layer_id: int) -> bool:
        if not self.layer_shard_enabled:
            return True
        local_idx = self._local_layer_idx(layer_id)
        owned_start, owned_end = self._owned_local_layer_range()
        return owned_start <= local_idx < owned_end

    def _get_layer_owner_rank(self, layer_id: int) -> int:
        return _get_layer_owner(
            self._local_layer_idx(layer_id), self.layer_shard_size, self.layer_num
        )

    def _init_layer_broadcast_comm(self) -> None:
        if not self.layer_shard_enabled:
            return

        cp_group = get_attention_cp_group()
        if cp_group.world_size <= 1 or cp_group.pynccl_comm is None:
            return

        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        self.layer_broadcast_comm = PyNcclCommunicator(
            group=cp_group.cpu_group,
            device=cp_group.device,
        )
        logger.info(
            "Initialized dedicated layer-shard broadcast NCCL communicator: "
            f"rank={cp_group.rank_in_group}, world_size={cp_group.world_size}"
        )

    def _broadcast_tensor_from_owner(
        self,
        tensor: torch.Tensor,
        layer_id: int,
        src_tensor: Optional[torch.Tensor] = None,
        use_layer_broadcast_comm: bool = False,
    ) -> torch.Tensor:
        if not self.layer_shard_enabled:
            return tensor

        owner_rank = self._get_layer_owner_rank(layer_id)
        if self.layer_shard_rank == owner_rank:
            assert src_tensor is not None
            tensor.copy_(src_tensor)

        if use_layer_broadcast_comm and self.layer_broadcast_comm is not None:
            with self.layer_broadcast_comm.change_state(enable=True):
                self.layer_broadcast_comm.broadcast(tensor, owner_rank)
        else:
            cp_group = get_attention_cp_group()
            pynccl_comm = cp_group.pynccl_comm
            if pynccl_comm is not None:
                with pynccl_comm.change_state(enable=True):
                    pynccl_comm.broadcast(tensor, owner_rank)
            else:
                cp_group.broadcast(tensor, src=owner_rank)
        return tensor

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError()

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError()

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


class MHATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream() if _is_cuda and enable_alt_stream else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # for store_cache JIT kernel
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

    def _init_kv_copy_and_warmup(self):
        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.v_head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))
        grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

        if N <= cap:
            upper = next_power_of_2(N)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            upper = next_power_of_2(chunk_len)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )


class MHATokenToKVPoolFP4(MHATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_v_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


class HybridLinearKVPool(KVCache):
    """KV cache with separate pools for full and linear attention layers."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
        use_nsa: bool = False,
        index_head_dim: Optional[int] = None,
        kv_cache_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        nsa_index_kpool: int = 1,
        max_running_requests: Optional[int] = None,
        layer_shard_rank: Optional[int] = None,
        layer_shard_size: int = 1,
        tail_extra_slots: int = 0,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        self.use_mla = use_mla
        self.use_nsa = use_nsa
        if use_mla:
            if use_nsa:
                # One NSA pool class handles both anchor (index_kpool == 1)
                # and dense kpool (index_kpool > 1) layouts. The kpool-only
                # knobs (tail_extra_slots, max_running_requests) are
                # ignored under anchor; ``__init__`` asserts the contract.
                self.full_kv_pool = NSATokenToKVPool(
                    size=size,
                    page_size=self.page_size,
                    dtype=dtype,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_head_dim,
                    layer_num=self.full_layer_nums,
                    device=self.device,
                    enable_memory_saver=enable_memory_saver,
                    start_layer=self.start_layer,
                    index_head_dim=index_head_dim,
                    kv_cache_dim=kv_cache_dim,
                    nsa_index_kpool=nsa_index_kpool,
                    tail_extra_slots=tail_extra_slots,
                    max_running_requests=max_running_requests,
                    layer_shard_rank=layer_shard_rank,
                    layer_shard_size=layer_shard_size,
                )
            else:

                TokenToKVPoolClass = MLATokenToKVPool

                if _is_npu:
                    from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                        NPUMLATokenToKVPool,
                    )

                    TokenToKVPoolClass = NPUMLATokenToKVPool

                self.full_kv_pool = TokenToKVPoolClass(
                    size=size,
                    page_size=self.page_size,
                    dtype=dtype,
                    layer_num=self.full_layer_nums,
                    device=device,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_head_dim,
                    enable_memory_saver=enable_memory_saver,
                )
        else:

            TokenToKVPoolClass = MHATokenToKVPool

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        self.slots_per_page = getattr(
            self.full_kv_pool, "slots_per_page", self.page_size
        )
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )

        if self.use_nsa:
            nsa_page_ptrs, nsa_page_lens, nsa_page_item_lens = (
                self.full_kv_pool.get_state_buf_infos()
            )
            mamba_data_ptrs.extend(nsa_page_ptrs)
            mamba_data_lens.extend(nsa_page_lens)
            mamba_item_lens.extend(nsa_page_item_lens)

            # Kpool tail buffers; anchor returns empty lists, so the
            # extend below no-ops without needing a type check.
            tail_ptrs, tail_lens, tail_item_lens = (
                self.full_kv_pool.get_tail_buf_infos()
            )
            mamba_data_ptrs.extend(tail_ptrs)
            mamba_data_lens.extend(tail_lens)
            mamba_item_lens.extend(tail_item_lens)
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_mamba_state_count(self) -> int:
        mamba_data_ptrs, _, _ = self.mamba_pool.get_contiguous_buf_infos()
        return len(mamba_data_ptrs)

    def get_nsa_page_state_count(self) -> int:
        if not self.use_nsa:
            return 0
        nsa_page_ptrs, _, _ = self.full_kv_pool.get_state_buf_infos()
        return len(nsa_page_ptrs)

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each mamba state tensor."""
        dims = list(self.mamba_pool.get_state_dim_per_tensor())
        if self.use_nsa:
            nsa_page_ptrs, _, _ = self.full_kv_pool.get_state_buf_infos()
            dims.extend([0] * len(nsa_page_ptrs))

            # Kpool tail buffers; anchor returns empty.
            tail_ptrs, _, _ = self.full_kv_pool.get_tail_buf_infos()
            dims.extend([0] * len(tail_ptrs))
        return dims

    def get_state_dim_components_per_tensor(self):
        """Get the packed sub-component sizes for each mamba state tensor."""
        comps = list(self.mamba_pool.get_state_dim_components_per_tensor())
        if self.use_nsa:
            nsa_page_ptrs, _, _ = self.full_kv_pool.get_state_buf_infos()
            comps.extend([[] for _ in nsa_page_ptrs])

            # Kpool tail buffers; anchor returns empty.
            tail_ptrs, _, _ = self.full_kv_pool.get_tail_buf_infos()
            comps.extend([[] for _ in tail_ptrs])
        return comps

    def maybe_get_custom_mem_pool(self):
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    @property
    def layer_shard_enabled(self) -> bool:
        return bool(getattr(self.full_kv_pool, "layer_shard_enabled", False))

    @property
    def layer_shard_rank(self) -> Optional[int]:
        return getattr(self.full_kv_pool, "layer_shard_rank", None)

    @property
    def layer_shard_size(self) -> int:
        return getattr(self.full_kv_pool, "layer_shard_size", 1)

    @property
    def layer_shard_start(self) -> int:
        return getattr(self.full_kv_pool, "layer_shard_start", self.start_layer)

    def _is_layer_owned(self, layer_id: int) -> bool:
        if not self.use_nsa:
            return True
        full_layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool._is_layer_owned(full_layer_id)

    def invalidate_index_buffer_for_layer(self, layer_id: int) -> None:
        if not self.use_nsa:
            return
        full_layer_id = self._transfer_full_attention_id(layer_id)
        self.full_kv_pool.invalidate_index_buffer_for_layer(full_layer_id)

    def register_layer_transfer_counter(
        self, layer_transfer_counter: "LayerDoneCounter"
    ):
        self.layer_transfer_counter = layer_transfer_counter
        # The layer-wise wait logic is executed at the Hybrid LinearPool level;
        # no additional wait is needed in the full_kv_pool
        self.full_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):

        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        kv_cpu = self.full_kv_pool.get_cpu_copy(indices, req_pool_index=req_pool_index)
        mamba_cpu = (
            self.mamba_pool.get_cpu_copy(mamba_indices)
            if mamba_indices is not None
            else None
        )
        return kv_cpu, mamba_cpu

    def load_cpu_copy(
        self, cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        kv_cpu, mamba_cpu = cache_cpu
        self.full_kv_pool.load_cpu_copy(kv_cpu, indices, req_pool_index=req_pool_index)
        if mamba_cpu is not None and mamba_indices is not None:
            self.mamba_pool.load_cpu_copy(mamba_cpu, mamba_indices)

    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    # Forwarders to the inner pool; defaults cover non-NSA inner pools.
    @property
    def nsa_kv_cache_store_fp8(self) -> bool:
        return getattr(self.full_kv_pool, "nsa_kv_cache_store_fp8", False)

    @property
    def index_kpool(self) -> int:
        return getattr(self.full_kv_pool, "index_kpool", 1)

    @property
    def kv_cache_dim(self) -> Optional[int]:
        return getattr(self.full_kv_pool, "kv_cache_dim", None)

    @property
    def index_head_dim(self) -> Optional[int]:
        return getattr(self.full_kv_pool, "index_head_dim", None)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)

    def prefetch_mla_kv_buffer(self, layer_id: int) -> None:
        if not self.use_mla or not hasattr(self.full_kv_pool, "prefetch_kv_buffer"):
            return
        if layer_id not in self.full_attention_layer_id_mapping:
            return

        full_layer_id = self._transfer_full_attention_id(layer_id)
        layer_transfer_idx = layer_id - self.start_layer
        self.full_kv_pool.prefetch_kv_buffer(
            full_layer_id,
            layer_transfer_counter=self.layer_transfer_counter,
            layer_transfer_idx=layer_transfer_idx,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        assert self.use_nsa, "set_index_k_scale_buffer called when use_nsa is False"
        layer_id = self._transfer_full_attention_id(layer_id)
        self.full_kv_pool.set_index_k_scale_buffer(
            layer_id, loc, index_k, index_k_scale
        )

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        assert (
            self.use_nsa
        ), "get_index_k_with_scale_buffer called when use_nsa is False"
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_index_k_with_scale_buffer(layer_id)

    def get_tail_buffers(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.use_nsa, "get_tail_buffers called when use_nsa is False"
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_tail_buffers(layer_id)

    # Forwarder so kpool-agnostic callers can read tail_extra_slots
    # uniformly through the hybrid wrapper. Default 0 = no scratch.
    @property
    def tail_extra_slots(self) -> int:
        return getattr(self.full_kv_pool, "tail_extra_slots", 0)

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        assert self.use_nsa, "get_index_k_continuous called when use_nsa is False"
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_index_k_continuous(layer_id, seq_len, page_indices)

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        assert self.use_nsa, "get_index_k_scale_continuous called when use_nsa is False"
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_index_k_scale_continuous(
            layer_id, seq_len, page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        assert self.use_nsa, "get_index_k_scale_buffer called when use_nsa is False"
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_index_k_scale_buffer(
            layer_id, seq_len_tensor, page_indices, seq_len_sum, max_seq_len
        )


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_nsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
        layer_shard_rank: Optional[int] = None,
        layer_shard_size: int = 1,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            layer_shard_rank,
            layer_shard_size,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_nsa = use_nsa
        self.nsa_kv_cache_store_fp8 = (
            use_nsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with nsa model, we assume the
        # override kv cache dim is correct and use it directly.
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.nsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_nsa:
            # NSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (
                            (self.size + self.page_size)
                            if self._is_layer_owned(self.start_layer + i)
                            else 0,
                            1,
                            self.kv_cache_dim,
                        ),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for i in range(self.layer_num)
                ]
                if self.layer_shard_enabled:
                    self.remote_kv_buffer = torch.empty(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    self.remote_kv_layer_id: Optional[int] = None
                    self.device_module = torch.get_device_module(self.device)
                    self.kv_broadcast_stream = self.device_module.Stream()
                    self.pending_remote_kv_layer_id: Optional[int] = None
                    self.pending_remote_kv_broadcast = False
        self._init_layer_broadcast_comm()

    def _clear_buffers(self):
        del self.kv_buffer
        if hasattr(self, "remote_kv_buffer"):
            del self.remote_kv_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer. Under layer sharding, return only the
        # buffers owned by the current rank.
        if self.layer_shard_enabled:
            owned_layer_ids = [
                i
                for i in range(self.layer_num)
                if self._is_layer_owned(self.start_layer + i)
            ]
        else:
            owned_layer_ids = list(range(self.layer_num))

        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in owned_layer_ids]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in owned_layer_ids]
        kv_item_lens = [
            (
                self.kv_buffer[i][0].nbytes * self.page_size
                if self.kv_buffer[i].shape[0] > 0
                else 0
            )
            for i in owned_layer_ids
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            return kv_buffer.view(self.dtype)

        return kv_buffer

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            return kv_buffer[..., : self.kv_lora_rank].view(self.dtype)
        return kv_buffer[..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        assert not self.nsa_kv_cache_store_fp8
        if (
            self.layer_shard_enabled
            and getattr(self, "pending_remote_kv_layer_id", None) == layer_id
        ):
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        if (
            self.layer_shard_enabled
            and getattr(self, "remote_kv_layer_id", None) == layer_id
        ):
            self.remote_kv_layer_id = None
        if not self._is_layer_owned(layer_id):
            return
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def _write_mla_kv_buffer(
        self,
        dst_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        if self.nsa_kv_cache_store_fp8:
            if _is_hip:
                # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
                # Fuse BF16/FP16 -> FP8 cast with paged KV write.
                fp8_dtype = (
                    torch.float8_e4m3fnuz if _is_fp8_fnuz else torch.float8_e4m3fn
                )
                set_mla_kv_buffer_triton_fp8_quant(
                    dst_buffer,
                    loc,
                    cache_k_nope,
                    cache_k_rope,
                    fp8_dtype,
                )
            else:
                # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
                # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
                # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
                cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                    cache_k_nope, cache_k_rope
                )

                # Reuse existing two-tensor write kernel (works with FP8 byte layout)
                # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
                # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
                set_mla_kv_buffer_triton(
                    dst_buffer,
                    loc,
                    cache_k_nope_fp8,
                    cache_k_rope_fp8,
                )
            return

        if cache_k_nope.dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
            cache_k_rope = cache_k_rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            cache_k_rope = cache_k_rope.view(self.store_dtype)

        set_mla_kv_buffer_triton(
            dst_buffer,
            loc,
            cache_k_nope,
            cache_k_rope,
        )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        remote_kv_updatable = False
        if self.layer_shard_enabled:
            if getattr(self, "pending_remote_kv_layer_id", None) == layer_id:
                self._finalize_pending_kv_broadcast(set_remote_layer_id=True)
            remote_kv_updatable = (
                getattr(self, "remote_kv_layer_id", None) == layer_id
            )

        if remote_kv_updatable:
            self._write_mla_kv_buffer(
                self.remote_kv_buffer, loc, cache_k_nope, cache_k_rope
            )
        if not self._is_layer_owned(layer_id):
            return

        self._write_mla_kv_buffer(
            self.kv_buffer[layer_id - self.start_layer],
            loc,
            cache_k_nope,
            cache_k_rope,
        )
        if (
            self.layer_shard_enabled
            and not remote_kv_updatable
            and getattr(self, "remote_kv_layer_id", None) == layer_id
        ):
            self.remote_kv_layer_id = None

    def _finalize_pending_kv_broadcast(
        self, *, set_remote_layer_id: bool = True
    ) -> None:
        if (
            not self.layer_shard_enabled
            or not getattr(self, "pending_remote_kv_broadcast", False)
        ):
            return
        self.device_module.current_stream().wait_stream(self.kv_broadcast_stream)
        self.pending_remote_kv_broadcast = False
        if set_remote_layer_id and self.pending_remote_kv_layer_id is not None:
            self.remote_kv_layer_id = self.pending_remote_kv_layer_id
        self.pending_remote_kv_layer_id = None

    def prefetch_kv_buffer(
        self,
        layer_id: int,
        layer_transfer_counter: Optional["LayerDoneCounter"] = None,
        layer_transfer_idx: Optional[int] = None,
    ) -> None:
        if not self.layer_shard_enabled:
            return
        if self.remote_kv_layer_id == layer_id:
            return
        if getattr(self, "pending_remote_kv_broadcast", False):
            if self.pending_remote_kv_layer_id == layer_id:
                return
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)

        local_idx = self._local_layer_idx(layer_id)
        src_tensor = (
            self.kv_buffer[local_idx] if self._is_layer_owned(layer_id) else None
        )
        if self.layer_broadcast_comm is None:
            self._broadcast_tensor_from_owner(
                self.remote_kv_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
            self.remote_kv_layer_id = layer_id
            return

        self.kv_broadcast_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(self.kv_broadcast_stream):
            if layer_transfer_counter is not None and layer_transfer_idx is not None:
                layer_transfer_counter.wait_until(layer_transfer_idx)
            self._broadcast_tensor_from_owner(
                self.remote_kv_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
        self.pending_remote_kv_layer_id = layer_id
        self.pending_remote_kv_broadcast = True

    def _get_broadcastable_kv_buffer(self, layer_id: int) -> torch.Tensor:
        if not self.layer_shard_enabled:
            return self.kv_buffer[layer_id - self.start_layer]
        if getattr(self, "pending_remote_kv_broadcast", False):
            if self.pending_remote_kv_layer_id == layer_id:
                self._finalize_pending_kv_broadcast(set_remote_layer_id=True)
            else:
                self._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        if self.remote_kv_layer_id != layer_id:
            local_idx = self._local_layer_idx(layer_id)
            src_tensor = (
                self.kv_buffer[local_idx] if self._is_layer_owned(layer_id) else None
            )
            self._broadcast_tensor_from_owner(
                self.remote_kv_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
            self.remote_kv_layer_id = layer_id
        return self.remote_kv_buffer

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )

        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            if self.kv_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            if self.kv_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()


class MLATokenToKVPoolFP4(MLATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        assert not self.nsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if self.nsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    KVFP4QuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class NSATokenToKVPool(MLATokenToKVPool):
    """NSA KV cache pool. One class, two layouts gated by ``nsa_index_kpool``:

    * **anchor** (``== 1``): one index slot per token; no per-req tail.
    * **dense kpool** (``> 1``): pack ``page_size // index_kpool`` pool
      entries per index_k page; per-req bf16 ring tail of width
      ``index_kpool + tail_extra_slots`` rooted at logical pos %
      ``TAIL_SIZE``. ``tail_extra_slots`` is scratch for EAGLE
      target_verify drafts; rejected drafts get naturally overwritten by
      the next decode/extend at the same logical positions. See
      ``nsa/kpool/`` for the ring invariants the kernels rely on.
    """

    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
        # Kpool knobs; anchor when nsa_index_kpool == 1.
        nsa_index_kpool: int = 1,
        tail_extra_slots: int = 0,
        max_running_requests: Optional[int] = None,
        layer_shard_rank: Optional[int] = None,
        layer_shard_size: int = 1,
    ):
        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )
        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_nsa=True,
            override_kv_cache_dim=override_dim,
            layer_shard_rank=layer_shard_rank,
            layer_shard_size=layer_shard_size,
        )
        assert index_head_dim == 128
        assert self.page_size == (1 if _is_hip else 64)
        assert (
            nsa_index_kpool > 1 or tail_extra_slots == 0
        ), f"tail_extra_slots is kpool-only; got {tail_extra_slots} with anchor."
        self.index_head_dim = index_head_dim
        self.index_kpool = nsa_index_kpool
        self.tail_extra_slots = tail_extra_slots
        # Anchor (==1) collapses to slots_per_page = page_size.
        self.slots_per_page = self.page_size // nsa_index_kpool

        # Per-layer index_k+scale buffer (anchor row = 64*132 = 8448 B;
        # dense row = 4*132 = 528 B), plus per-req bf16 tail ring (kpool
        # only). _tail_k/_score == None on anchor short-circuits getters.
        index_buf_size = size if index_buf_size is None else index_buf_size
        num_pages = (index_buf_size + self.page_size + 1) // self.page_size
        row_bytes = self.slots_per_page * (
            index_head_dim + index_head_dim // self.quant_block_size * 4
        )
        self._tail_k: Optional[List[torch.Tensor]] = None
        self._tail_score: Optional[List[torch.Tensor]] = None
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    (
                        num_pages if self._is_layer_owned(self.start_layer + i) else 0,
                        row_bytes,
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for i in range(layer_num)
            ]
            self.index_k_with_scale_buffer_ptrs = torch.tensor(
                [x.data_ptr() for x in self.index_k_with_scale_buffer],
                dtype=torch.uint64,
                device=self.device,
            )
            if self.layer_shard_enabled:
                self.remote_index_k_with_scale_buffer = torch.empty(
                    (num_pages, row_bytes),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                self.remote_index_layer_id: Optional[int] = None
            if nsa_index_kpool > 1:
                assert (
                    max_running_requests is not None
                ), "kpool layout requires max_running_requests for the per-req tail."
                # +1 mirrors req_to_token_pool.size + 1 (sentinel slot).
                shape = (
                    max_running_requests + 1,
                    nsa_index_kpool + tail_extra_slots,
                    index_head_dim,
                )
                self._tail_k = [
                    torch.zeros(shape, dtype=torch.bfloat16, device=device)
                    for _ in range(layer_num)
                ]
                self._tail_score = [
                    torch.zeros(shape, dtype=torch.bfloat16, device=device)
                    for _ in range(layer_num)
                ]

        self._finalize_allocation_log(size)

    def get_tail_buffers(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(tail_k, tail_score)`` for one NSA layer. Kpool only; anchor
        callers must gate on ``index_kpool > 1`` (assert is a backstop)."""
        assert (
            self._tail_k is not None and self._tail_score is not None
        ), "get_tail_buffers on anchor layout; gate with `index_kpool > 1`."
        idx = layer_id - self.start_layer
        return self._tail_k[idx], self._tail_score[idx]

    def get_tail_buf_infos(self):
        """``(data_ptrs, data_lens, item_lens)`` over ``2 * layer_num``
        tail tensors (all _tail_k then all _tail_score). Anchor returns
        empty lists so disagg/hybrid registration is layout-agnostic."""
        if self._tail_k is None:
            return [], [], []
        bufs = self._tail_k + self._tail_score
        return (
            [b.data_ptr() for b in bufs],
            [b.nbytes for b in bufs],
            [b[0].nbytes for b in bufs],
        )

    def _clear_buffers(self):
        super()._clear_buffers()
        del self.index_k_with_scale_buffer
        if hasattr(self, "remote_index_k_with_scale_buffer"):
            del self.remote_index_k_with_scale_buffer
        if self._tail_k is not None:
            del self._tail_k
            del self._tail_score

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_broadcastable_index_buffer(layer_id)
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_broadcastable_index_buffer(layer_id)
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_broadcastable_index_buffer(layer_id)
        self.prefetch_kv_buffer(layer_id)
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        self.invalidate_index_buffer_for_layer(layer_id)
        if not self._is_layer_owned(layer_id):
            return
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def invalidate_index_buffer_for_layer(self, layer_id: int) -> None:
        if (
            self.layer_shard_enabled
            and getattr(self, "remote_index_layer_id", None) == layer_id
        ):
            self.remote_index_layer_id = None

    def _get_broadcastable_index_buffer(self, layer_id: int) -> torch.Tensor:
        if not self.layer_shard_enabled:
            return self.index_k_with_scale_buffer[layer_id - self.start_layer]
        if self.remote_index_layer_id != layer_id:
            local_idx = self._local_layer_idx(layer_id)
            src_tensor = (
                self.index_k_with_scale_buffer[local_idx]
                if self._is_layer_owned(layer_id)
                else None
            )
            self._broadcast_tensor_from_owner(
                self.remote_index_k_with_scale_buffer,
                layer_id,
                src_tensor=src_tensor,
            )
            self.remote_index_layer_id = layer_id
        return self.remote_index_k_with_scale_buffer

    def get_state_buf_infos(self):
        if self.layer_shard_enabled:
            owned_layer_ids = [
                i
                for i in range(self.layer_num)
                if self._is_layer_owned(self.start_layer + i)
            ]
        else:
            owned_layer_ids = list(range(self.layer_num))

        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in owned_layer_ids
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in owned_layer_ids
        ]
        item_lens = [
            (
                self.index_k_with_scale_buffer[i][0].nbytes
                if self.index_k_with_scale_buffer[i].shape[0] > 0
                else 0
            )
            for i in owned_layer_ids
        ]
        return data_ptrs, data_lens, item_lens

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        # super() trailing-syncs after its own D2Hs, so the queue is empty
        # here; we issue our copies non_blocking and sync once at the end.
        kv_cache_cpu = super().get_cpu_copy(indices)

        # Page-indexed D2H of the index_k_with_scale_buffer.
        page_indices = indices[:: self.page_size] // self.page_size
        chunk_size = self.cpu_offloading_chunk_size
        # token-level -> page-level chunk size.
        page_chunk_size = max(1, chunk_size // self.page_size)
        index_k_cpu: List[List[torch.Tensor]] = []
        for layer_id in range(self.layer_num):
            index_k_cpu.append([])
            if self.index_k_with_scale_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = self.index_k_with_scale_buffer[layer_id][
                    chunk_page_indices
                ].to("cpu", non_blocking=True)
                index_k_cpu[-1].append(idx_cpu)

        result = {"kv": kv_cache_cpu, "index_k": index_k_cpu}

        # Kpool-only per-req tail buffer; anchor pools skip this entirely.
        if self._tail_k is not None and req_pool_index is not None:
            result["tail_k"] = [
                self._tail_k[i][req_pool_index].to("cpu", non_blocking=True)
                for i in range(self.layer_num)
            ]
            result["tail_score"] = [
                self._tail_score[i][req_pool_index].to("cpu", non_blocking=True)
                for i in range(self.layer_num)
            ]

        torch.cuda.synchronize()  # drains all non_blocking D2H above.
        return result

    def load_cpu_copy(
        self, kv_cache_cpu_dict, indices, mamba_indices=None, req_pool_index=None
    ):
        # super() trailing-syncs after its own H2Ds; we follow the same
        # non_blocking-issue + single-sync pattern.
        super().load_cpu_copy(kv_cache_cpu_dict["kv"], indices)

        # Page-indexed H2D of the index_k_with_scale_buffer.
        page_indices = indices[:: self.page_size] // self.page_size
        index_k_cpu = kv_cache_cpu_dict["index_k"]
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        idx_dev = self.index_k_with_scale_buffer[0].device
        for layer_id in range(self.layer_num):
            if self.index_k_with_scale_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                self.index_k_with_scale_buffer[layer_id][chunk_page_indices] = (
                    idx_cpu.to(idx_dev, non_blocking=True)
                )

        # Restore tail into the (possibly newly allocated) req_pool_idx slot.
        if (
            self._tail_k is not None
            and req_pool_index is not None
            and "tail_k" in kv_cache_cpu_dict
        ):
            tail_k_cpu = kv_cache_cpu_dict["tail_k"]
            tail_score_cpu = kv_cache_cpu_dict["tail_score"]
            for i in range(self.layer_num):
                self._tail_k[i][req_pool_index] = tail_k_cpu[i].to(
                    self._tail_k[i].device, non_blocking=True
                )
                self._tail_score[i][req_pool_index] = tail_score_cpu[i].to(
                    self._tail_score[i].device, non_blocking=True
                )

        torch.cuda.synchronize()  # drains all non_blocking H2D above.

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                self.k_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

                # [size, head_num, heavy_channel_num] for each layer
                self.label_buffer = [
                    torch.zeros(
                        (size + 1, head_num, heavy_channel_num),
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        self.label_buffer[layer_id - self.start_layer][loc] = cache_label


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)
