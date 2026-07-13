from __future__ import annotations

import os
import random
from collections import deque
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Type, overload

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base.conn import KVArgs, StateType
    from sglang.srt.disaggregation.common.conn import (
        CommonKVBootstrapServer,
        CommonKVManager,
        CommonKVReceiver,
        CommonKVSender,
    )
    from sglang.srt.managers.schedule_batch import Req

#########################
# Constants & Enums
#########################
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"


#########################
# Synchronization
#########################

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))


def poll_and_all_reduce(pollers, gloo_group: dist.ProcessGroup):
    # at a certain prob, the poll is failed to simulate failure
    if FAILURE_PROB > 0:
        from sglang.srt.disaggregation.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_attn_cp_tp_group(
    pollers,
    attn_cp_cpu_group: dist.ProcessGroup,
    attn_tp_cpu_group: dist.ProcessGroup,
):
    # First sync across attn-tp ranks so all TP participants for a given (dp, cp)
    # shard observe the same status transitions.
    polls = poll_and_all_reduce(pollers, attn_tp_cpu_group)

    # Then sync across attn-cp ranks, so all TPxCP participants in one DP shard
    # converge to the same global status.
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(
        tensor_to_reduce,
        op=dist.ReduceOp.MIN,
        group=attn_cp_cpu_group,
    )
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_with_staging(
    decode_reqs, staging_handler, gloo_group: dist.ProcessGroup
):
    """Staging-aware polling: advance scatter, demote incomplete transfers, all_reduce."""
    from sglang.srt.disaggregation.base import KVPoll

    for decode_req in decode_reqs:
        if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
            decode_req
        ):
            staging_handler.advance_scatter(decode_req)

    raw_polls = [int(dr.kv_receiver.poll()) for dr in decode_reqs]
    for i, decode_req in enumerate(decode_reqs):
        if raw_polls[i] == int(KVPoll.Success):
            if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
                decode_req
            ):
                raw_polls[i] = int(KVPoll.Transferring)
    poll_tensor = torch.tensor(raw_polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(poll_tensor, op=dist.ReduceOp.MIN, group=gloo_group)
    return poll_tensor.tolist()


#########################
# Metadata Buffers
#########################


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self) -> Optional[int]:
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        self.free_slots.append(free_index)


class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        custom_mem_pool: torch.cuda.MemPool = None,
        mtp_topk_indices_dim: int = 0,
    ):
        self.custom_mem_pool = custom_mem_pool
        self.mtp_topk_indices_dim = mtp_topk_indices_dim
        bootstrap_room_dtype = torch.uint64
        device = "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            device = "npu"
            # TODO: Fix me when npu backend supports torch.uint64
            bootstrap_room_dtype = torch.int64
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            device = "cpu"
        elif envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() == "INTRA_NODE_NVLINK":
            device = "cpu"
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_token_logprobs_val = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_token_logprobs_idx = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_top_logprobs_val = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.float32, device=device
            )
            self.output_top_logprobs_idx = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.int32, device=device
            )
            # For PD + spec decode
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            self.output_hidden_states = torch.zeros(
                (size, hidden_size), dtype=hidden_states_dtype, device=device
            )
            if self.mtp_topk_indices_dim > 0:
                self.output_mtp_topk_indices = torch.zeros(
                    (size, self.mtp_topk_indices_dim),
                    dtype=torch.int32,
                    device=device,
                )
                self.output_mtp_topk_indices_valid = torch.zeros(
                    (size, 1), dtype=torch.uint8, device=device
                )
            else:
                self.output_mtp_topk_indices = None
                self.output_mtp_topk_indices_valid = None
            # Request validation: store bootstrap_room to detect metadata corruption
            self.bootstrap_room = torch.zeros(
                (size, 8), dtype=bootstrap_room_dtype, device=device
            )

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.cached_tokens.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.output_topk_p.data_ptr(),
            self.output_topk_index.data_ptr(),
            self.output_hidden_states.data_ptr(),
            self.bootstrap_room.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.cached_tokens.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.output_topk_p.nbytes,
            self.output_topk_index.nbytes,
            self.output_hidden_states.nbytes,
            self.bootstrap_room.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.cached_tokens[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.output_topk_p[0].nbytes,
            self.output_topk_index[0].nbytes,
            self.output_hidden_states[0].nbytes,
            self.bootstrap_room[0].nbytes,
        ]
        if self.output_mtp_topk_indices is not None:
            ptrs[-1:-1] = [
                self.output_mtp_topk_indices.data_ptr(),
                self.output_mtp_topk_indices_valid.data_ptr(),
            ]
            data_lens[-1:-1] = [
                self.output_mtp_topk_indices.nbytes,
                self.output_mtp_topk_indices_valid.nbytes,
            ]
            item_lens[-1:-1] = [
                self.output_mtp_topk_indices[0].nbytes,
                self.output_mtp_topk_indices_valid[0].nbytes,
            ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx].clone(),
            self.cached_tokens[idx].clone(),
            self.output_token_logprobs_val[idx].clone(),
            self.output_token_logprobs_idx[idx].clone(),
            self.output_top_logprobs_val[idx].clone(),
            self.output_top_logprobs_idx[idx].clone(),
            self.output_topk_p[idx].clone(),
            self.output_topk_index[idx].clone(),
            self.output_hidden_states[idx].clone(),
            (
                self.output_mtp_topk_indices[idx].clone()
                if self.output_mtp_topk_indices is not None
                and bool(self.output_mtp_topk_indices_valid[idx, 0].item())
                else None
            ),
            self.bootstrap_room[idx].clone(),
        )

    def set_buf(self, req: Req):

        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        self.cached_tokens[req.metadata_buffer_index][1] = req.cached_tokens_device
        self.cached_tokens[req.metadata_buffer_index][2] = req.cached_tokens_host
        self.cached_tokens[req.metadata_buffer_index][3] = req.cached_tokens_storage
        if req.return_logprob:
            if req.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_val[0]
                )
            if req.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_idx[0]
                )

            if req.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                )
            if req.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                )
        # For PD + spec decode
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )
        if self.output_mtp_topk_indices is not None:
            valid = req.mtp_topk_indices_tensor is not None
            self.output_mtp_topk_indices_valid[
                req.metadata_buffer_index, 0
            ] = valid
            if valid:
                self.output_mtp_topk_indices[req.metadata_buffer_index].copy_(
                    req.mtp_topk_indices_tensor
                )
        # Store bootstrap_room for validation on decode side
        self.bootstrap_room[req.metadata_buffer_index, 0] = (
            req.bootstrap_room if req.bootstrap_room is not None else 0
        )


#########################
# Transfer Backend
#########################


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


class KVClassType(Enum):
    KVARGS = "kvargs"
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.KVARGS]
) -> Type[KVArgs]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.MANAGER]
) -> Type[CommonKVManager]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.SENDER]
) -> Type[CommonKVSender]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.RECEIVER]
) -> Type[CommonKVReceiver]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.BOOTSTRAP_SERVER]
) -> Type[CommonKVBootstrapServer]: ...


def get_kv_class(
    transfer_backend: TransferBackend, class_type: KVClassType
) -> Optional[Type]:
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.MORI:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mori import (
            MoriKVBootstrapServer,
            MoriKVManager,
            MoriKVReceiver,
            MoriKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MoriKVManager,
            KVClassType.SENDER: MoriKVSender,
            KVClassType.RECEIVER: (MoriKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MoriKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.ASCEND:
        from sglang.srt.disaggregation.ascend import (
            AscendKVBootstrapServer,
            AscendKVManager,
            AscendKVReceiver,
            AscendKVSender,
        )
        from sglang.srt.disaggregation.base import KVArgs

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: AscendKVManager,
            KVClassType.SENDER: AscendKVSender,
            KVClassType.RECEIVER: (AscendKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: AscendKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.fake import (
            FakeKVManager,
            FakeKVReceiver,
            FakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: FakeKVManager,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


def page_indices_to_cp_rank_page_indices(
    page_indices: np.ndarray,
    total_pages: int,
    cp_rank: int,
    cp_size: int,
) -> np.ndarray:
    """
    Filter page_indices (which are *global* page ids in the KV pool) to those
    belonging to the given CP rank for this request.

    For a single request, its pages occupy a contiguous global range
    [first_page, first_page + total_pages). We first compute the local
    split [0, total_pages) across cp_size ranks, then shift that local
    range by first_page back into the global page id space and take
    the intersection with page_indices.

    Returns:
        Subset of page_indices that fall in this rank's global
        [start_page, end_page) slice for the given CP rank.
    """
    if cp_size <= 1:
        return page_indices

    if page_indices.size == 0:
        return np.asarray(page_indices)

    first_page = int(page_indices.min())
    base = total_pages // cp_size
    rem = total_pages % cp_size

    if rem == 0:
        local_start = cp_rank * base
        local_end = local_start + base
    else:
        local_start = cp_rank * base + min(cp_rank, rem)
        n_pages = base + (1 if cp_rank < rem else 0)
        local_end = local_start + n_pages

    # Map back to global page ids.
    start_page = first_page + local_start
    end_page = first_page + local_end

    mask = (page_indices >= start_page) & (page_indices < end_page)
    return np.asarray(page_indices)[mask]


def _get_cp_rank_page_bounds(
    total_pages: int, cp_rank: int, cp_size: int
) -> Tuple[int, int]:
    base = total_pages // cp_size
    rem = total_pages % cp_size
    local_start = cp_rank * base + min(cp_rank, rem)
    n_pages = base + (1 if cp_rank < rem else 0)
    return local_start, local_start + n_pages


def filter_kv_indices_for_cp_rank(
    kv_mgr: CommonKVManager,
    kv_indices: np.ndarray,
    index_slice: slice,
    total_pages: Optional[int] = None,
) -> Tuple[np.ndarray, slice]:
    """Partition kv_indices/index_slice across CP ranks for KV transfer.

    Every CP rank holds the identical per-request page list in the normal
    non-layer-split path. To avoid all ranks sending the same pages, each rank
    sends a contiguous positional slice of that page list; together the slices
    tile the request exactly once.
    """
    if total_pages is None:
        total_pages = len(kv_indices)
    cp_rank = kv_mgr.attn_cp_rank
    cp_size = kv_mgr.attn_cp_size

    if cp_size <= 1:
        return kv_indices, index_slice

    rank_start, rank_end = _get_cp_rank_page_bounds(total_pages, cp_rank, cp_size)
    chunk_start = index_slice.start if index_slice.start is not None else 0
    chunk_end = index_slice.stop if index_slice.stop is not None else total_pages
    first_pos = max(rank_start, chunk_start) - chunk_start
    last_pos = min(rank_end, chunk_end) - chunk_start

    if last_pos <= first_pos:
        new_kv_indices = kv_indices[:0]
        new_index_slice = slice(chunk_start, chunk_start)
    else:
        new_kv_indices = kv_indices[first_pos:last_pos]
        new_index_slice = slice(
            chunk_start + first_pos,
            chunk_start + last_pos,
        )
    return new_kv_indices, new_index_slice


#########################
# Misc
#########################


def is_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, (MLATokenToKVPool, DeepSeekV4TokenToKVPool))


def is_hybrid_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MLATokenToKVPool

    return isinstance(target_kv_pool, HybridLinearKVPool) and isinstance(
        target_kv_pool.full_kv_pool, MLATokenToKVPool
    )


def append_state_component(
    kv_args: KVArgs,
    state_type: StateType,
    data_ptrs: List[int],
    data_lens: List[int],
    item_lens: List[int],
    dim_per_tensor: Optional[List[int]] = None,
    dim_components_per_tensor: Optional[List[List[int]]] = None,
    dim_outer_per_tensor: Optional[List[int]] = None,
) -> None:
    """Append one side-state component.

    Caller orders state_types consistently on prefill and decode sides.
    A no-op when the buffer group is empty.
    """
    if not data_ptrs:
        return
    kv_args.state_types.append(state_type)
    kv_args.state_data_ptrs.append(data_ptrs)
    kv_args.state_data_lens.append(data_lens)
    kv_args.state_item_lens.append(item_lens)
    kv_args.state_dim_per_tensor.append(dim_per_tensor or [])
    kv_args.state_dim_components_per_tensor.append(dim_components_per_tensor or [])
    kv_args.state_dim_outer_per_tensor.append(dim_outer_per_tensor or [])


def setup_state_kv_args(
    kv_args: KVArgs,
    token_to_kv_pool,
    draft_token_to_kv_pool=None,
    total_kv_layers: int = None,
    req_to_token_pool=None,
) -> None:
    """Populate ``kv_args`` state-buffer fields from the given pool.
    Shared by prefill and decode bootstrap paths so the state_type dispatch
    lives in one place. Components are appended in this order when present:

        mamba(MAMBA) -> target_nsa(NSA) -> target_tail(NSA_TAIL)
                     -> draft_nsa(NSA)  -> draft_tail(NSA_TAIL)

    NSA tail components are emitted only when the pool exposes dense kpool tail
    buffers. Anchor-layout NSA keeps the old component order.
    """
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, NSATokenToKVPool

    kv_args.state_types = []
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_dim_per_tensor = []
    kv_args.state_dim_components_per_tensor = []
    kv_args.state_dim_outer_per_tensor = []

    def _append_nsa(pool):
        append_state_component(kv_args, StateType.NSA, *pool.get_state_buf_infos())
        if getattr(pool, "index_kpool", 1) > 1 and hasattr(
            pool, "get_tail_buf_infos"
        ):
            append_state_component(
                kv_args, StateType.NSA_TAIL, *pool.get_tail_buf_infos()
            )

    def _append_draft_nsa():
        draft_nsa_pool = getattr(
            draft_token_to_kv_pool, "full_kv_pool", draft_token_to_kv_pool
        )
        if isinstance(draft_nsa_pool, NSATokenToKVPool):
            _append_nsa(draft_nsa_pool)

    if hasattr(token_to_kv_pool, "get_state_buf_infos"):
        data_ptrs, data_lens, item_lens = token_to_kv_pool.get_state_buf_infos()

        # DeepSeekV4TokenToKVPool inherits BaseSWAKVPool; its heterogeneous
        # state list is described per-entry via get_state_buf_infos.
        if isinstance(token_to_kv_pool, BaseSWAKVPool):
            append_state_component(
                kv_args, StateType.SWA, data_ptrs, data_lens, item_lens
            )
        elif isinstance(token_to_kv_pool, HybridLinearKVPool):
            dim = (
                token_to_kv_pool.get_state_dim_per_tensor()
                if hasattr(token_to_kv_pool, "get_state_dim_per_tensor")
                else None
            )
            dim_components = (
                token_to_kv_pool.get_state_dim_components_per_tensor()
                if hasattr(token_to_kv_pool, "get_state_dim_components_per_tensor")
                else None
            )
            dim_outer = (
                token_to_kv_pool.get_state_dim_outer_per_tensor()
                if hasattr(token_to_kv_pool, "get_state_dim_outer_per_tensor")
                else None
            )
            append_state_component(
                kv_args,
                StateType.MAMBA,
                data_ptrs,
                data_lens,
                item_lens,
                dim,
                dim_components,
                dim_outer,
            )
            if token_to_kv_pool.use_nsa:
                _append_nsa(token_to_kv_pool.full_kv_pool)
                _append_draft_nsa()
        elif isinstance(token_to_kv_pool, (NSATokenToKVPool, NPUMLATokenToKVPool)):
            if isinstance(token_to_kv_pool, NPUMLATokenToKVPool):
                kv_args.kv_buf_groups = (
                    len(kv_args.kv_data_ptrs) // token_to_kv_pool.layer_num
                )
                kv_args.total_kv_layers = total_kv_layers
            else:
                _append_nsa(token_to_kv_pool)
                _append_draft_nsa()

    if (
        StateType.MAMBA not in kv_args.state_types
        and req_to_token_pool is not None
        and hasattr(req_to_token_pool, "get_state_buf_infos")
    ):
        data_ptrs, data_lens, item_lens = req_to_token_pool.get_state_buf_infos()
        if data_ptrs:
            dim = (
                req_to_token_pool.get_state_dim_per_tensor()
                if hasattr(req_to_token_pool, "get_state_dim_per_tensor")
                else None
            )
            dim_components = (
                req_to_token_pool.get_state_dim_components_per_tensor()
                if hasattr(req_to_token_pool, "get_state_dim_components_per_tensor")
                else None
            )
            dim_outer = (
                req_to_token_pool.get_state_dim_outer_per_tensor()
                if hasattr(req_to_token_pool, "get_state_dim_outer_per_tensor")
                else None
            )
            append_state_component(
                kv_args,
                StateType.MAMBA,
                data_ptrs,
                data_lens,
                item_lens,
                dim,
                dim_components,
                dim_outer,
            )


def build_state_indices(
    *,
    token_to_kv_pool,
    draft_token_to_kv_pool=None,
    req_to_token,
    req_pool_idx: int,
    seq_len: int,
    page_size: int,
    mamba_index: Optional[int] = None,
    swa_window_size: Optional[int] = None,
    swa_translate_loc=None,
) -> Optional[List[List[int]]]:
    """Build per-request state indices in the same order as setup_state_kv_args.

    This is the request-side dual of ``setup_state_kv_args``. The connection
    layer pairs component ``i`` in ``state_indices`` with component ``i`` in
    ``kv_args.state_types`` / ``state_data_ptrs``. Keep this order in sync with
    ``setup_state_kv_args`` when adding new state components.
    """
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.common import kv_to_page_indices
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, NSATokenToKVPool

    components: List[List[int]] = []

    def _nsa_pages(pool):
        kv = req_to_token[req_pool_idx, :seq_len]
        return kv_to_page_indices(kv.cpu().numpy(), pool.page_size).tolist()

    def _append_nsa(pool):
        components.append(_nsa_pages(pool))
        if getattr(pool, "index_kpool", 1) > 1 and hasattr(
            pool, "get_tail_buf_infos"
        ):
            # Dense kpool tail is a ring of size index_kpool + tail_extra_slots.
            # Only the open pool's accumulated prefix slots contain live data.
            pool_size = pool.index_kpool
            tail_size = pool_size + getattr(pool, "tail_extra_slots", 0)
            n_valid = seq_len % pool_size
            if n_valid:
                start_phys = (seq_len - n_valid) % tail_size
                first_n = min(n_valid, tail_size - start_phys)
                second_n = n_valid - first_n
                components.append(
                    [req_pool_idx, start_phys, first_n, 0, second_n, tail_size]
                )
            else:
                components.append([])

    def _append_draft_nsa():
        draft_nsa_pool = getattr(
            draft_token_to_kv_pool, "full_kv_pool", draft_token_to_kv_pool
        )
        if isinstance(draft_nsa_pool, NSATokenToKVPool):
            _append_nsa(draft_nsa_pool)

    if isinstance(token_to_kv_pool, HybridLinearKVPool):
        assert mamba_index is not None, "HybridLinearKVPool requires mamba_index"
        components.append([int(mamba_index)])
        if token_to_kv_pool.use_nsa and isinstance(
            token_to_kv_pool.full_kv_pool, NSATokenToKVPool
        ):
            _append_nsa(token_to_kv_pool.full_kv_pool)
            _append_draft_nsa()
    elif isinstance(token_to_kv_pool, BaseSWAKVPool):
        assert (
            swa_window_size is not None and swa_translate_loc is not None
        ), "SWA state indices require window size and SWA loc translator"
        window_start = max(0, seq_len - swa_window_size)
        window_start = (window_start // page_size) * page_size
        window_kv = req_to_token[req_pool_idx, window_start:seq_len]
        swa_loc = swa_translate_loc(window_kv)
        components.append(
            kv_to_page_indices(swa_loc.cpu().numpy(), page_size).tolist()
        )
    elif isinstance(token_to_kv_pool, NSATokenToKVPool):
        _append_nsa(token_to_kv_pool)
        _append_draft_nsa()

    if not components:
        return None
    return components


def prepare_abort(req: Req, error_message: str, status_code=None):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.input_token_logprobs_val = []
        req.input_token_logprobs_idx = []
        req.input_top_logprobs_val = []
        req.input_top_logprobs_idx = []
        req.input_token_ids_logprobs_val = []
        req.input_token_ids_logprobs_idx = []
