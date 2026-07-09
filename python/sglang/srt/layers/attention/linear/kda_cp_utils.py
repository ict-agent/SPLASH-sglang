from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import List, Sequence

import torch

from sglang.srt.server_args import get_global_server_args


@dataclass
class KDAPrefillContextParallelMetadata:
    """Metadata for KDA plain-split prefill context parallelism.

    KDA plain-split partitions the current packed prefill chunk by contiguous
    token ranges. Request boundaries still matter because KDA kernels consume
    varlen metadata and prefix-cache state is tracked per request.
    """

    total_tokens: int
    cp_rank: int
    cp_size: int
    local_start: int
    local_end: int
    local_num_tokens: int
    local_seq_lens_cpu: List[int]
    local_req_indices_cpu: List[int]
    local_req_extend_offsets_cpu: List[int]
    local_req_global_starts_cpu: List[int]
    local_req_global_ends_cpu: List[int]
    local_segment_global_starts_cpu: List[int]
    local_segment_global_ends_cpu: List[int]
    local_query_start_loc: torch.Tensor


def is_kda_prefill_cp_enabled() -> bool:
    return bool(
        getattr(get_global_server_args(), "enable_kda_prefill_context_parallel", False)
    )


def is_kda_prefill_cp_plain_split() -> bool:
    return (
        is_kda_prefill_cp_enabled()
        and get_global_server_args().kda_prefill_cp_mode == "plain-split"
    )


def _plain_split_bounds(total_tokens: int, cp_rank: int, cp_size: int) -> tuple[int, int]:
    base = total_tokens // cp_size
    rem = total_tokens % cp_size
    start = cp_rank * base + min(cp_rank, rem)
    end = start + base + (1 if cp_rank < rem else 0)
    return start, end


def kda_cp_plain_split_bounds(
    total_tokens: int, cp_rank: int, cp_size: int
) -> tuple[int, int]:
    return _plain_split_bounds(total_tokens, cp_rank, cp_size)


def kda_cp_owner_of_global_token(
    global_token_idx: int, total_tokens: int, cp_size: int
) -> int:
    if global_token_idx < 0 or global_token_idx >= total_tokens:
        raise ValueError(
            f"global_token_idx={global_token_idx} is outside [0, {total_tokens})"
        )
    base = total_tokens // cp_size
    rem = total_tokens % cp_size
    large = (base + 1) * rem
    if global_token_idx < large:
        return global_token_idx // (base + 1)
    return rem + (global_token_idx - large) // base


def prepare_kda_prefill_cp_metadata(
    *,
    total_tokens: int,
    extend_seq_lens_cpu: Sequence[int],
    cp_rank: int,
    cp_size: int,
    device: torch.device,
) -> KDAPrefillContextParallelMetadata:
    local_start, local_end = _plain_split_bounds(total_tokens, cp_rank, cp_size)
    local_seq_lens_cpu: List[int] = []
    local_req_indices_cpu: List[int] = []
    local_req_extend_offsets_cpu: List[int] = []
    local_req_global_starts_cpu: List[int] = []
    local_req_global_ends_cpu: List[int] = []
    local_segment_global_starts_cpu: List[int] = []
    local_segment_global_ends_cpu: List[int] = []

    req_starts = [0] + list(accumulate(int(x) for x in extend_seq_lens_cpu))
    for req_idx, (req_start, req_end) in enumerate(zip(req_starts, req_starts[1:])):
        seg_start = max(local_start, req_start)
        seg_end = min(local_end, req_end)
        if seg_start >= seg_end:
            continue
        local_seq_lens_cpu.append(seg_end - seg_start)
        local_req_indices_cpu.append(req_idx)
        local_req_extend_offsets_cpu.append(seg_start - req_start)
        local_req_global_starts_cpu.append(req_start)
        local_req_global_ends_cpu.append(req_end)
        local_segment_global_starts_cpu.append(seg_start)
        local_segment_global_ends_cpu.append(seg_end)

    query_start_values = [0] + list(accumulate(local_seq_lens_cpu))
    local_query_start_loc = torch.tensor(
        query_start_values, dtype=torch.int32, device=device
    )

    return KDAPrefillContextParallelMetadata(
        total_tokens=total_tokens,
        cp_rank=cp_rank,
        cp_size=cp_size,
        local_start=local_start,
        local_end=local_end,
        local_num_tokens=local_end - local_start,
        local_seq_lens_cpu=local_seq_lens_cpu,
        local_req_indices_cpu=local_req_indices_cpu,
        local_req_extend_offsets_cpu=local_req_extend_offsets_cpu,
        local_req_global_starts_cpu=local_req_global_starts_cpu,
        local_req_global_ends_cpu=local_req_global_ends_cpu,
        local_segment_global_starts_cpu=local_segment_global_starts_cpu,
        local_segment_global_ends_cpu=local_segment_global_ends_cpu,
        local_query_start_loc=local_query_start_loc,
    )


def kda_cp_plain_split_tensor(
    input_tensor: torch.Tensor, metadata: KDAPrefillContextParallelMetadata
) -> torch.Tensor:
    return input_tensor[metadata.local_start : metadata.local_end].contiguous()


def kda_cp_continuation_segment_index(
    metadata: KDAPrefillContextParallelMetadata,
) -> int | None:
    """Index of the local segment that continues a sequence started on a
    previous rank (the only segment needing a cross-rank incoming state), or
    ``None`` if this rank starts every sequence it holds."""
    for local_idx, (seg_start, req_start) in enumerate(
        zip(
            metadata.local_segment_global_starts_cpu,
            metadata.local_req_global_starts_cpu,
        )
    ):
        if seg_start > req_start:
            return local_idx
    return None


def build_kda_fla_cp_context(
    metadata: KDAPrefillContextParallelMetadata,
    group,
    conv1d_kernel_size: int | None = None,
):
    """Build a forward-only ``FLACPContext`` from SGLang's remainder-aware
    plain-split metadata (do NOT use flash-linear-attention's ``build_cp_context``,
    which re-derives a uniform ``total // cp_size`` split and drops the remainder).
    """
    from sglang.srt.layers.attention.fla.cp import FLACPContext

    cont_idx = kda_cp_continuation_segment_index(metadata)
    is_first_rank = cont_idx is None
    pre_num_ranks = 0
    pre_num_conv_tokens = 0
    if not is_first_rank:
        req_start = metadata.local_req_global_starts_cpu[cont_idx]
        seg_start = metadata.local_segment_global_starts_cpu[cont_idx]
        prev_owner = kda_cp_owner_of_global_token(
            seg_start - 1, metadata.total_tokens, metadata.cp_size
        )
        pre_num_ranks = metadata.cp_rank - prev_owner
        pre_num_conv_tokens = max(0, seg_start - req_start)

    # Whether the last local segment continues onto a following rank.
    is_last_rank = True
    for local_idx in range(len(metadata.local_seq_lens_cpu) - 1, -1, -1):
        if (
            metadata.local_segment_global_ends_cpu[local_idx]
            < metadata.local_req_global_ends_cpu[local_idx]
        ):
            is_last_rank = False
            break

    return FLACPContext(
        group=group,
        cu_seqlens=metadata.local_query_start_loc,
        cu_seqlens_cpu=None,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        pre_num_ranks=pre_num_ranks,
        pre_num_conv_tokens=pre_num_conv_tokens,
    )
