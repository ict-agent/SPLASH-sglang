from __future__ import annotations

import logging
import os

import torch

from sglang.srt.layers.dp_attention import dp_gather_replicate, dp_scatter
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

try:
    from mla_only_dp_kernels import (
        is_available as _transfer_kernel_is_available,
        pack_dp_rows_to_tp_shards as _pack_dp_rows_to_tp_shards_kernel,
        pack_two_tp_head_shards as _pack_two_tp_head_shards_kernel,
        tp_heads_to_dp_rows as _tp_heads_to_dp_rows_kernel,
        unpack_two_tp_head_shards as _unpack_two_tp_head_shards_kernel,
    )
except Exception:
    _transfer_kernel_is_available = None
    _pack_dp_rows_to_tp_shards_kernel = None
    _pack_two_tp_head_shards_kernel = None
    _tp_heads_to_dp_rows_kernel = None
    _unpack_two_tp_head_shards_kernel = None

_warned_missing_kernel = False


def _fused_q_transfer_max_rows() -> int:
    return int(os.getenv("SGLANG_MLA_ONLY_DP_FUSED_Q_TRANSFER_MAX_ROWS", "32"))


def _fused_q_transfer_min_rows() -> int:
    return int(os.getenv("SGLANG_MLA_ONLY_DP_FUSED_Q_TRANSFER_MIN_ROWS", "512"))


def _use_transfer_kernel(x: torch.Tensor) -> bool:
    if os.getenv("SGLANG_MLA_ONLY_DP_DISABLE_TRANSFER_KERNEL") == "1":
        return False
    return bool(
        x.is_cuda
        and x.is_contiguous()
        and _transfer_kernel_is_available is not None
        and _transfer_kernel_is_available()
        and _tp_heads_to_dp_rows_kernel is not None
        and _pack_dp_rows_to_tp_shards_kernel is not None
    )


def _use_pair_transfer_kernel(q_nope: torch.Tensor, q_pe: torch.Tensor) -> bool:
    if os.getenv("SGLANG_MLA_ONLY_DP_DISABLE_TRANSFER_KERNEL") == "1":
        return False
    if not q_nope.is_cuda or not q_pe.is_cuda:
        return False
    if q_nope.device != q_pe.device or q_nope.dtype != q_pe.dtype:
        return False
    if q_nope.stride(-1) != 1 or q_pe.stride(-1) != 1:
        return False
    return bool(
        _transfer_kernel_is_available is not None
        and _transfer_kernel_is_available()
        and _pack_two_tp_head_shards_kernel is not None
        and _unpack_two_tp_head_shards_kernel is not None
    )


def _use_fused_q_transfer(
    token_counts: list[int], q_nope: torch.Tensor, q_pe: torch.Tensor
) -> bool:
    max_rows = max(token_counts, default=0)
    return (
        (
            max_rows <= _fused_q_transfer_max_rows()
            or max_rows >= _fused_q_transfer_min_rows()
        )
        and _use_pair_transfer_kernel(q_nope, q_pe)
    )


def _warn_kernel_fallback_once() -> None:
    global _warned_missing_kernel
    if _warned_missing_kernel:
        return
    _warned_missing_kernel = True
    logger.warning(
        "mla_only_dp_kernels is disabled or unavailable; using PyTorch packing "
        "around the MLA-only-DP all-to-all transfer."
    )


def _get_transfer_token_counts(forward_batch, tp_size: int) -> list[int]:
    global_num_tokens = forward_batch.global_num_tokens_cpu
    if global_num_tokens is None:
        raise RuntimeError("MLA-only-DP transfer requires global_num_tokens_cpu.")

    counts = [int(count) for count in global_num_tokens]
    if len(counts) != tp_size:
        raise RuntimeError(
            "MLA-only-DP requires tp_size == dp_size; "
            f"got tp_size={tp_size}, token-count entries={len(counts)}."
        )
    if get_parallel().tp_rank != get_parallel().attn_dp_rank:
        raise RuntimeError(
            "MLA-only-DP requires matching TP and attention-DP rank order."
        )
    return counts


def _all_to_all_flat(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
) -> None:
    torch.distributed.all_to_all_single(
        output.view(-1),
        input.contiguous().view(-1),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=get_parallel().tp_group.device_group,
    )


def _unpack_tp_head_shards_torch(
    recv_tp_rows: torch.Tensor, out: torch.Tensor
) -> None:
    """Convert [source_tp, rows, local_heads, dim] to row-major full heads."""
    out.copy_(recv_tp_rows.permute(1, 0, 2, 3).reshape_as(out))


def _pack_dp_rows_to_tp_shards_torch(
    local_full_heads: torch.Tensor, send_tp_shards: torch.Tensor
) -> None:
    """Convert row-major full heads to [destination_tp, rows, local_heads, dim]."""
    tp_size, local_rows, num_local_heads, head_dim = send_tp_shards.shape
    send_tp_shards.copy_(
        local_full_heads.view(local_rows, tp_size, num_local_heads, head_dim).permute(
            1, 0, 2, 3
        )
    )


def _pack_two_tp_head_shards_torch(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    out: torch.Tensor,
) -> None:
    out.copy_(torch.cat([q_nope, q_pe], dim=-1))


def _unpack_two_tp_head_shards_torch(
    recv_tp_rows: torch.Tensor,
    out_nope: torch.Tensor,
    out_pe: torch.Tensor,
    dim_nope: int,
) -> None:
    fused_rows = recv_tp_rows.permute(1, 0, 2, 3).reshape(
        out_nope.shape[0], out_nope.shape[1], dim_nope + out_pe.shape[2]
    )
    out_nope.copy_(fused_rows[..., :dim_nope])
    out_pe.copy_(fused_rows[..., dim_nope:])


def tp_head_shards_to_dp_rows(
    tp_head_shard: torch.Tensor,
    local_rows_like: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    """Exchange TP head shards for this attention-DP rank's token rows."""
    tp_size = get_parallel().tp_size
    if tp_size == 1:
        scattered = tp_head_shard.new_empty(
            (local_rows_like.shape[0], *tp_head_shard.shape[1:])
        )
        dp_scatter(
            scattered.view(scattered.shape[0], -1),
            tp_head_shard.contiguous().view(tp_head_shard.shape[0], -1),
            forward_batch,
        )
        return scattered

    global_tokens, local_heads, head_dim = tp_head_shard.shape
    local_rows = local_rows_like.shape[0]
    token_counts = _get_transfer_token_counts(forward_batch, tp_size)
    if sum(token_counts) != global_tokens:
        raise RuntimeError(
            "MLA-only-DP TP-to-DP input rows do not match the synchronized token "
            f"counts: rows={global_tokens}, counts={token_counts}."
        )
    if token_counts[get_parallel().attn_dp_rank] != local_rows:
        raise RuntimeError(
            "MLA-only-DP TP-to-DP local rows do not match this DP rank: "
            f"rows={local_rows}, counts={token_counts}."
        )

    shard_numel = local_heads * head_dim
    recv_tp_rows = tp_head_shard.new_empty(
        (tp_size, local_rows, local_heads, head_dim)
    )
    _all_to_all_flat(
        recv_tp_rows,
        tp_head_shard,
        [local_rows * shard_numel] * tp_size,
        [count * shard_numel for count in token_counts],
    )

    out = tp_head_shard.new_empty((local_rows, tp_size * local_heads, head_dim))
    if _use_transfer_kernel(recv_tp_rows):
        _tp_heads_to_dp_rows_kernel(recv_tp_rows, out, 0, local_rows)
    else:
        _warn_kernel_fallback_once()
        _unpack_tp_head_shards_torch(recv_tp_rows, out)
    return out


def tp_two_head_shards_to_dp_rows(
    q_nope_shard: torch.Tensor,
    q_pe_shard: torch.Tensor,
    local_rows_like: torch.Tensor,
    forward_batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange q_nope and q_pe TP head shards with one combined all-to-all."""
    tp_size = get_parallel().tp_size
    if tp_size == 1:
        return (
            tp_head_shards_to_dp_rows(q_nope_shard, local_rows_like, forward_batch),
            tp_head_shards_to_dp_rows(q_pe_shard, local_rows_like, forward_batch),
        )

    global_tokens, local_heads, dim_nope = q_nope_shard.shape
    if q_pe_shard.shape[0] != global_tokens or q_pe_shard.shape[1] != local_heads:
        raise RuntimeError(
            "MLA-only-DP fused TP-to-DP transfer requires matching token/head "
            f"dimensions: q_nope={tuple(q_nope_shard.shape)}, "
            f"q_pe={tuple(q_pe_shard.shape)}."
        )

    dim_pe = q_pe_shard.shape[2]
    local_rows = local_rows_like.shape[0]
    token_counts = _get_transfer_token_counts(forward_batch, tp_size)
    if sum(token_counts) != global_tokens:
        raise RuntimeError(
            "MLA-only-DP fused TP-to-DP input rows do not match the synchronized "
            f"token counts: rows={global_tokens}, counts={token_counts}."
        )
    if token_counts[get_parallel().attn_dp_rank] != local_rows:
        raise RuntimeError(
            "MLA-only-DP fused TP-to-DP local rows do not match this DP rank: "
            f"rows={local_rows}, counts={token_counts}."
        )

    if not _use_fused_q_transfer(token_counts, q_nope_shard, q_pe_shard):
        return (
            tp_head_shards_to_dp_rows(q_nope_shard, local_rows_like, forward_batch),
            tp_head_shards_to_dp_rows(q_pe_shard, local_rows_like, forward_batch),
        )

    fused_shard = q_nope_shard.new_empty(
        (global_tokens, local_heads, dim_nope + dim_pe)
    )
    _pack_two_tp_head_shards_kernel(q_nope_shard, q_pe_shard, fused_shard)

    shard_numel = local_heads * (dim_nope + dim_pe)
    recv_tp_rows = fused_shard.new_empty(
        (tp_size, local_rows, local_heads, dim_nope + dim_pe)
    )
    _all_to_all_flat(
        recv_tp_rows,
        fused_shard,
        [local_rows * shard_numel] * tp_size,
        [count * shard_numel for count in token_counts],
    )

    local_q_nope = q_nope_shard.new_empty(
        (local_rows, tp_size * local_heads, dim_nope)
    )
    local_q_pe = q_pe_shard.new_empty((local_rows, tp_size * local_heads, dim_pe))
    _unpack_two_tp_head_shards_kernel(
        recv_tp_rows, local_q_nope, local_q_pe, dim_nope
    )
    return local_q_nope, local_q_pe


def dp_rows_to_tp_head_shard(
    local_full_heads: torch.Tensor,
    forward_batch,
    num_local_heads: int,
) -> torch.Tensor:
    """Exchange DP-local full-head rows back to TP head shards."""
    tp_size = get_parallel().tp_size
    local_full_heads = local_full_heads.contiguous()
    if tp_size == 1:
        global_head_shard = local_full_heads.new_empty(
            (
                forward_batch.global_dp_buffer_len,
                num_local_heads,
                local_full_heads.shape[2],
            )
        )
        dp_gather_replicate(
            global_head_shard.view(forward_batch.global_dp_buffer_len, -1),
            local_full_heads.view(local_full_heads.shape[0], -1),
            forward_batch,
        )
        return global_head_shard

    local_rows, full_heads, head_dim = local_full_heads.shape
    if full_heads != tp_size * num_local_heads:
        raise RuntimeError(
            "MLA-only-DP attention output has an invalid head dimension: "
            f"heads={full_heads}, expected={tp_size * num_local_heads}."
        )
    token_counts = _get_transfer_token_counts(forward_batch, tp_size)
    if token_counts[get_parallel().attn_dp_rank] != local_rows:
        raise RuntimeError(
            "MLA-only-DP DP-to-TP local rows do not match this DP rank: "
            f"rows={local_rows}, counts={token_counts}."
        )
    global_tokens = sum(token_counts)
    if global_tokens != forward_batch.global_dp_buffer_len:
        raise RuntimeError(
            "MLA-only-DP DP-to-TP output rows do not match the synchronized "
            f"buffer: rows={global_tokens}, "
            f"buffer={forward_batch.global_dp_buffer_len}."
        )

    send_tp_shards = local_full_heads.new_empty(
        (tp_size, local_rows, num_local_heads, head_dim)
    )
    if _use_transfer_kernel(local_full_heads):
        _pack_dp_rows_to_tp_shards_kernel(local_full_heads, send_tp_shards)
    else:
        _warn_kernel_fallback_once()
        _pack_dp_rows_to_tp_shards_torch(local_full_heads, send_tp_shards)

    global_head_shard = local_full_heads.new_empty(
        (global_tokens, num_local_heads, head_dim)
    )
    shard_numel = num_local_heads * head_dim
    _all_to_all_flat(
        global_head_shard,
        send_tp_shards,
        [count * shard_numel for count in token_counts],
        [local_rows * shard_numel] * tp_size,
    )
    return global_head_shard
