from __future__ import annotations

import logging
import os

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_rank,
)

logger = logging.getLogger(__name__)

try:
    from mla_only_dp_kernels import (
        is_available as _transfer_kernel_is_available,
        pack_dp_rows_to_tp_shards as _raw_pack_dp_rows_to_tp_shards_kernel,
        pack_two_tp_head_shards as _raw_pack_two_tp_head_shards_kernel,
        tp_heads_to_dp_rows as _raw_tp_heads_to_dp_rows_kernel,
        unpack_two_tp_head_shards as _raw_unpack_two_tp_head_shards_kernel,
    )
except Exception:
    _transfer_kernel_is_available = None
    _raw_pack_dp_rows_to_tp_shards_kernel = None
    _raw_pack_two_tp_head_shards_kernel = None
    _raw_tp_heads_to_dp_rows_kernel = None
    _raw_unpack_two_tp_head_shards_kernel = None

if _transfer_kernel_is_available is not None:
    from sglang.srt.utils.custom_op import register_custom_op

    @register_custom_op(
        op_name="mla_only_dp_tp_heads_to_dp_rows",
        mutates_args=["out"],
    )
    def _tp_heads_to_dp_rows_kernel(
        gathered_tp_heads: torch.Tensor,
        out: torch.Tensor,
        local_start_pos: int,
        local_num_tokens: int,
    ) -> None:
        _raw_tp_heads_to_dp_rows_kernel(
            gathered_tp_heads, out, local_start_pos, local_num_tokens
        )

    @register_custom_op(
        op_name="mla_only_dp_pack_dp_rows_to_tp_shards",
        mutates_args=["out"],
    )
    def _pack_dp_rows_to_tp_shards_kernel(
        full_heads: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        _raw_pack_dp_rows_to_tp_shards_kernel(full_heads, out)

    @register_custom_op(
        op_name="mla_only_dp_pack_two_tp_head_shards",
        mutates_args=["out"],
    )
    def _pack_two_tp_head_shards_kernel(
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        _raw_pack_two_tp_head_shards_kernel(q_nope, q_pe, out)

    @register_custom_op(
        op_name="mla_only_dp_unpack_two_tp_head_shards",
        mutates_args=["out_nope", "out_pe"],
    )
    def _unpack_two_tp_head_shards_kernel(
        gathered: torch.Tensor,
        out_nope: torch.Tensor,
        out_pe: torch.Tensor,
        dim_nope: int,
    ) -> None:
        _raw_unpack_two_tp_head_shards_kernel(
            gathered, out_nope, out_pe, dim_nope
        )
else:
    _pack_dp_rows_to_tp_shards_kernel = None
    _pack_two_tp_head_shards_kernel = None
    _tp_heads_to_dp_rows_kernel = None
    _unpack_two_tp_head_shards_kernel = None

_warned_missing_kernel = False
_debug_logged_tp_to_dp = False
_debug_logged_dp_to_tp = False


def _all_gather_transfer_max_rows() -> int:
    return int(os.getenv("SGLANG_MLA_ONLY_DP_ALLGATHER_TRANSFER_MAX_ROWS", "0"))


def _max_int(values: list[int]) -> int:
    max_value = 0
    for value in values:
        if value > max_value:
            max_value = value
    return max_value


def _all_equal_int(values: list[int]) -> bool:
    if len(values) <= 1:
        return True
    first = values[0]
    for value in values:
        if value != first:
            return False
    return True


def _use_transfer_kernel(x: torch.Tensor) -> bool:
    if os.getenv("SGLANG_MLA_ONLY_DP_DISABLE_TRANSFER_KERNEL") == "1":
        return False
    if not x.is_cuda or not x.is_contiguous():
        return False
    return bool(
        _transfer_kernel_is_available is not None
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
    return _use_pair_transfer_kernel(q_nope, q_pe)


def _use_all_gather_transfer(token_counts: list[int], x: torch.Tensor) -> bool:
    max_rows = _max_int(token_counts)
    return (
        _all_gather_transfer_max_rows() > 0
        and max_rows <= _all_gather_transfer_max_rows()
        and x.is_cuda
        and x.is_contiguous()
    )


def _use_dp_to_tp_all_gather_transfer(
    token_counts: list[int], x: torch.Tensor
) -> bool:
    return _use_all_gather_transfer(token_counts, x) and _all_equal_int(token_counts)


def _warn_kernel_fallback_once() -> None:
    global _warned_missing_kernel
    if torch.compiler.is_compiling():
        return
    if _warned_missing_kernel:
        return
    _warned_missing_kernel = True
    logger.warning(
        "mla_only_dp_kernels is disabled or unavailable; using PyTorch packing "
        "around the MLA-only-DP all-to-all transfer."
    )


def _debug_transfer_begin(kind: str, **kwargs) -> bool:
    global _debug_logged_tp_to_dp, _debug_logged_dp_to_tp
    if torch.compiler.is_compiling():
        return False
    if os.getenv("SGLANG_MLA_ONLY_DP_DEBUG") != "1":
        return False
    if kind == "tp_to_dp":
        if _debug_logged_tp_to_dp:
            return False
    elif kind == "dp_to_tp":
        if _debug_logged_dp_to_tp:
            return False
    torch.cuda.synchronize()
    logger.warning(
        "mla-only-dp transfer %s begin: %s",
        kind,
        ", ".join(f"{key}={value}" for key, value in kwargs.items()),
    )
    return True


def _debug_transfer_end(kind: str, enabled: bool) -> None:
    global _debug_logged_tp_to_dp, _debug_logged_dp_to_tp
    if not enabled:
        return
    torch.cuda.synchronize()
    logger.warning("mla-only-dp transfer %s end", kind)
    if kind == "tp_to_dp":
        _debug_logged_tp_to_dp = True
    elif kind == "dp_to_tp":
        _debug_logged_dp_to_tp = True


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
    if get_tensor_model_parallel_rank() != get_attention_dp_rank():
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
    output_flat = output.view(-1)
    input_flat = input.contiguous().view(-1)
    if output_split_sizes == input_split_sizes and _all_equal_int(input_split_sizes):
        torch.distributed.all_to_all_single(
            output_flat,
            input_flat,
            group=get_tp_group().device_group,
        )
    else:
        torch.distributed.all_to_all_single(
            output_flat,
            input_flat,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=get_tp_group().device_group,
        )


def _tp_head_shards_to_dp_rows_all_gather(
    tp_head_shard: torch.Tensor,
    local_rows_like: torch.Tensor,
    forward_batch,
    token_counts: list[int],
) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    global_tokens, local_heads, head_dim = tp_head_shard.shape
    flat = tp_head_shard.contiguous().view(global_tokens, local_heads * head_dim)
    gathered = flat.new_empty((tp_size * global_tokens, flat.shape[1]))
    debug = _debug_transfer_begin(
        "tp_to_dp",
        mode=forward_batch.forward_mode,
        all_gather=True,
        input_shape=tuple(tp_head_shard.shape),
        output_rows=local_rows_like.shape[0],
        token_counts=token_counts,
        dtype=str(tp_head_shard.dtype),
    )
    get_tp_group().all_gather_into_tensor(gathered, flat)
    full_heads = (
        gathered.view(tp_size, global_tokens, local_heads, head_dim)
        .permute(1, 0, 2, 3)
        .reshape(global_tokens, tp_size * local_heads, head_dim)
        .contiguous()
    )
    out = tp_head_shard.new_empty(
        (local_rows_like.shape[0], tp_size * local_heads, head_dim)
    )
    dp_scatter(
        out.view(out.shape[0], tp_size * local_heads * head_dim),
        full_heads.view(global_tokens, -1),
        forward_batch,
    )
    _debug_transfer_end("tp_to_dp", debug)
    return out


def _tp_two_head_shards_to_dp_rows_all_gather(
    q_nope_shard: torch.Tensor,
    q_pe_shard: torch.Tensor,
    local_rows_like: torch.Tensor,
    forward_batch,
    token_counts: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    tp_size = get_tensor_model_parallel_world_size()
    global_tokens, local_heads, dim_nope = q_nope_shard.shape
    dim_pe = q_pe_shard.shape[2]
    fused_shard = q_nope_shard.new_empty(
        (global_tokens, local_heads, dim_nope + dim_pe)
    )
    if _use_pair_transfer_kernel(q_nope_shard, q_pe_shard):
        _pack_two_tp_head_shards_kernel(q_nope_shard, q_pe_shard, fused_shard)
    else:
        _pack_two_tp_head_shards_torch(q_nope_shard, q_pe_shard, fused_shard)

    local_fused = _tp_head_shards_to_dp_rows_all_gather(
        fused_shard, local_rows_like, forward_batch, token_counts
    )
    local_q_nope = q_nope_shard.new_empty(
        (local_fused.shape[0], tp_size * local_heads, dim_nope)
    )
    local_q_pe = q_pe_shard.new_empty(
        (local_fused.shape[0], tp_size * local_heads, dim_pe)
    )
    local_q_nope.copy_(local_fused[..., :dim_nope])
    local_q_pe.copy_(local_fused[..., dim_nope:])
    return local_q_nope, local_q_pe


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
    tp_size = get_tensor_model_parallel_world_size()
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
    if token_counts[get_attention_dp_rank()] != local_rows:
        raise RuntimeError(
            "MLA-only-DP TP-to-DP local rows do not match this DP rank: "
            f"rows={local_rows}, counts={token_counts}."
        )

    if _use_all_gather_transfer(token_counts, tp_head_shard):
        return _tp_head_shards_to_dp_rows_all_gather(
            tp_head_shard, local_rows_like, forward_batch, token_counts
        )

    shard_numel = local_heads * head_dim
    recv_tp_rows = tp_head_shard.new_empty(
        (tp_size, local_rows, local_heads, head_dim)
    )
    debug = _debug_transfer_begin(
        "tp_to_dp",
        mode=forward_batch.forward_mode,
        input_shape=tuple(tp_head_shard.shape),
        output_rows=local_rows,
        token_counts=token_counts,
        dtype=str(tp_head_shard.dtype),
    )
    _all_to_all_flat(
        recv_tp_rows,
        tp_head_shard,
        [local_rows * shard_numel] * tp_size,
        [count * shard_numel for count in token_counts],
    )

    out = tp_head_shard.new_empty(
        (local_rows, tp_size * local_heads, head_dim)
    )
    if _use_transfer_kernel(recv_tp_rows):
        _tp_heads_to_dp_rows_kernel(recv_tp_rows, out, 0, local_rows)
    else:
        _warn_kernel_fallback_once()
        out.copy_(recv_tp_rows.permute(1, 0, 2, 3).reshape_as(out))
    _debug_transfer_end("tp_to_dp", debug)
    return out


def tp_two_head_shards_to_dp_rows(
    q_nope_shard: torch.Tensor,
    q_pe_shard: torch.Tensor,
    local_rows_like: torch.Tensor,
    forward_batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange q_nope and q_pe TP head shards with one combined all-to-all."""
    tp_size = get_tensor_model_parallel_world_size()
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
    if token_counts[get_attention_dp_rank()] != local_rows:
        raise RuntimeError(
            "MLA-only-DP fused TP-to-DP local rows do not match this DP rank: "
            f"rows={local_rows}, counts={token_counts}."
        )

    if _use_all_gather_transfer(token_counts, q_nope_shard):
        return _tp_two_head_shards_to_dp_rows_all_gather(
            q_nope_shard,
            q_pe_shard,
            local_rows_like,
            forward_batch,
            token_counts,
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
    debug = _debug_transfer_begin(
        "tp_to_dp",
        mode=forward_batch.forward_mode,
        fused=True,
        q_nope_shape=tuple(q_nope_shard.shape),
        q_pe_shape=tuple(q_pe_shard.shape),
        output_rows=local_rows,
        token_counts=token_counts,
        dtype=str(q_nope_shard.dtype),
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
    _debug_transfer_end("tp_to_dp", debug)
    return local_q_nope, local_q_pe


def tp_fused_q_head_shard_to_dp_rows(
    q_shard: torch.Tensor,
    local_rows_like: torch.Tensor,
    dim_nope: int,
    forward_batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange an already-fused q tensor and split it into DP-local q parts."""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        local_q = tp_head_shards_to_dp_rows(q_shard, local_rows_like, forward_batch)
        return local_q[..., :dim_nope].contiguous(), local_q[..., dim_nope:].contiguous()

    if q_shard.dim() != 3:
        raise RuntimeError(
            "MLA-only-DP fused-q transfer requires q_shard with shape "
            "[tokens, local_heads, dim_nope + dim_pe]."
        )
    if not q_shard.is_contiguous():
        raise RuntimeError("MLA-only-DP fused-q transfer requires contiguous q_shard.")

    global_tokens, local_heads, total_dim = q_shard.shape
    if dim_nope <= 0 or dim_nope >= total_dim:
        raise RuntimeError(
            "MLA-only-DP fused-q transfer got invalid split: "
            f"dim_nope={dim_nope}, total_dim={total_dim}."
        )

    local_rows = local_rows_like.shape[0]
    token_counts = _get_transfer_token_counts(forward_batch, tp_size)
    if sum(token_counts) != global_tokens:
        raise RuntimeError(
            "MLA-only-DP fused-q transfer input rows do not match synchronized "
            f"token counts: rows={global_tokens}, counts={token_counts}."
        )
    if token_counts[get_attention_dp_rank()] != local_rows:
        raise RuntimeError(
            "MLA-only-DP fused-q transfer local rows do not match this DP rank: "
            f"rows={local_rows}, counts={token_counts}."
        )

    if _use_all_gather_transfer(token_counts, q_shard):
        local_q = _tp_head_shards_to_dp_rows_all_gather(
            q_shard, local_rows_like, forward_batch, token_counts
        )
        return local_q[..., :dim_nope].contiguous(), local_q[..., dim_nope:].contiguous()

    recv_tp_rows = q_shard.new_empty((tp_size, local_rows, local_heads, total_dim))
    shard_numel = local_heads * total_dim
    debug = _debug_transfer_begin(
        "tp_to_dp",
        mode=forward_batch.forward_mode,
        fused_q=True,
        input_shape=tuple(q_shard.shape),
        output_rows=local_rows,
        token_counts=token_counts,
        dtype=str(q_shard.dtype),
    )
    _all_to_all_flat(
        recv_tp_rows,
        q_shard,
        [local_rows * shard_numel] * tp_size,
        [count * shard_numel for count in token_counts],
    )

    dim_pe = total_dim - dim_nope
    local_q_nope = q_shard.new_empty((local_rows, tp_size * local_heads, dim_nope))
    local_q_pe = q_shard.new_empty((local_rows, tp_size * local_heads, dim_pe))
    if _use_pair_transfer_kernel(local_q_nope, local_q_pe):
        _unpack_two_tp_head_shards_kernel(
            recv_tp_rows, local_q_nope, local_q_pe, dim_nope
        )
    else:
        _warn_kernel_fallback_once()
        _unpack_two_tp_head_shards_torch(
            recv_tp_rows, local_q_nope, local_q_pe, dim_nope
        )
    _debug_transfer_end("tp_to_dp", debug)
    return local_q_nope, local_q_pe


def dp_packed_rows_to_tp_head_shard(
    send_tp_shards: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    """Exchange pre-packed [tp, local_rows, local_heads, dim] rows back to TP."""
    tp_size = get_tensor_model_parallel_world_size()
    if send_tp_shards.dim() != 4:
        raise RuntimeError(
            "MLA-only-DP packed DP-to-TP input must have shape "
            "[tp, local_rows, local_heads, dim]."
        )
    if send_tp_shards.shape[0] != tp_size:
        raise RuntimeError(
            "MLA-only-DP packed DP-to-TP input has an invalid TP dimension: "
            f"tp={send_tp_shards.shape[0]}, expected={tp_size}."
        )

    local_rows = send_tp_shards.shape[1]
    num_local_heads = send_tp_shards.shape[2]
    head_dim = send_tp_shards.shape[3]
    token_counts = _get_transfer_token_counts(forward_batch, tp_size)
    if token_counts[get_attention_dp_rank()] != local_rows:
        raise RuntimeError(
            "MLA-only-DP packed DP-to-TP local rows do not match this DP rank: "
            f"rows={local_rows}, counts={token_counts}."
        )
    global_tokens = sum(token_counts)
    if global_tokens != forward_batch.global_dp_buffer_len:
        raise RuntimeError(
            "MLA-only-DP packed DP-to-TP output rows do not match the synchronized "
            f"buffer: rows={global_tokens}, buffer={forward_batch.global_dp_buffer_len}."
        )

    global_head_shard = send_tp_shards.new_empty(
        (global_tokens, num_local_heads, head_dim)
    )
    shard_numel = num_local_heads * head_dim
    debug = _debug_transfer_begin(
        "dp_to_tp",
        mode=forward_batch.forward_mode,
        packed=True,
        input_shape=tuple(send_tp_shards.shape),
        output_rows=global_tokens,
        token_counts=token_counts,
        dtype=str(send_tp_shards.dtype),
    )
    _all_to_all_flat(
        global_head_shard,
        send_tp_shards,
        [count * shard_numel for count in token_counts],
        [local_rows * shard_numel] * tp_size,
    )
    _debug_transfer_end("dp_to_tp", debug)
    return global_head_shard


def dp_packed_head_major_to_tp_head_shard(
    send_tp_shards: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    """Exchange pre-packed [tp, local_heads, local_rows, dim] rows back to TP.

    This layout is produced by a single batched V-projection GEMM over
    ``tp_size * local_heads`` heads. The wire format is still destination-rank
    major; after the all-to-all, each received chunk is transposed back to the
    row-major [global_tokens, local_heads, dim] layout expected by the TP path.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if send_tp_shards.dim() != 4:
        raise RuntimeError(
            "MLA-only-DP head-major DP-to-TP input must have shape "
            "[tp, local_heads, local_rows, dim]."
        )
    if send_tp_shards.shape[0] != tp_size:
        raise RuntimeError(
            "MLA-only-DP head-major DP-to-TP input has an invalid TP dimension: "
            f"tp={send_tp_shards.shape[0]}, expected={tp_size}."
        )

    num_local_heads = send_tp_shards.shape[1]
    local_rows = send_tp_shards.shape[2]
    head_dim = send_tp_shards.shape[3]
    token_counts = _get_transfer_token_counts(forward_batch, tp_size)
    if token_counts[get_attention_dp_rank()] != local_rows:
        raise RuntimeError(
            "MLA-only-DP head-major DP-to-TP local rows do not match this DP rank: "
            f"rows={local_rows}, counts={token_counts}."
        )
    global_tokens = sum(token_counts)
    if global_tokens != forward_batch.global_dp_buffer_len:
        raise RuntimeError(
            "MLA-only-DP head-major DP-to-TP output rows do not match the "
            f"synchronized buffer: rows={global_tokens}, "
            f"buffer={forward_batch.global_dp_buffer_len}."
        )

    shard_numel = num_local_heads * head_dim
    debug = _debug_transfer_begin(
        "dp_to_tp",
        mode=forward_batch.forward_mode,
        packed=True,
        head_major=True,
        input_shape=tuple(send_tp_shards.shape),
        output_rows=global_tokens,
        token_counts=token_counts,
        dtype=str(send_tp_shards.dtype),
    )

    if _all_equal_int(token_counts):
        recv_tp_rows = send_tp_shards.new_empty(
            (tp_size, num_local_heads, local_rows, head_dim)
        )
        _all_to_all_flat(
            recv_tp_rows,
            send_tp_shards,
            [local_rows * shard_numel] * tp_size,
            [local_rows * shard_numel] * tp_size,
        )
        global_head_shard = (
            recv_tp_rows.permute(0, 2, 1, 3)
            .reshape(global_tokens, num_local_heads, head_dim)
            .contiguous()
        )
    else:
        recv_flat = send_tp_shards.new_empty((global_tokens * shard_numel,))
        _all_to_all_flat(
            recv_flat,
            send_tp_shards,
            [count * shard_numel for count in token_counts],
            [local_rows * shard_numel] * tp_size,
        )
        global_head_shard = send_tp_shards.new_empty(
            (global_tokens, num_local_heads, head_dim)
        )
        src_offset = 0
        dst_offset = 0
        for count in token_counts:
            chunk_numel = count * shard_numel
            if count:
                global_head_shard[dst_offset : dst_offset + count].copy_(
                    recv_flat[src_offset : src_offset + chunk_numel]
                    .view(num_local_heads, count, head_dim)
                    .permute(1, 0, 2)
                )
            src_offset += chunk_numel
            dst_offset += count

    _debug_transfer_end("dp_to_tp", debug)
    return global_head_shard


def dp_rows_to_tp_head_shard(
    local_full_heads: torch.Tensor,
    forward_batch,
    num_local_heads: int,
) -> torch.Tensor:
    """Exchange DP-local full-head rows back to TP head shards."""
    tp_size = get_tensor_model_parallel_world_size()
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
    if token_counts[get_attention_dp_rank()] != local_rows:
        raise RuntimeError(
            "MLA-only-DP DP-to-TP local rows do not match this DP rank: "
            f"rows={local_rows}, counts={token_counts}."
        )
    global_tokens = sum(token_counts)
    if global_tokens != forward_batch.global_dp_buffer_len:
        raise RuntimeError(
            "MLA-only-DP DP-to-TP output rows do not match the synchronized "
            f"buffer: rows={global_tokens}, buffer={forward_batch.global_dp_buffer_len}."
        )

    if _use_dp_to_tp_all_gather_transfer(token_counts, local_full_heads):
        if global_tokens == 0:
            return local_full_heads.new_empty((0, num_local_heads, head_dim))
        global_full_heads = local_full_heads.new_empty(
            (global_tokens, full_heads, head_dim)
        )
        debug = _debug_transfer_begin(
            "dp_to_tp",
            mode=forward_batch.forward_mode,
            all_gather=True,
            input_shape=tuple(local_full_heads.shape),
            output_rows=global_tokens,
            token_counts=token_counts,
            dtype=str(local_full_heads.dtype),
        )
        dp_gather_replicate(
            global_full_heads.view(global_tokens, -1),
            local_full_heads.view(local_rows, full_heads * head_dim),
            forward_batch,
        )
        _debug_transfer_end("dp_to_tp", debug)
        rank = get_tensor_model_parallel_rank()
        start = rank * num_local_heads
        return global_full_heads[:, start : start + num_local_heads, :].contiguous()

    send_tp_shards = local_full_heads.new_empty(
        (tp_size, local_rows, num_local_heads, head_dim)
    )
    if _use_transfer_kernel(local_full_heads):
        _pack_dp_rows_to_tp_shards_kernel(local_full_heads, send_tp_shards)
    else:
        _warn_kernel_fallback_once()
        send_tp_shards.copy_(
            local_full_heads.view(local_rows, tp_size, num_local_heads, head_dim)
            .permute(1, 0, 2, 3)
        )

    return dp_packed_rows_to_tp_head_shard(send_tp_shards, forward_batch)
