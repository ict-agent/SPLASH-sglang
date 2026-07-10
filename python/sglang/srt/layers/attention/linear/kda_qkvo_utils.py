"""KDA-TP collective wrappers for the qkv_proj / o_proj head-shard path.

Forward sequence on each rank in the KDA qkvo-tp subgroup:

    (A) AllGather(hidden, dim=0)                    ── n_local → n_global
    qkv_proj  (ColumnParallel by head, weight 1/tp) ── (n_global, qkv_full/tp)
    (B) AllToAll head→token                         ── (n_local,  qkv_full)
    conv1d / fused_recurrent_kda / o_norm           ── per-DP, no collective
    (C) AllToAll token→head                         ── (n_global, head_full/tp)
    o_proj    (RowParallel by head, weight 1/tp)    ── (n_global, h) partial
    (D) ReduceScatter(dim=0)                        ── (n_local,  h)

This module provides both the plain NCCL implementation of (A)–(D) and a
``*_sym`` symm-mem fast-path dispatcher for each step. The dispatchers try
the ``triton_communicator`` multimem free function first; when the
per-process state is unavailable or the call returns ``None`` (comm buffer
too small / dtype mismatch / etc.), they fall back to the NCCL version in
this same module. Capacity / dtype / shape eligibility lives inside the
free functions themselves (``fits_comm_buffer`` + head-of-function
asserts), so callers do NOT need a ``should_use_*`` predicate — a ``None``
return is the sole fast-path miss signal.
"""

from __future__ import annotations

import torch

from sglang.srt.distributed.device_communicators.triton_communicator import (
    all_gather,
    alltoall_head_to_token,
    alltoall_token_to_head,
    reduce_scatter,
)
from sglang.srt.distributed.parallel_state import (
    get_symmetric_group,
)
from sglang.srt.layers.dp_attention import (
    get_global_dp_buffer,
    get_local_dp_buffer,
    get_triton_multimem_state,
)


def get_kda_qkvo_tp_size() -> int:
    return get_symmetric_group().world_size


def get_kda_qkvo_tp_local_rank() -> int:
    return get_symmetric_group().rank_in_group


def assert_kda_qkvo_tp_padding(forward_batch) -> None:
    """Validate the runtime invariant the kda-qkv-proj-tp-shard path relies on.

    Requires ``dp_padding_mode == MAX_LEN`` so every rank carries the same
    padded n_local. AllToAll requires equal send/recv counts across ranks;
    ``dp_gather_replicate`` would otherwise pick the SUM_LEN (all_reduce)
    branch under unbalanced batches and break that assumption downstream.

    Cheap CPU-only check -- we read the padding mode enum (Python-side)
    rather than ``.item()``-ing any GPU tensor, so this is safe to call inside
    the hot path / cuda graph.
    """
    assert forward_batch.dp_padding_mode.is_max_len(), (
        "kda qkv-proj tp shard requires dp_padding_mode == MAX_LEN so that "
        "every rank carries the same padded n_local. Got "
        f"{forward_batch.dp_padding_mode!r}."
    )

def kda_qkvo_tp_all_gather_hidden(local_hidden, forward_batch) -> torch.Tensor:
    """(A) Replicated gather of hidden_states along the KDA subgroup.

    Input:  ``(n_local, h)`` per rank.
    Output: ``(kda_qkvo_tp_size * n_local, h)`` on every rank in the subgroup,
            rank-major concatenation under MAX_LEN — rows
            ``[r*n_local : (r+1)*n_local]`` belong to subgroup-local rank ``r``.
    """
    tp_size = get_kda_qkvo_tp_size()
    if tp_size <= 1:
        return local_hidden
    local_hidden = local_hidden.contiguous()
    n_local, h = local_hidden.shape
    out = get_global_dp_buffer()[: n_local * tp_size,]
    get_symmetric_group().all_gather_into_tensor(out, local_hidden)
    return out


def kda_qkvo_tp_alltoall_head_to_token(
    qkv_global_shard: torch.Tensor,
    *,
    num_qkv_groups: int = 3,
) -> torch.Tensor:
    """(B) AllToAll (qkv) from head-shard layout to per-rank full-head layout.

    Input on rank ``r``:
        ``(n_global, 3 * F_local)`` where ``F_local = head_dim * num_heads / tp``.
        ``QKVParallelLinear`` emits ``[q_shard, k_shard, v_shard]`` along the
        last dim (each shard is the rank-r slice across heads of q, k, v
        respectively).

    Output on rank ``r``:
        ``(n_local, 3 * head_dim * num_heads)`` with feature layout
        ``[q_full, k_full, v_full]`` — q across ALL heads first, then k, then
        v. This matches what ``causal_conv1d_update`` / the KDA backend
        expects (``qkv.split([q_dim, k_dim, v_dim], dim=-1)`` then
        ``unflatten(-1, (-1, head_dim))`` works only if heads were
        concatenated in the q-first order).

    Implementation: split q / k / v as separate (n_global, F_local) tensors
    BEFORE the AllToAll, stack them on a new "group" axis, then do ONE
    all_to_all_single. After the collective, axis order is
    ``(src_rank, n_local, group, F_local)``; we move ``group`` to the front
    (so q from all ranks comes first), then ``src_rank`` so heads concat by
    rank within each group, and finally flatten.
    """
    assert qkv_global_shard.dim() == 2, (
        f"kda_qkvo_tp_alltoall_head_to_token expects (n_global, 3*F_local), got "
        f"{qkv_global_shard.shape}"
    )
    tp_size = get_kda_qkvo_tp_size()
    n_global, feat = qkv_global_shard.shape
    assert n_global % tp_size == 0, (
        f"n_global={n_global} must be divisible by tp_size={tp_size}; this "
        f"holds only when dp_padding_mode == MAX_LEN."
    )
    assert feat % num_qkv_groups == 0, (
        f"feat={feat} must be divisible by num_qkv_groups={num_qkv_groups}"
    )
    n_local = n_global // tp_size
    f_local = feat // num_qkv_groups  # head_dim * num_heads / tp

    # (n_global, 3, F_local) → (3, n_global, F_local) — keep groups contiguous
    # per token; this is what we want delivered separately on the recv side.
    send = (
        qkv_global_shard.view(tp_size, n_local, num_qkv_groups, f_local)
        .transpose(1, 2)
        .contiguous()
    )
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(
        recv, send, group=get_symmetric_group().device_group
    )
    # recv[src_rank, group, t, k] = group-g shard of token t produced by src.
    # Desired output per token: [q from rank 0 .. rank T-1, k from 0 .. T-1,
    # v from 0 .. T-1]. So group becomes the slowest axis; within each group,
    # head shards concat by src_rank. Reorder:
    #   (src, group, t, F_local) → (t, group, src, F_local)
    out = recv.permute(2, 1, 0, 3).contiguous()
    return out.view(n_local, num_qkv_groups * tp_size * f_local)


def kda_qkvo_tp_alltoall_token_to_head(
    core_attn_full: torch.Tensor,
) -> torch.Tensor:
    assert core_attn_full.dim() == 2, (
        f"kda_qkvo_tp_c_alltoall_token_to_head expects (n_local, head_full), got "
        f"{core_attn_full.shape}"
    )
    tp_size = get_kda_qkvo_tp_size()
    n_local, head_full = core_attn_full.shape
    assert head_full % tp_size == 0, (
        f"head_full={head_full} must be divisible by tp_size={tp_size}"
    )
    f_local = head_full // tp_size

    # Split feature dim into (tp_size dst-rank, F_local) and move dst-rank to
    # axis 0 → all_to_all_single sends one (n_local, F_local) slab to each peer.
    send = core_attn_full.view(n_local, tp_size, f_local).transpose(0, 1).contiguous()
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(
        recv, send, group=get_symmetric_group().device_group
    )
    # recv[src, t, k] = head shard of token t produced by rank src. Flatten src
    # × tokens → (tp_size * n_local, F_local) = (n_global, F_local), rank-major.
    return recv.view(tp_size * n_local, f_local)


def kda_qkvo_tp_reduce_scatter_hidden(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    hidden_states, global_hidden_states = (
        get_local_dp_buffer(),
        hidden_states,
    )
    get_symmetric_group().reduce_scatter_tensor(hidden_states, global_hidden_states)
    return hidden_states


def kda_qkvo_tp_all_gather_hidden_sym(
    local_hidden: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    """Try symm-mem AllGather first; fall back to NCCL on a miss."""
    state = get_triton_multimem_state()
    if state is not None:
        n_local = local_hidden.shape[0]
        out = all_gather(
            state,
            local_hidden,
            tp_num_tokens=n_local * state.world_size,
            safe=False,
        )
        if out is not None:
            return out
    return kda_qkvo_tp_all_gather_hidden(local_hidden, forward_batch)


def kda_qkvo_tp_alltoall_head_to_token_sym(
    qkv_global_shard: torch.Tensor,
    *,
    num_qkv_groups: int = 3,
) -> torch.Tensor:
    """Try symm-mem AllToAll head->token first; fall back to NCCL on a miss."""
    state = get_triton_multimem_state()
    if state is not None:
        out = alltoall_head_to_token(
            state, qkv_global_shard, num_qkv_groups=num_qkv_groups, safe=False
        )
        if out is not None:
            return out
    return kda_qkvo_tp_alltoall_head_to_token(
        qkv_global_shard, num_qkv_groups=num_qkv_groups
    )


def kda_qkvo_tp_alltoall_token_to_head_sym(
    core_attn_full: torch.Tensor,
) -> torch.Tensor:
    """Try symm-mem token->head first; fall back to NCCL if it cannot be staged.

    Uses direct compact AllToAll for the symm-mem path.
    """
    state = get_triton_multimem_state()
    if state is not None:
        out = alltoall_token_to_head(state, core_attn_full, safe=False)
        if out is not None:
            return out
    return kda_qkvo_tp_alltoall_token_to_head(core_attn_full)


def kda_qkvo_tp_reduce_scatter_hidden_sym(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Try symm-mem ReduceScatter first; fall back to NCCL on a miss."""
    state = get_triton_multimem_state()
    if state is not None:
        n_global = hidden_states.shape[0]
        out = reduce_scatter(state, hidden_states, tp_num_tokens=n_global)
        if out is not None:
            return out
    return kda_qkvo_tp_reduce_scatter_hidden(hidden_states)
