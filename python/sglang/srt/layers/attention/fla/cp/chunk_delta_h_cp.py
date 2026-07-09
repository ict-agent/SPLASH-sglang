# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported (forward-only subset) from flash-linear-attention:
#   fla/ops/cp/chunk_delta_h.py
#
# Changes vs. upstream:
#   * Backward kernels and intra-card mode are dropped (inference-only).
#   * The CP branch of ``merge_fwd_bwd_kernel`` gains an optional ``h0`` seed so
#     SGLang can inject prefix-cache state as the start of the merge chain
#     (upstream only seeds ``h0`` in the intra-card branch).

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.cp.comm import all_gather_into_tensor
from sglang.srt.layers.attention.fla.op import exp2
from sglang.srt.layers.attention.fla.utils import (
    autotune_cache_kwargs,
    check_shared_mem,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.fla.cp.context import FLACPContext


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_BG': lambda args: args['bg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_kernel_merged(
    k,
    v,
    w,
    g,
    gk,
    bg,
    u,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MULTI_SEQS: tl.constexpr,
):
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    if MULTI_SEQS:
        i_n = tl.program_id(2)
        # Offset hm for this subseq: hm[i_n, h, k, v+k]
        hm += i_n * HV * K * (K + V) + i_h * K * (K + V)
    else:
        i_n = 0
        hm += i_h * K * (K + V)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # Determine if this block handles h (V part) or m (K part)
    # i_col is in range [0, cdiv(V + K, BLOCK_SIZE))
    # Columns [0, V) are for h, columns [V, V+K) are for m
    is_h_part = i_col * BLOCK_SIZE < V
    # For DPLR (USE_BG), w and bg share the same head dim H as k/ag.
    # For GDN/KDA, w has head dim HV (same as v).
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
        bg += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    stride_k = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_h_part:
        # ====== Stage 1: Compute h (K x V) ======
        v += ((bos * HV + i_h) * V).to(tl.int64)
        if USE_BG:
            # DPLR keeps u and v as separate tensors; both need the per-head offset.
            # For GDN/KDA, u is aliased to v at the Python wrapper level.
            u += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        # Initialize h accumulators
        b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        # Main recurrence for h
        for i_t in range(NT):
            # Compute decayed v
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h4.to(b_w.dtype))

            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            if USE_BG:
                # DPLR mode: v2 = w @ h + u, h += kg^T @ v + bg^T @ v2
                b_v_orig = tl.load(p_v, boundary_check=(0, 1))
                p_u = tl.make_block_ptr(u, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
                b_v = b_v_decay + tl.load(p_u, boundary_check=(0, 1))
            else:
                # GDN/KDA mode: v_new = v - w @ h
                b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v_decay

            last_idx = min((i_t + 1) * BT, T) - 1

            # Apply g decay
            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            # Apply gk decay
            if USE_GK:
                o_k1 = tl.arange(0, 64)
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                b_h1 *= exp2(b_gk_last1)[:, None]
                if K > 64:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                    b_h2 *= exp2(b_gk_last2)[:, None]
                if K > 128:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                    b_h3 *= exp2(b_gk_last3)[:, None]
                if K > 192:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                    b_h4 *= exp2(b_gk_last4)[:, None]
            b_v = b_v.to(k.dtype.element_ty)

            # Update h
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_BG:
                # DPLR mode: h += kg^T @ v + bg^T @ v2
                p_bg = tl.make_block_ptr(bg, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
                b_bg = tl.load(p_bg, boundary_check=(0, 1))
                b_h1 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 64:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    p_bg = tl.make_block_ptr(bg, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                    b_bg = tl.load(p_bg, boundary_check=(0, 1))
                    b_h2 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 128:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    p_bg = tl.make_block_ptr(bg, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                    b_bg = tl.load(p_bg, boundary_check=(0, 1))
                    b_h3 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 192:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    p_bg = tl.make_block_ptr(bg, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                    b_bg = tl.load(p_bg, boundary_check=(0, 1))
                    b_h4 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
            else:
                # GDN/KDA mode: h += k^T @ v_new
                b_h1 += tl.dot(b_k, b_v)
                if K > 64:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h2 += tl.dot(b_k, b_v)
                if K > 128:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h3 += tl.dot(b_k, b_v)
                if K > 192:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h4 += tl.dot(b_k, b_v)

        # Store h results
        stride_hm_kv = K + V
        p_h1 = tl.make_block_ptr(hm, (K, V), (stride_hm_kv, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(hm, (K, V), (stride_hm_kv, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(hm, (K, V), (stride_hm_kv, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(hm, (K, V), (stride_hm_kv, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
    else:
        # ====== Stage 2: Compute m (K x K) ======
        # i_col is for m part, map to K dimension
        # m starts at column V, so offset = i_col * BLOCK_SIZE - V
        # Use tl.cdiv to correctly compute the number of blocks for V dimension
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        # Following stage2 kernel design:
        # - BK1 is the full K dimension (next_power_of_2(K))
        # - BLOCK_SIZE is the column block size (like BK2=32 in stage2)
        # Each block computes a (BK1, BLOCK_SIZE) sub-matrix of m
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize as identity matrix: M_0 = I
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i_t in range(NT):
            # Load k and w with full BK1 rows
            if USE_BG:
                # DPLR mode: use bg for transition matrix
                # bg was already offset at the beginning of the kernel
                p_k = tl.make_block_ptr(bg, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            else:
                p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            # Compute m update
            if USE_BG:
                # DPLR mode: M = (diag + bg^T @ w) @ M
                # bg was already offset at the beginning of the kernel
                b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
                b_m_i = b_diag + b_kw
            else:
                # GDN/KDA mode: M = (diag - k^T @ w) @ M
                b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
                b_m_i = b_diag - b_kw
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

        # Store m result
        stride_hm_kv = K + V
        p_m = tl.make_block_ptr(hm + V, (K, K), (stride_hm_kv, 1), (0, i_k_col * BLOCK_SIZE), (BK1, BLOCK_SIZE), (1, 0))
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_H0': lambda args: args['h0'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['HV', 'K', 'V'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['pre_or_post_num_ranks', 'rank'])
def merge_fwd_kernel(
    h,                      # [HV, K, V] — this rank's incoming initial state (written)
    ag_hm,                  # [world_size, HV, K, K+V] — all-gathered [S_ext | M]
    pre_or_post_num_ranks,  # number of contributing previous ranks
    rank,                   # this rank's index within the CP group
    h0,                     # None or [HV, K, V] fp32 — prefix seed for the merge chain
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,
    HAS_H0: tl.constexpr,
    STATE_V_FIRST: tl.constexpr = False,
):
    """CP merge: reconstruct this rank's incoming state by chaining
    ``h <- M_j @ h + S_ext_j`` over the contributing previous ranks, optionally
    seeded from a prefix state ``h0`` instead of zero.
    """
    i_v = tl.program_id(0)
    i_h = tl.program_id(1)
    num_ranks = pre_or_post_num_ranks.to(tl.int32)
    h += i_h * K * V
    ag_hm += i_h * K * (K + V)
    stride = HV * K * (K + V)
    if HAS_H0:
        h0 += i_h * K * V
        if STATE_V_FIRST:
            p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        else:
            p_h0 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    else:
        if STATE_V_FIRST:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for idx in range(num_ranks):
        if FORWARD:
            cur_rank = rank - num_ranks + idx
        else:
            cur_rank = rank + num_ranks - idx
        p_ag_h = tl.make_block_ptr(ag_hm + cur_rank * stride, (K, V), (K + V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        b_ag_h = tl.load(p_ag_h, boundary_check=(0, 1))
        p_ag_m = tl.make_block_ptr(ag_hm + cur_rank * stride + V, (K, K), (K + V, 1), (0, 0), (BK, BK), (1, 0))
        b_ag_m = tl.load(p_ag_m, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_ag_m).to(tl.float32)) + tl.trans(b_ag_h).to(tl.float32)
        else:
            b_h = tl.dot(b_ag_m.to(tl.float32), b_h.to(tl.float32)) + b_ag_h.to(tl.float32)
    if STATE_V_FIRST:
        p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
    else:
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h_pre_process(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    context: "FLACPContext" = None,
    h0: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute this rank's cross-rank incoming state via all-gather + merge.

    Every rank computes its local ``[S_ext | M]`` (``hm``) over its **last**
    local segment (the only one that can continue onto the next rank),
    all-gathers ``hm`` across the group, then — if this rank continues a
    sequence from previous ranks (``not is_first_rank``) — chains the
    contributions of ``pre_num_ranks`` previous ranks (optionally seeded by the
    sequence's prefix state ``h0``) into a single ``[HV, V, K]`` (V-first) /
    ``[HV, K, V]`` state.

    Returns:
        The merged ``[HV, V, K]`` (or ``[HV, K, V]``) fp32 state for this rank's
        first (continuation) local segment, or ``None`` when this rank owns the
        sequence's first token (``is_first_rank``) — in that case the caller
        seeds the segment from prefix/zero itself. Returns ``None`` when CP is
        disabled.
    """
    if context is None or context.group is None:
        return None

    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    BK = triton.next_power_of_2(K)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    hm = k.new_zeros(HV, K, (V + K), dtype=torch.float32)

    # Stage 1+2: local [S_ext | M] over the last local segment. Only ranks that
    # hand state to a following rank need to produce it.
    if not context.is_last_rank:
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
        pre_process_fwd_kernel_merged[grid](
            k=k,
            v=u if v is None else v,
            w=w,
            g=g,
            gk=gk,
            bg=bg,
            u=u,
            hm=hm,
            cu_seqlens=cu_seqlens[-2:],
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            BLOCK_SIZE=BLOCK_SIZE,
            MULTI_SEQS=False,
        )

    # Collective: every rank must participate even when it produced a zero hm.
    ag_hm, _ = all_gather_into_tensor(hm, group=context.group)

    if context.is_first_rank:
        # This rank owns the sequence's first token; no cross-rank predecessors.
        return None

    if state_v_first:
        merged = k.new_zeros(1, HV, V, K, dtype=torch.float32)
    else:
        merged = k.new_zeros(1, HV, K, V, dtype=torch.float32)

    rank = dist.get_rank(group=context.group)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), HV)

    merge_fwd_kernel[grid](
        h=merged[0],
        ag_hm=ag_hm,
        pre_or_post_num_ranks=context.pre_num_ranks,
        rank=rank,
        h0=h0,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        FORWARD=True,
        STATE_V_FIRST=state_v_first,
    )
    return merged[0]
