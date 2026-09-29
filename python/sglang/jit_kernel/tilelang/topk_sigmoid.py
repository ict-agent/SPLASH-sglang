from typing import Tuple

import tilelang
import torch
from tilelang import language as T


@T.macro
def warp_reduce_max_with_idx(score_var: T.Ref, idx_var: T.Ref):
    """Argmax across a 32-lane warp via 5-stage XOR butterfly."""
    for i in T.unroll(5):
        other_score = T.shfl_xor(score_var, 1 << i)
        other_idx = T.shfl_xor(idx_var, 1 << i)
        if (other_score > score_var) or (other_score == score_var and other_idx < idx_var):
            score_var = other_score
            idx_var = other_idx


@T.macro
def scoring(
    scores: T.Tensor,
    bias: T.SharedBuffer,
    origin_score_local: T.LocalBuffer,
    scores_local: T.LocalBuffer,
    token_idx: int,
    local_idx: int,
    expert_idx: int,
):
    """Sigmoid."""
    sig = 1.0 / (1.0 + T.__exp(-T.float32(scores[token_idx, expert_idx])))
    origin_score_local[local_idx] = sig
    scores_local[local_idx] = sig + bias[expert_idx]


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_topk_sigmoid_kernel(
    num_experts: int,
    num_topk: int,
    num_fused_shared_experts: int,
    renormalize: bool,
    apply_routed_scaling_factor: bool,
    score_dtype: T.dtype,
):
    warp_size = 32
    num_threads = 128
    num_warps = num_threads // warp_size

    vec_size = next(v for v in (4, 2, 1) if num_experts % (warp_size * v) <= warp_size)
    num_elems_per_warp_load = warp_size * vec_size

    num_vec_load_iters = num_experts // num_elems_per_warp_load
    num_vec_load_experts = num_vec_load_iters * num_elems_per_warp_load
    num_vec_load_elems = num_vec_load_iters * vec_size

    num_tail_load_experts = num_experts - num_vec_load_experts
    has_tail_experts = num_tail_load_experts > 0

    num_elems_per_thread = num_vec_load_elems + (1 if has_tail_experts else 0)
    num_aligned_experts = ((num_experts + warp_size - 1) // warp_size) * warp_size
    num_routed_topk = num_topk - num_fused_shared_experts

    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def topk_sigmoid(
        scores: T.Tensor[(num_tokens, num_experts), score_dtype],
        bias: T.Tensor[(num_experts,), T.float32],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        topk_ids: T.Tensor[(num_tokens, num_topk), T.int32],
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(T.ceildiv(num_tokens, num_warps), threads=num_threads) as pid:
            thread_idx = T.get_thread_binding()
            warp_idx = thread_idx // warp_size
            lane_idx = thread_idx % warp_size
            token_idx = pid * num_warps + warp_idx

            bias_shared = T.alloc_shared((num_aligned_experts,), T.float32)
            for e in T.serial(0, num_experts, num_threads):
                if e + thread_idx < num_experts:
                    bias_shared[e + thread_idx] = bias[e + thread_idx]
            T.sync_threads()

            topk_ids_shared = T.alloc_shared((num_warps, num_topk), T.int32)
            topk_weights_shared = T.alloc_shared((num_warps, num_topk), T.float32)

            if token_idx < num_tokens:
                scores_local = T.alloc_local((num_elems_per_thread,), T.float32)
                origin_score_local = T.alloc_local((num_elems_per_thread,), T.float32)

                max_score_var = T.alloc_var(dtype=T.float32)
                max_idx_var = T.alloc_var(dtype=T.int32)
                win_lane_idx_var = T.alloc_var(dtype=T.int32)
                win_local_idx_var = T.alloc_var(dtype=T.int32)
                origin_var = T.alloc_var(dtype=T.float32)
                sum_var = T.alloc_var(dtype=T.float32)

                for i in T.unroll(num_vec_load_iters):
                    for j in T.vectorized(vec_size):
                        scoring(
                            scores,
                            bias_shared,
                            origin_score_local,
                            scores_local,
                            token_idx,
                            i * vec_size + j,
                            i * num_elems_per_warp_load + lane_idx * vec_size + j,
                        )

                if has_tail_experts:
                    tail_expert_idx = num_vec_load_experts + lane_idx
                    if tail_expert_idx < num_experts:
                        scoring(
                            scores,
                            bias_shared,
                            origin_score_local,
                            scores_local,
                            token_idx,
                            num_vec_load_elems,
                            tail_expert_idx,
                        )
                    else:
                        origin_score_local[num_vec_load_elems] = 0.0
                        scores_local[num_vec_load_elems] = -T.infinity(T.float32)

                sum_var = 0.0
                for k in T.unroll(num_routed_topk):
                    max_score_var = -T.infinity(T.float32)
                    max_idx_var = T.int32(0)

                    for v in T.unroll(num_vec_load_elems):
                        if scores_local[v] > max_score_var:
                            max_score_var = scores_local[v]
                            max_idx_var = (
                                (v // vec_size) * num_elems_per_warp_load
                                + lane_idx * vec_size
                                + (v % vec_size)
                            )

                    if has_tail_experts:
                        if scores_local[num_vec_load_elems] > max_score_var:
                            max_score_var = scores_local[num_vec_load_elems]
                            max_idx_var = num_vec_load_experts + lane_idx

                    warp_reduce_max_with_idx(max_score_var, max_idx_var)

                    if max_idx_var < num_vec_load_experts:
                        win_lane_idx_var = (
                            max_idx_var % num_elems_per_warp_load
                        ) // vec_size
                        win_local_idx_var = (
                            max_idx_var // num_elems_per_warp_load
                        ) * vec_size + (max_idx_var % vec_size)
                    else:
                        win_lane_idx_var = max_idx_var - num_vec_load_experts
                        win_local_idx_var = num_vec_load_elems

                    origin_var = 0.0
                    if lane_idx == win_lane_idx_var:
                        for v in T.unroll(num_elems_per_thread):
                            if v == win_local_idx_var:
                                origin_var = origin_score_local[v]
                                scores_local[v] = -T.infinity(T.float32)

                    origin_var = T.shfl_sync(origin_var, win_lane_idx_var, mask=0xFFFFFFFF)

                    if lane_idx == 0:
                        topk_ids_shared[warp_idx, k] = max_idx_var
                        topk_weights_shared[warp_idx, k] = origin_var

                    sum_var = sum_var + origin_var

                if num_fused_shared_experts > 0:
                    if lane_idx == 0:
                        shared_weight = sum_var / routed_scaling_factor
                        for s in T.unroll(num_fused_shared_experts):
                            topk_ids_shared[warp_idx, num_routed_topk + s] = (
                                num_experts + s
                            )
                            topk_weights_shared[warp_idx, num_routed_topk + s] = (
                                shared_weight
                            )

                T.sync_warp()
                if renormalize:
                    if apply_routed_scaling_factor:
                        inv_sum = routed_scaling_factor / sum_var
                    else:
                        inv_sum = 1.0 / sum_var
                    if lane_idx < num_topk:
                        topk_weights_shared[warp_idx, lane_idx] = (
                            topk_weights_shared[warp_idx, lane_idx] * inv_sum
                        )

                if lane_idx < num_topk:
                    topk_ids[token_idx, lane_idx] = topk_ids_shared[warp_idx, lane_idx]
                    topk_weights[token_idx, lane_idx] = topk_weights_shared[
                        warp_idx, lane_idx
                    ]

    return topk_sigmoid


def topk_sigmoid(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor: bool = False,
    num_fused_shared_experts: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the top-k experts per token from scores.

    Args:
        scores: Gating scores with shape ``[num_tokens, num_experts]``.
        bias: Correction bias tensor with shape ``[num_experts]``.
        topk: Number of experts to select per token.
        renormalize: Whether to renormalize the topk weights.
        routed_scaling_factor: Scaling factor for the renormalized weights.
        apply_routed_scaling_factor: Apply scaling factor or not.
        num_fused_shared_experts: Number of trailing shared-expert.

    Returns:
        tuple:
        - topk_weights: Expert weights with shape ``[num_tokens, topk]`` and ``torch.float32``.
        - topk_idx: Selected expert indices with shape ``[num_tokens, topk]`` and ``torch.int32``.
    """
    assert scores.dim() == 2 and scores.is_contiguous()
    assert bias.dim() == 1 and bias.is_contiguous() and bias.dtype == torch.float32
    assert scores.size(1) == bias.size(0)
    assert 0 <= num_fused_shared_experts < topk
    assert 0 < topk - num_fused_shared_experts <= 32

    num_tokens, num_experts = scores.shape
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=scores.device)
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=scores.device)
    if num_tokens == 0:
        return topk_weights, topk_ids

    kernel = get_topk_sigmoid_kernel(
        num_experts,
        topk,
        num_fused_shared_experts,
        renormalize,
        apply_routed_scaling_factor,
        score_dtype=T.dtype(scores.dtype),
    )
    kernel(scores, bias, topk_weights, topk_ids, routed_scaling_factor)

    return topk_weights, topk_ids
