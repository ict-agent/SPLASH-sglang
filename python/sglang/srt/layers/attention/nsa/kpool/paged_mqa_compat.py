"""Opt-in Hopper fallback for GLM kpool=4 / real page=64 (pooled page=16).

It preserves the packed FP8 K + FP32 scales cache and the existing top-k path.
"""

from functools import lru_cache
import os

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def _hopper_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] == 9


def use_hopper_kpool_fallback(block_kv: int, device: torch.device) -> bool:
    # Use the fallback only for Hopper with pooled page size 16.
    return (
        os.environ.get("SGLANG_NSA_KPOOL_HOPPER_FALLBACK") == "1"
        and block_kv == 16
        and device.type == "cuda"
        and _hopper_device(device.index if device.index is not None else torch.cuda.current_device())
    )


def get_kpool_paged_mqa_logits_metadata(context_lens, block_kv, num_sms):
    if use_hopper_kpool_fallback(block_kv, context_lens.device):
        # This kernel has a fixed launch grid; no persistent DeepGEMM schedule.
        return None
    import deep_gemm

    return deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)


def kpool_fp8_paged_mqa_logits(
    q, kv_cache, weights, context_lens, block_tables, schedule_metadata,
    max_seq_len, clean_logits=False,
):
    if use_hopper_kpool_fallback(kv_cache.shape[1], q.device):
        return hopper_pooled_fp8_logits(q, kv_cache, weights, context_lens, block_tables, max_seq_len)
    import deep_gemm

    return deep_gemm.fp8_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, schedule_metadata,
        max_seq_len, clean_logits=clean_logits,
    )


def hopper_pooled_fp8_logits(q, kv_cache, weights, context_lens, block_tables, max_seq_len):
    """sum_h(weight_h * relu(dot(q_fp8_h, k_fp8))) * k_scale.

    q is [rows, 1, heads, 128]. A cache row contains all 16*128 FP8
    K bytes followed by 16 FP32 scales, NOT interleaved (K, scale) tuples.
    Verify is already flattened into one row per draft query by its caller.
    Invalid logits are -inf; existing pooled top-k lengths mask them as well.
    No CPU tensor reads or data-dependent allocation sizes are used.
    """
    assert q.is_cuda and q.dtype == torch.float8_e4m3fn
    assert q.ndim == 4 and q.shape[1] == 1 and q.shape[-1] == 128
    rows, _, heads, _ = q.shape
    assert 1 <= heads <= 64
    assert kv_cache.dtype == torch.uint8 and kv_cache.is_contiguous()
    assert kv_cache.ndim == 4 and kv_cache.shape[1:] == (16, 1, 132)
    assert weights.shape == (rows, heads)
    assert weights.dtype == torch.float32
    assert context_lens.shape == (rows, 1)
    assert context_lens.dtype == torch.int32
    assert block_tables.ndim == 2 and block_tables.shape[0] == rows
    assert block_tables.dtype == torch.int32
    assert 0 <= max_seq_len <= block_tables.shape[1] * 16
    assert all(t.device == q.device for t in (kv_cache, weights, context_lens, block_tables))
    logits = torch.empty((rows, max_seq_len), dtype=torch.float32, device=q.device)
    if rows == 0 or max_seq_len == 0:
        return logits
    _hopper_pooled_fp8_logits[(rows, triton.cdiv(max_seq_len, 32))](
        q.view(torch.uint8), kv_cache, kv_cache.view(torch.float32),
        weights, context_lens, block_tables, logits,
        q.stride(0), q.stride(2), q.stride(3),
        weights.stride(0), weights.stride(1), context_lens.stride(0),
        block_tables.stride(0), block_tables.stride(1), kv_cache.stride(0),
        MAX_SEQ_LEN=max_seq_len, NUM_HEADS=heads,
        BLOCK_H=max(16, triton.next_power_of_2(heads)), BLOCK_N=32, D=128,
        num_warps=4, num_stages=2,
    )
    return logits


@triton.jit
def _hopper_pooled_fp8_logits(
    Q, K, SCALE, WEIGHT, LENS, TABLE, OUT,
    q_row_stride, q_head_stride, q_dim_stride,
    weight_row_stride, weight_head_stride, lens_row_stride,
    table_row_stride, table_col_stride, cache_page_stride,
    MAX_SEQ_LEN: tl.constexpr, NUM_HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    hs = tl.arange(0, BLOCK_H)
    ds = tl.arange(0, D)
    seq_len = tl.load(LENS + row * lens_row_stride)
    # Uniform CTA branch: graph uses a context-wide grid, but short requests
    # must not issue Q/K loads or a dot for every entirely invalid tile.
    if tl.program_id(1) * BLOCK_N < seq_len:
        valid = (cols < MAX_SEQ_LEN) & (cols < seq_len)
        pages = tl.load(
            TABLE + row * table_row_stride + (cols // 16) * table_col_stride,
            mask=valid, other=0,
        ).to(tl.int64)
        slots = cols % 16
        # Triton 3.6 cannot cast the integer masked-load fill value directly
        # to fp8e4nv. Load exact bytes (zero byte is FP8 +0), then bitcast.
        q = tl.load(
            Q + row * q_row_stride + hs[:, None] * q_head_stride + ds[None, :] * q_dim_stride,
            mask=hs[:, None] < NUM_HEADS, other=0,
        ).to(tl.float8e4nv, bitcast=True)
        k = tl.load(
            K + pages[:, None] * cache_page_stride + slots[:, None] * D + ds[None, :],
            mask=valid[:, None], other=0,
        ).to(tl.float8e4nv, bitcast=True)
        dots = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        weight = tl.load(
            WEIGHT + row * weight_row_stride + hs * weight_head_stride,
            mask=hs < NUM_HEADS, other=0,
        ).to(tl.float32)
        # Scale data is stored after all K vectors within each physical page.
        scale = tl.load(
            SCALE + pages * (cache_page_stride // 4) + (16 * D // 4) + slots,
            mask=valid, other=0,
        )
        scores = tl.sum(tl.maximum(dots, 0.0) * weight[:, None], axis=0) * scale
        scores = tl.where(valid, scores, float("-inf"))
    else:
        scores = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    tl.store(OUT + row * MAX_SEQ_LEN + cols, scores, mask=cols < MAX_SEQ_LEN)
