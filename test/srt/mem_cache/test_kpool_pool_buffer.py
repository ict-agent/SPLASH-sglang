"""Regression tests for the dense kpool buffer layout.

NSATokenToKVPool packs ``PAGE_SIZE // index_kpool`` pool entries per
64-token page when kpool-compress is enabled (anchor mode keeps the
legacy ``slots = PAGE_SIZE`` layout, but is unreachable from the kpool
kernel path -- IndexerKPool requires kpool>1 with compress=True).
Dense layout reduces the index buffer row width from 8448 B to 528 B
(16x memory saving for kpool=16).

These tests cover the address-arithmetic invariants of the page-table
helpers (which now derive ``slots = PAGE_SIZE // pool_size`` internally)
and the device-side row width branching of NSATokenToKVPool. Higher-level
HiCache + end-to-end accuracy validation lives in the remote
sglang-glm5next-kpool harness.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.attention.nsa.kpool.page_table import (
    build_pooled_page_table_64,
    compute_pooled_write_locs,
    compute_pooled_write_locs_batched,
)


# ---------------- page_table address arithmetic ----------------


def test_build_pooled_page_table_64_dense_4per_page():
    # Dense: slots_per_pool_page = PAGE_SIZE // pool_size = 4
    # -> stride = 16*4//64 = 1 -> identity gather.
    page_table = torch.arange(64, dtype=torch.int32)
    out = build_pooled_page_table_64(page_table, pool_size=16)
    assert out.tolist() == list(range(64))


def test_compute_pooled_write_locs_dense_4per_page():
    # Dense: loc = packed_page * 4 + pool_id % 4,
    # packed_page picked from page_table_64[pool_id // 4 * 1] (1 token page per pool group).
    page_table = torch.arange(64, dtype=torch.int64) * 100
    pool_ids = torch.tensor([0, 1, 3, 4, 5, 7, 12], dtype=torch.int64)
    out = compute_pooled_write_locs(page_table, pool_ids, pool_size=16)
    # group 0 (pool 0..3) -> page_table[0] = 0
    # group 1 (pool 4..7) -> page_table[1] = 100
    # group 3 (pool 12..15) -> page_table[3] = 300
    assert out.tolist() == [
        0 * 4 + 0,
        0 * 4 + 1,
        0 * 4 + 3,
        100 * 4 + 0,
        100 * 4 + 1,
        100 * 4 + 3,
        300 * 4 + 0,
    ]


def test_compute_pooled_write_locs_batched_dense_4per_page():
    block_tables = torch.arange(2 * 64, dtype=torch.int64).reshape(2, 64) * 10
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    pool_ids = torch.tensor([0, 5, 0, 7], dtype=torch.int64)
    out = compute_pooled_write_locs_batched(
        block_tables, batch_idx, pool_ids, pool_size=16
    )
    # batch 0, pool 0 -> block_tables[0, 0]
    # batch 0, pool 5 -> block_tables[0, 1]  (5 // 4 = 1)
    # batch 1, pool 0 -> block_tables[1, 0]
    # batch 1, pool 7 -> block_tables[1, 1]
    assert out.tolist() == [
        block_tables[0, 0].item() * 4 + 0,
        block_tables[0, 1].item() * 4 + 1,
        block_tables[1, 0].item() * 4 + 0,
        block_tables[1, 1].item() * 4 + 3,
    ]


# ---------------- NSATokenToKVPool row-width branching ----------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nsa_pool_anchor_row_width_default():
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

    pool = NSATokenToKVPool(
        size=128,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
        index_kpool=1,
        index_kpool_compress=False,
    )
    assert pool.slots_per_pool_page == 64
    # row_bytes = 64 * (128 + 128//128 * 4) = 64 * 132 = 8448
    assert pool.index_k_with_scale_buffer[0].shape[1] == 64 * 132


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nsa_pool_dense_row_width_kpool16():
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

    pool = NSATokenToKVPool(
        size=128,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
        index_kpool=16,
        index_kpool_compress=True,
        max_running_requests=4,
    )
    assert pool.slots_per_pool_page == 4
    # row_bytes = 4 * (128 + 128//128 * 4) = 4 * 132 = 528
    assert pool.index_k_with_scale_buffer[0].shape[1] == 4 * 132
