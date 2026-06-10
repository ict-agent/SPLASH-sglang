"""Page-table arithmetic for the kpool path.

These helpers convert between three views of the NSA K-cache layout:

  * ``real_page_table``  -- (batch, num_pages); each page covers ``PAGE_SIZE``
                            tokens of FP8 K-cache. This is the DeepGEMM
                            ``fp8_paged_mqa_logits`` page table.
  * ``pooled_page_table`` -- (batch, num_page_groups); one entry per
                             pool_size consecutive real pages. The
                             compressed pool slots for a group are packed
                             into the *first* real page of that group.
  * ``write_locs``       -- flat index into the compressed cache,
                            ``page_idx * PAGE_SIZE + slot``.

No CUDA / Triton here -- pure torch index math. Triton kernels for the
compress/scatter/gather/topk passes live in ``kernels.py``.
"""

from __future__ import annotations

import torch

# DeepGEMM's fp8_paged_mqa_logits is hard-coded to page_size = 64 tokens.
# Every NSA kpool routine assumes the real K-cache page table has this
# page size; compressed pool slots are packed PAGE_SIZE-per-page.
PAGE_SIZE = 64


def build_pooled_page_table_64(
    page_table_64: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Pack one logical pool page into the first token page of each page group.

    Uses advanced indexing (gather) rather than strided slicing so the result
    is always a freshly allocated row-major tensor. Strided slicing produces
    a view whose .contiguous() short-circuits for shape==(1, 1), leaving
    stride(-1) == pool_size and breaking downstream kernels that require
    stride(-1) == 1 (e.g. deep_gemm.fp8_paged_mqa_logits).

    The dense kpool-compress layout (``slots = PAGE_SIZE // pool_size``,
    = 4 for kpool=16) makes ``stride = 1`` so the gather is identity --
    each token page already holds its own pool entries. IndexerKPool
    requires kpool>1 with compress=True (see indexer.py:104), so the
    anchor layout is unreachable from this function.
    """
    assert (
        PAGE_SIZE % pool_size == 0
    ), f"pool_size ({pool_size}) must divide page_size ({PAGE_SIZE})"
    slots_per_pool_page = PAGE_SIZE // pool_size
    stride = max(1, pool_size * slots_per_pool_page // PAGE_SIZE)
    idx = torch.arange(
        0, page_table_64.shape[-1], stride, device=page_table_64.device
    )
    return page_table_64[..., idx]


def compute_pooled_write_locs(
    page_table_64: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Map logical pooled-K ids to packed physical index-cache locations.

    Dense kpool-compress: ``slots = PAGE_SIZE // pool_size`` (= 4 for
    kpool=16), each token page holds 4 slots, address arithmetic is
    ``packed_page * 4 + pool_id % 4``. Anchor layout is unreachable from
    this function (see ``build_pooled_page_table_64`` for details).
    """
    assert page_table_64.ndim == 1
    pool_ids = pool_ids.to(torch.int64)
    slots_per_pool_page = PAGE_SIZE // pool_size
    token_pages_per_pool_group = max(1, pool_size * slots_per_pool_page // PAGE_SIZE)
    pool_page_group = torch.div(pool_ids, slots_per_pool_page, rounding_mode="floor")
    token_page_row = pool_page_group * token_pages_per_pool_group
    packed_page = page_table_64.index_select(0, token_page_row.to(torch.int64))
    return packed_page.to(torch.int64) * slots_per_pool_page + torch.remainder(
        pool_ids, slots_per_pool_page
    )


def compute_pooled_write_locs_batched(
    block_tables: torch.Tensor,
    batch_idx: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Vectorized version of ``compute_pooled_write_locs`` across batches.

    ``block_tables`` has shape ``(batch, num_pages)``. ``batch_idx[r]`` picks
    which row of ``block_tables`` pool ``r`` lives in.
    """
    assert block_tables.ndim == 2
    assert batch_idx.shape == pool_ids.shape
    pool_ids = pool_ids.to(torch.int64)
    batch_idx = batch_idx.to(torch.int64)
    slots_per_pool_page = PAGE_SIZE // pool_size
    token_pages_per_pool_group = max(1, pool_size * slots_per_pool_page // PAGE_SIZE)
    pool_page_group = torch.div(pool_ids, slots_per_pool_page, rounding_mode="floor")
    token_page_row = pool_page_group * token_pages_per_pool_group
    packed_page = block_tables[batch_idx, token_page_row]
    return packed_page.to(torch.int64) * slots_per_pool_page + torch.remainder(
        pool_ids, slots_per_pool_page
    )
