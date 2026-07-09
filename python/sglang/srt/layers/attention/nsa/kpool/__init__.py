"""Kpool indexer: pooled-history sparse indexer for NSA models.

Layered architecture:

    planner.py    -- build the per-forward plan (CPU planning + H2D)
    indexer.py    -- the IndexerKPool class (forward_cuda entry point)
    kernels.py    -- triton kernels (compress / topk / gather / scatter)

Public API:
    IndexerKPool
    KPoolExtendPlan, KPoolWritePlan, PoolWriteRows, TailWriteRows, KPoolCpInfo
    init_kpool_extend_metadata, init_kpool_write_plan,
    init_pooled_paged_mqa_metadata, update_pooled_paged_mqa_metadata
"""

from sglang.srt.layers.attention.nsa.kpool.indexer import IndexerKPool
from sglang.srt.layers.attention.nsa.kpool.planner import (
    KPoolCpInfo,
    KPoolExtendPlan,
    KPoolWritePlan,
    PoolWriteRows,
    TailWriteRows,
    init_kpool_extend_metadata,
    init_kpool_write_plan,
    init_pooled_paged_mqa_metadata,
    update_pooled_paged_mqa_metadata,
)

__all__ = [
    "IndexerKPool",
    "KPoolCpInfo",
    "KPoolExtendPlan",
    "KPoolWritePlan",
    "PoolWriteRows",
    "TailWriteRows",
    "init_kpool_extend_metadata",
    "init_kpool_write_plan",
    "init_pooled_paged_mqa_metadata",
    "update_pooled_paged_mqa_metadata",
]
