"""Kpool planner: build the layer-invariant kpool extend plan and the
decode paged-MQA metadata once per forward batch. Mutates the
caller-provided ``NSAMetadata`` via ``object.__setattr__`` (frozen
dataclass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.attention.nsa.kpool.kernels import INDEX_HEAD_DIM
from sglang.srt.layers.attention.nsa.kpool.page_table import (
    PAGE_SIZE,
    build_pooled_page_table_64,
    compute_pooled_write_locs_batched,
)
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.attention.nsa_backend import NSAMetadata, TopkTransformMethod
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclass(frozen=True)
class PoolWriteRows:
    """All pool slots to compress + write this forward, flattened across batches.

    ``n_from_tail`` is 0 for bulk pools and equals the splice prefix
    length for splice pools.
    """

    req: torch.Tensor  # int64 [N]
    pool_id: torch.Tensor  # int64 [N]
    n_from_tail: torch.Tensor  # int32 [N]
    chunk_src: torch.Tensor  # int64 [N]
    write_loc: torch.Tensor  # int64 [N]

    @property
    def is_empty(self) -> bool:
        return self.pool_id.shape[0] == 0


@dataclass(frozen=True)
class TailWriteRows:
    """Per-request tail-buffer write; one row per batch with leftover chunk tokens."""

    req: torch.Tensor  # int64 [B]
    dst_offset: torch.Tensor  # int32 [B]
    chunk_src: torch.Tensor  # int64 [B]
    n_write: torch.Tensor  # int32 [B]

    @property
    def is_empty(self) -> bool:
        return self.req.shape[0] == 0


@dataclass(frozen=True)
class KPoolCpInfo:
    """CP-ownership for kpool writes under nsa_enable_prefill_cp.

    Each row i is owned by ``owner_rank[i]`` -- the rank that writes its
    compressed slot. ``all_gather_and_scatter_pool_slots`` then
    propagates writes so every rank's buf converges.
    """

    size: int
    rank: int
    owner_rank: torch.Tensor  # int32 [N]


@dataclass(frozen=True)
class KPoolExtendPlan:
    """Precomputed kpool extend metadata, layer-invariant."""

    writes: PoolWriteRows
    tails: TailWriteRows

    pooled_seq_lens_expanded: torch.Tensor  # int32 [sum(extend_seq_lens)]

    # Concatenated per-batch pooled page table -- ``page_indices`` for a
    # single ``gather_index_k_scale_prefix_into`` filling a flat
    # ``[total_k_rows, head_dim]`` K buffer.
    ragged_concat_page_table: torch.Tensor  # int32 [sum_pool_pages]
    # Per-q K start row in the flat K buffer, page-aligned so each
    # batch's K starts at a PAGE_SIZE boundary (keeps the gather
    # kernel's ``page_indices[token_id // PAGE_SIZE]`` lookup correct).
    ragged_q_ks: torch.Tensor  # int32 [sum(extend_seq_lens)]
    ragged_total_k_rows: int  # sum_pool_pages * PAGE_SIZE
    # Per-layer scratch for the ragged gather. Allocated once here so all
    # NSA layers share the same buffers (alloc + free out of the layer
    # hot path). ``None`` when total_k_rows == 0.
    ragged_k_u8: Optional[torch.Tensor]  # uint8 [total_k_rows, head_dim]
    ragged_k_scale: Optional[torch.Tensor]  # fp32 [total_k_rows]
    # ``req_to_token[req_pool_idx_of_q, :max_seq_len]`` row-replicated;
    # None when topk_transform_method != PAGED or fuse-topk off.
    ragged_paged_page_table: Optional[torch.Tensor]  # int32 [sum_q, max_seq_len]

    cp: Optional[KPoolCpInfo] = None


@dataclass
class _KPoolCpuPlan:
    """Raw per-batch lists; converted to GPU tensors in ``_kpool_plan_to_gpu``."""

    pool_batch_idx: List[int] = field(default_factory=list)
    pool_req: List[int] = field(default_factory=list)
    pool_pool_id: List[int] = field(default_factory=list)
    pool_n_from_tail: List[int] = field(default_factory=list)
    pool_chunk_src: List[int] = field(default_factory=list)

    tail_req: List[int] = field(default_factory=list)
    tail_dst_offset: List[int] = field(default_factory=list)
    tail_chunk_src: List[int] = field(default_factory=list)
    tail_n_write: List[int] = field(default_factory=list)

    ragged_batch_idx: List[int] = field(default_factory=list)
    ragged_q_len: List[int] = field(default_factory=list)
    ragged_pool_pages: List[int] = field(default_factory=list)


def _kpool_cpu_plan(
    forward_batch: "ForwardBatch",
    pool_size: int,
) -> _KPoolCpuPlan:
    """Emit pool rows for every pool whose right boundary lies in
    ``(first_pos, seq_len]``, plus a tail row for batches with leftover
    chunk tokens (skipped when the chunk aligns to a pool boundary).

    Row 0 may "splice" ``first_slot`` saved-tail tokens with
    ``pool_size - first_slot`` chunk tokens to close the mid-pool the
    prefix started in; subsequent rows are aligned-bulk pools. When
    ``first_slot == 0`` the splice degenerates to a plain bulk row
    (``n_from_tail = 0``), so a single uniform code path covers both.
    """
    plan = _KPoolCpuPlan()
    # IndexerKPool guards pool_size>1 ∧ compress=True (see indexer.py:104),
    # so the dense layout is the only reachable case here.
    slots_per_pool_page = PAGE_SIZE // pool_size

    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
    if isinstance(extend_seq_lens_cpu, torch.Tensor):
        extend_seq_lens_cpu = extend_seq_lens_cpu.tolist()
    seq_lens_cpu = forward_batch.seq_lens_cpu.tolist()
    req_pool_indices_cpu = forward_batch.req_pool_indices.tolist()

    q_offset = 0
    for i in range(forward_batch.batch_size):
        q_len = extend_seq_lens_cpu[i]
        assert (
            q_len > 0
        ), f"extend_seq_lens_cpu[{i}] = {q_len}; expected > 0 in extend-prefill"

        seq_len = seq_lens_cpu[i]
        req = req_pool_indices_cpu[i]
        first_pos = seq_len - q_len
        first_slot = first_pos % pool_size
        base_pool = first_pos // pool_size
        pool_seq_len = seq_len // pool_size
        n_pool = pool_seq_len - base_pool

        plan.ragged_batch_idx.append(i)
        plan.ragged_q_len.append(q_len)
        plan.ragged_pool_pages.append(
            (pool_seq_len + slots_per_pool_page - 1) // slots_per_pool_page
        )

        # ``list.extend`` over range/[x]*n stays on CPython's C-level
        # fastpath; ~2-3x faster than N Python appends for typical
        # low-B, high-q_len prefill where n_pool is large.
        if n_pool > 0:
            plan.pool_batch_idx.extend([i] * n_pool)
            plan.pool_req.extend([req] * n_pool)
            plan.pool_pool_id.extend(range(base_pool, base_pool + n_pool))
            plan.pool_n_from_tail.append(first_slot)
            plan.pool_n_from_tail.extend([0] * (n_pool - 1))
            bulk_start = q_offset + pool_size - first_slot
            plan.pool_chunk_src.append(q_offset)
            plan.pool_chunk_src.extend(
                range(bulk_start, bulk_start + (n_pool - 1) * pool_size, pool_size)
            )

        # Tail: leftover chunk tokens that don't fill a pool. Skipped
        # when n_remain == 0 (chunk aligned to a pool boundary); pool
        # rows already covered all chunk tokens.
        consumed = max(0, n_pool * pool_size - first_slot)
        n_remain = q_len - consumed
        if n_remain > 0:
            # When n_pool == 0 nothing closed the mid-pool, so tail
            # extends it starting at first_slot; otherwise tail starts
            # fresh at 0.
            dst_offset = first_slot if n_pool == 0 else 0
            plan.tail_req.append(req)
            plan.tail_dst_offset.append(dst_offset)
            plan.tail_chunk_src.append(q_offset + consumed)
            plan.tail_n_write.append(n_remain)

        q_offset += q_len

    return plan


def _kpool_plan_to_gpu(
    cpu: _KPoolCpuPlan,
    metadata: "NSAMetadata",
    forward_batch: "ForwardBatch",
    pool_size: int,
    topk_transform_method: "TopkTransformMethod",
) -> KPoolExtendPlan:
    """Pack lists into two pinned tensors (int64 indices + int32 small
    counts) for one H2D each; ragged-topk ``src_idx`` is computed on
    GPU from per-batch ``pool_pages``.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.nsa_backend import TopkTransformMethod

    device = forward_batch.seq_lens.device
    n_pool = len(cpu.pool_pool_id)
    n_tail = len(cpu.tail_req)
    n_rag = len(cpu.ragged_batch_idx)

    # IndexerKPool requires pool_size>1 ∧ compress=True (indexer.py:104).
    slots_per_pool_page = PAGE_SIZE // pool_size
    total_pool_pages = sum(cpu.ragged_pool_pages)
    ragged_total_k_rows = total_pool_pages * slots_per_pool_page

    need_paged = (
        topk_transform_method == TopkTransformMethod.PAGED
        and envs.SGLANG_NSA_FUSE_TOPK.get()
        and n_rag > 0
    )

    # int64 H2D: pool_req | pool_pool_id | pool_chunk_src | pool_batch_idx
    #          | tail_req | tail_chunk_src | ragged_batch_idx
    i64_total = 4 * n_pool + 2 * n_tail + n_rag
    if i64_total > 0:
        i64_cpu = torch.tensor(
            cpu.pool_req
            + cpu.pool_pool_id
            + cpu.pool_chunk_src
            + cpu.pool_batch_idx
            + cpu.tail_req
            + cpu.tail_chunk_src
            + cpu.ragged_batch_idx,
            dtype=torch.int64,
            pin_memory=True,
        )
        i64_gpu = i64_cpu.to(device, non_blocking=True)
        c = 0
        pool_req_t = i64_gpu[c : c + n_pool]
        c += n_pool
        pool_pool_id_t = i64_gpu[c : c + n_pool]
        c += n_pool
        pool_chunk_src_t = i64_gpu[c : c + n_pool]
        c += n_pool
        pool_batch_idx_t = i64_gpu[c : c + n_pool]
        c += n_pool
        tail_req_t = i64_gpu[c : c + n_tail]
        c += n_tail
        tail_chunk_src_t = i64_gpu[c : c + n_tail]
        c += n_tail
        ragged_batch_idx_t = i64_gpu[c : c + n_rag]
    else:
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
        pool_req_t = pool_pool_id_t = pool_chunk_src_t = pool_batch_idx_t = empty_i64
        tail_req_t = tail_chunk_src_t = empty_i64
        ragged_batch_idx_t = empty_i64

    # int32 H2D: pool_n_from_tail | tail_dst_offset | tail_n_write
    #          | ragged_pool_pages | ragged_q_len
    i32_total = n_pool + 2 * n_tail + 2 * n_rag
    if i32_total > 0:
        i32_cpu = torch.tensor(
            cpu.pool_n_from_tail
            + cpu.tail_dst_offset
            + cpu.tail_n_write
            + cpu.ragged_pool_pages
            + cpu.ragged_q_len,
            dtype=torch.int32,
            pin_memory=True,
        )
        i32_gpu = i32_cpu.to(device, non_blocking=True)
        c = 0
        pool_n_from_tail_t = i32_gpu[c : c + n_pool]
        c += n_pool
        tail_dst_offset_t = i32_gpu[c : c + n_tail]
        c += n_tail
        tail_n_write_t = i32_gpu[c : c + n_tail]
        c += n_tail
        ragged_pool_pages_t = i32_gpu[c : c + n_rag]
        c += n_rag
        ragged_q_len_t = i32_gpu[c : c + n_rag]
    else:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        pool_n_from_tail_t = empty_i32
        tail_dst_offset_t = tail_n_write_t = empty_i32
        ragged_pool_pages_t = ragged_q_len_t = empty_i32

    if n_pool > 0:
        pool_write_locs = compute_pooled_write_locs_batched(
            metadata.real_page_table,
            pool_batch_idx_t,
            pool_pool_id_t,
            pool_size,
        )
    else:
        pool_write_locs = torch.empty((0,), dtype=torch.int64, device=device)

    pooled_page_table_all = build_pooled_page_table_64(
        metadata.real_page_table,
        pool_size,
    ).contiguous()

    pooled_seq_lens_expanded = torch.div(
        metadata.nsa_seqlens_expanded, pool_size, rounding_mode="floor"
    ).to(torch.int32)

    if n_rag > 0:
        max_pool_pages = pooled_page_table_all.shape[1]
        # src_idx[r] = ragged_batch_idx[k] * max_pool_pages + (r - cu_pages[k])
        # for the k whose [cu_pages[k], cu_pages[k+1]) contains r.
        cu_pages_excl = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(ragged_pool_pages_t, dim=0, dtype=torch.int32),
            )
        )[:-1]
        page_to_sel = torch.repeat_interleave(
            torch.arange(n_rag, device=device, dtype=torch.int32),
            ragged_pool_pages_t,
        )
        all_pages = torch.arange(total_pool_pages, device=device, dtype=torch.int32)
        intra = all_pages - cu_pages_excl.index_select(0, page_to_sel)
        src_idx_t = (
            ragged_batch_idx_t.index_select(0, page_to_sel.to(torch.int64))
            * max_pool_pages
            + intra
        )

        q_ks_per_batch_t = cu_pages_excl * slots_per_pool_page
        ragged_q_ks = torch.repeat_interleave(q_ks_per_batch_t, ragged_q_len_t)

        ragged_concat_page_table = (
            pooled_page_table_all.view(-1).index_select(0, src_idx_t).to(torch.int32)
        )
    else:
        empty_i32_dev = torch.empty((0,), dtype=torch.int32, device=device)
        ragged_concat_page_table = empty_i32_dev
        ragged_q_ks = empty_i32_dev

    # Build once per forward (vs per-layer per-batch via .expand) to
    # avoid an O(B*layers) Python loop in the indexer.
    ragged_paged_page_table = None
    if need_paged:
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        # max_seq_len varies per batch; use the global max and rely on
        # consumer masking via per-row `lengths`.
        max_seq_len = int(forward_batch.seq_lens_cpu.max().item())
        req_pool_indices_per_batch = forward_batch.req_pool_indices.to(
            torch.int64
        ).index_select(0, ragged_batch_idx_t)
        req_pool_indices_per_q = torch.repeat_interleave(
            req_pool_indices_per_batch, ragged_q_len_t
        )
        ragged_paged_page_table = req_to_token[req_pool_indices_per_q, :max_seq_len].to(
            torch.int32
        )

    # Layer-shared scratch for the ragged gather + fp8_mqa_logits.
    # All NSA layers consume the same (total_k_rows, head_dim) shape,
    # so alloc once and reuse.
    if ragged_total_k_rows > 0:
        ragged_k_u8 = torch.empty(
            (ragged_total_k_rows, INDEX_HEAD_DIM), dtype=torch.uint8, device=device
        )
        ragged_k_scale = torch.empty(
            (ragged_total_k_rows,), dtype=torch.float32, device=device
        )
    else:
        ragged_k_u8 = None
        ragged_k_scale = None

    return KPoolExtendPlan(
        writes=PoolWriteRows(
            req=pool_req_t,
            pool_id=pool_pool_id_t,
            n_from_tail=pool_n_from_tail_t,
            chunk_src=pool_chunk_src_t,
            write_loc=pool_write_locs,
        ),
        tails=TailWriteRows(
            req=tail_req_t,
            dst_offset=tail_dst_offset_t,
            chunk_src=tail_chunk_src_t,
            n_write=tail_n_write_t,
        ),
        pooled_seq_lens_expanded=pooled_seq_lens_expanded,
        ragged_concat_page_table=ragged_concat_page_table,
        ragged_q_ks=ragged_q_ks,
        ragged_total_k_rows=ragged_total_k_rows,
        ragged_k_u8=ragged_k_u8,
        ragged_k_scale=ragged_k_scale,
        ragged_paged_page_table=ragged_paged_page_table,
        cp=_kpool_cp_owner_rank(forward_batch, n_pool, device),
    )


def _kpool_cp_owner_rank(
    forward_batch: "ForwardBatch",
    n_pool: int,
    device: torch.device,
) -> Optional[KPoolCpInfo]:
    """Owner rank = ``row_idx % cp_size``; balances writes by flat row
    index regardless of per-request pool distribution.
    """
    from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp

    if not nsa_use_prefill_cp(forward_batch):
        return None

    from sglang.srt.layers.dp_attention import (
        get_attention_cp_rank,
        get_attention_cp_size,
    )

    cp_size = get_attention_cp_size()
    if cp_size <= 1:
        return None

    owner = (
        torch.arange(n_pool, dtype=torch.int32, device=device) % cp_size
        if n_pool > 0
        else torch.empty((0,), dtype=torch.int32, device=device)
    )
    return KPoolCpInfo(size=cp_size, rank=get_attention_cp_rank(), owner_rank=owner)


def init_kpool_extend_metadata(
    metadata: "NSAMetadata",
    forward_batch: "ForwardBatch",
    *,
    pool_size: int,
    real_page_size: int,
    topk_transform_method: "TopkTransformMethod",
) -> None:
    """Build the layer-invariant kpool extend plan once per forward.

    No-op unless: pool_size > 1, extend_without_speculative, page_size
    == 64, and 64 % pool_size == 0 (kpool kernel requirements).
    """
    if (
        pool_size <= 1
        or not forward_batch.forward_mode.is_extend_without_speculative()
        or forward_batch.extend_seq_lens_cpu is None
        or forward_batch.seq_lens_cpu is None
        or real_page_size != 64
        or 64 % pool_size != 0
    ):
        return

    cpu = _kpool_cpu_plan(forward_batch, pool_size)
    plan = _kpool_plan_to_gpu(
        cpu,
        metadata,
        forward_batch,
        pool_size,
        topk_transform_method,
    )
    object.__setattr__(metadata, "kpool_extend_plan", plan)


def init_pooled_paged_mqa_metadata(
    metadata: "NSAMetadata",
    seqlens_32: torch.Tensor,
    real_page_table: torch.Tensor,
    forward_mode: "ForwardMode",
    *,
    pool_size: int,
    real_page_size: int,
) -> None:
    """Build decode-side pooled cache seqlens + page table + deep_gemm schedule."""
    if (
        pool_size <= 1
        or not is_cuda()
        or not forward_mode.is_decode_or_idle()
        or real_page_size != 64
        or 64 % pool_size != 0
    ):
        return

    slots_per_pool_page = PAGE_SIZE // pool_size
    object.__setattr__(metadata, "pooled_index_kpool", pool_size)
    object.__setattr__(
        metadata,
        "pooled_cache_seqlens_int32",
        torch.div(seqlens_32, pool_size, rounding_mode="floor").to(torch.int32),
    )
    object.__setattr__(
        metadata,
        "pooled_real_page_table",
        build_pooled_page_table_64(real_page_table, pool_size).contiguous(),
    )
    try:
        import deep_gemm

        object.__setattr__(
            metadata,
            "pooled_paged_mqa_schedule_metadata",
            deep_gemm.get_paged_mqa_logits_metadata(
                metadata.pooled_cache_seqlens_int32.unsqueeze(-1),
                slots_per_pool_page,
                deep_gemm.get_num_sms(),
            ),
        )
    except (ImportError, ModuleNotFoundError):
        object.__setattr__(metadata, "pooled_paged_mqa_schedule_metadata", None)


def update_pooled_paged_mqa_metadata(
    metadata: "NSAMetadata",
    seqlens_32: torch.Tensor,
    real_page_table: torch.Tensor,
    forward_mode: "ForwardMode",
    *,
    pool_size: int,
    real_page_size: int,
    page_tables_already_updated: bool = False,
) -> None:
    """In-place refresh of decode-side pooled metadata (cuda-graph replay)."""
    if (
        pool_size <= 1
        or not is_cuda()
        or not forward_mode.is_decode_or_idle()
        or real_page_size != 64
        or 64 % pool_size != 0
    ):
        object.__setattr__(metadata, "pooled_index_kpool", 1)
        object.__setattr__(metadata, "pooled_cache_seqlens_int32", None)
        object.__setattr__(metadata, "pooled_real_page_table", None)
        object.__setattr__(metadata, "pooled_paged_mqa_schedule_metadata", None)
        return

    slots_per_pool_page = PAGE_SIZE // pool_size
    pool_seqlens = torch.div(seqlens_32, pool_size, rounding_mode="floor").to(
        torch.int32
    )

    object.__setattr__(metadata, "pooled_index_kpool", pool_size)
    if metadata.pooled_cache_seqlens_int32 is None:
        object.__setattr__(metadata, "pooled_cache_seqlens_int32", pool_seqlens)
    else:
        metadata.pooled_cache_seqlens_int32[: pool_seqlens.shape[0]].copy_(pool_seqlens)

    if not page_tables_already_updated or metadata.pooled_real_page_table is None:
        pool_page_table = build_pooled_page_table_64(real_page_table, pool_size)
        if metadata.pooled_real_page_table is None:
            object.__setattr__(
                metadata, "pooled_real_page_table", pool_page_table.contiguous()
            )
        else:
            rows, cols = pool_page_table.shape
            metadata.pooled_real_page_table[:rows, :cols].copy_(pool_page_table)

    try:
        import deep_gemm

        new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            pool_seqlens.unsqueeze(-1), slots_per_pool_page, deep_gemm.get_num_sms()
        )
        if metadata.pooled_paged_mqa_schedule_metadata is None:
            object.__setattr__(
                metadata, "pooled_paged_mqa_schedule_metadata", new_schedule
            )
        else:
            metadata.pooled_paged_mqa_schedule_metadata.copy_(new_schedule)
    except (ImportError, ModuleNotFoundError):
        object.__setattr__(metadata, "pooled_paged_mqa_schedule_metadata", None)
