"""Kpool planner: build the layer-invariant kpool extend plan and the
decode paged-MQA metadata once per forward batch.

The ``init_*`` builders return a new ``NSAMetadata`` (frozen) via
``dataclasses.replace``; the ``update_*`` variant mutates in place
because cuda-graph replay requires the captured tensors to stay put.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa.kpool.kernels import (
    INDEX_HEAD_DIM,
    kpool_build_ragged_layout,
)
from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
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
    propagates writes so every rank's buf converges. ``local_write_mask``
    is the per-row ``owner_rank == rank`` bool, materialized once per
    forward so every NSA layer can reuse it.
    """

    size: int
    rank: int
    owner_rank: torch.Tensor  # int32 [N]
    local_write_mask: torch.Tensor  # bool [N]


@dataclass(frozen=True)
class KPoolExtendPlan:
    """Precomputed kpool extend metadata, layer-invariant.

    Compress side (``writes`` / ``tails``) is always full-seq: K is
    all-gathered before compress under CP, so write_locs index the full
    pool table.

    Local view (``ragged_*``, ``pooled_seq_lens_expanded``,
    ``seq_lens_expanded``) is rank-local under CP round-robin-split,
    full-seq otherwise. It is sized to match this rank's q
    (``forward_cuda`` passes the rank-local ``q_fp8`` straight into
    ``_get_topk_ragged`` without any all_gather).
    """

    writes: PoolWriteRows
    tails: TailWriteRows

    pooled_seq_lens_expanded: torch.Tensor  # int32 [sum_q]
    seq_lens_expanded: torch.Tensor  # int32 [sum_q]

    # page_indices for a single gather_index_k_scale_prefix_into into a
    # flat [total_k_rows, head_dim] K buffer.
    ragged_concat_page_table: torch.Tensor  # int32 [sum_pool_pages]
    # Per-q K start row; page-aligned so the gather kernel's
    # page_indices[token_id // page_size] lookup is correct.
    # Per-q K end row; equals ``ragged_q_ks + pooled_seq_lens_expanded``.
    # Precomputed here so every NSA layer reuses the same tensor instead
    # of re-adding the two each forward.
    ragged_q_ks: torch.Tensor  # int32 [sum_q]
    ragged_q_ke: torch.Tensor  # int32 [sum_q]
    ragged_total_k_rows: int  # sum_pool_pages * page_size
    # Layer-shared scratch (alloc out of the per-layer hot path).
    ragged_k_u8: Optional[torch.Tensor]  # uint8 [total_k_rows, head_dim]
    ragged_k_scale: Optional[torch.Tensor]  # fp32 [total_k_rows]
    # req_to_token row-replicated; None unless PAGED + fuse-topk on.
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

    ragged_q_len: List[int] = field(default_factory=list)
    ragged_pool_pages: List[int] = field(default_factory=list)
    # Exclusive prefix sums (per batch start offsets). Built on CPU and
    # H2D'd alongside the other i32 counters so the Triton ragged-layout
    # kernel can read them directly without a GPU cumsum.
    cu_pages_excl: List[int] = field(default_factory=list)
    cu_q_len_excl: List[int] = field(default_factory=list)
    # Rolling sum of ragged_pool_pages, kept here so _kpool_plan_to_gpu
    # avoids a redundant Python sum() pass.
    total_pool_pages: int = 0
    # n_rag = len(ragged_q_len) = batch_size (every batch contributes a row).
    # ragged_batch_idx is therefore arange(batch_size); recomputed in
    # _kpool_plan_to_gpu instead of being H2D'd as a redundant list.


def _append_compress_rows(
    plan: _KPoolCpuPlan,
    pool_size: int,
    batch_size: int,
    extend_seq_lens_cpu: List[int],
    seq_lens_cpu: List[int],
    req_pool_indices_cpu: List[int],
) -> None:
    """Compress side: per-pool write rows + per-batch tail rows.

    Under CP round-robin-split, ``key`` is all-gathered to full-seq before
    compress, so the compress side still operates on the *full* batch /
    seq layout regardless of CP split mode.

    Row 0 may "splice" ``first_slot`` saved-tail tokens with
    ``pool_size - first_slot`` chunk tokens to close the mid-pool the
    prefix started in; subsequent rows are aligned-bulk pools. When
    ``first_slot == 0`` the splice degenerates to a plain bulk row
    (``n_from_tail = 0``), so a single uniform code path covers both.
    """
    q_offset = 0
    for i in range(batch_size):
        q_len = extend_seq_lens_cpu[i]
        assert q_len > 0, f"extend_seq_lens_cpu[{i}] = {q_len}; expected > 0"

        seq_len = seq_lens_cpu[i]
        req = req_pool_indices_cpu[i]
        first_pos = seq_len - q_len
        first_slot = first_pos % pool_size
        base_pool = first_pos // pool_size
        pool_seq_len = seq_len // pool_size
        n_pool = pool_seq_len - base_pool

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

        consumed = max(0, n_pool * pool_size - first_slot)
        n_remain = q_len - consumed
        if n_remain > 0:
            dst_offset = first_slot if n_pool == 0 else 0
            plan.tail_req.append(req)
            plan.tail_dst_offset.append(dst_offset)
            plan.tail_chunk_src.append(q_offset + consumed)
            plan.tail_n_write.append(n_remain)

        q_offset += q_len


def _append_local_rows(
    plan: _KPoolCpuPlan,
    pool_size: int,
    page_size: int,
    local_extend_seq_lens_cpu: List[int],
    local_seq_lens_cpu: List[int],
) -> None:
    """Local view: per-batch ragged layout (q_len + pool_pages + prefix sums).

    Under CP round-robin-split, these are the *rank-local* per-batch
    counts (only batches present on this rank, with this rank's q
    counts); under non-CP they equal the full-seq counts. q is NOT
    gathered, so the ragged layout sizes match this rank's q footprint.
    """
    slots_per_page = page_size // pool_size
    q_offset = 0
    for q_len, seq_len in zip(
        local_extend_seq_lens_cpu, local_seq_lens_cpu, strict=True
    ):
        assert q_len > 0, f"local_extend_seq_lens_cpu has non-positive {q_len = }"
        plan.ragged_q_len.append(q_len)
        pool_seq_len = seq_len // pool_size
        pool_pages_i = (pool_seq_len + slots_per_page - 1) // slots_per_page
        plan.ragged_pool_pages.append(pool_pages_i)
        plan.cu_pages_excl.append(plan.total_pool_pages)
        plan.cu_q_len_excl.append(q_offset)
        plan.total_pool_pages += pool_pages_i
        q_offset += q_len


def _kpool_cpu_plan(
    forward_batch: "ForwardBatch",
    pool_size: int,
    page_size: int = 64,
    *,
    local_extend_seq_lens_cpu: Optional[List[int]] = None,
    local_seq_lens_cpu: Optional[List[int]] = None,
) -> _KPoolCpuPlan:
    """Build the combined compress + local CPU plan.

    Compress side: always full-seq (K is all-gathered before compress
    under CP, so write_locs index the full pool table).

    Local view: same full layout when no override is provided; otherwise
    the rank-local slice for CP round-robin-split, which lets
    ``_get_topk_ragged`` build per-q ks/ke for this rank's tokens
    instead of all-gathering q. ``local_*`` come paired (both None or
    both set).
    """
    plan = _KPoolCpuPlan()

    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
    if isinstance(extend_seq_lens_cpu, torch.Tensor):
        extend_seq_lens_cpu = extend_seq_lens_cpu.tolist()
    seq_lens_cpu = forward_batch.seq_lens_cpu.tolist()
    req_pool_indices_cpu = forward_batch.req_pool_indices.tolist()

    _append_compress_rows(
        plan,
        pool_size,
        forward_batch.batch_size,
        extend_seq_lens_cpu,
        seq_lens_cpu,
        req_pool_indices_cpu,
    )

    if local_extend_seq_lens_cpu is None:
        local_extend_seq_lens_cpu = extend_seq_lens_cpu
        local_seq_lens_cpu = seq_lens_cpu

    _append_local_rows(
        plan,
        pool_size,
        page_size,
        local_extend_seq_lens_cpu,
        local_seq_lens_cpu,
    )
    return plan


def _kpool_plan_to_gpu(
    cpu: _KPoolCpuPlan,
    forward_batch: "ForwardBatch",
    full_real_page_table: torch.Tensor,
    local_real_page_table: torch.Tensor,
    local_seqlens_expanded: torch.Tensor,
    local_req_pool_indices: torch.Tensor,
    local_max_seq_len: int,
    pool_size: int,
    page_size: int,
    topk_transform_method: "TopkTransformMethod",
) -> KPoolExtendPlan:
    """Pack lists into two pinned tensors (int64 indices + int32 small
    counts) for one H2D each; ragged-topk ``src_idx`` is computed on
    GPU from per-batch ``pool_pages``.

    ``full_real_page_table`` is used for the compress-side ``write_locs``
    (the FP8 cache index is full-seq because K is gathered before
    compress). ``local_real_page_table`` / ``local_seqlens_expanded`` are
    rank-local under CP round-robin-split, matching this rank's q
    footprint -- see ``_get_topk_ragged`` for how they're consumed.
    """
    from sglang.srt.layers.attention.nsa_backend import TopkTransformMethod

    device = forward_batch.seq_lens.device
    n_pool = len(cpu.pool_pool_id)
    n_tail = len(cpu.tail_req)
    n_rag = len(cpu.ragged_q_len)

    slots_per_page = page_size // pool_size
    total_pool_pages = cpu.total_pool_pages
    ragged_total_k_rows = total_pool_pages * slots_per_page

    need_paged = (
        topk_transform_method == TopkTransformMethod.PAGED
        and envs.SGLANG_NSA_FUSE_TOPK.get()
        and n_rag > 0
    )

    # int64 H2D: pool_req | pool_pool_id | pool_chunk_src | pool_batch_idx
    #          | tail_req | tail_chunk_src
    i64_total = 4 * n_pool + 2 * n_tail
    if i64_total > 0:
        i64_cpu = torch.tensor(
            cpu.pool_req
            + cpu.pool_pool_id
            + cpu.pool_chunk_src
            + cpu.pool_batch_idx
            + cpu.tail_req
            + cpu.tail_chunk_src,
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
    else:
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
        pool_req_t = pool_pool_id_t = pool_chunk_src_t = pool_batch_idx_t = empty_i64
        tail_req_t = tail_chunk_src_t = empty_i64

    # int32 H2D: pool_n_from_tail | tail_dst_offset | tail_n_write
    #          | ragged_pool_pages | ragged_q_len
    #          | cu_pages_excl | cu_q_len_excl
    i32_total = n_pool + 2 * n_tail + 4 * n_rag
    if i32_total > 0:
        i32_cpu = torch.tensor(
            cpu.pool_n_from_tail
            + cpu.tail_dst_offset
            + cpu.tail_n_write
            + cpu.ragged_pool_pages
            + cpu.ragged_q_len
            + cpu.cu_pages_excl
            + cpu.cu_q_len_excl,
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
        c += n_rag
        cu_pages_excl_t = i32_gpu[c : c + n_rag]
        c += n_rag
        cu_q_len_excl_t = i32_gpu[c : c + n_rag]
    else:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        pool_n_from_tail_t = empty_i32
        tail_dst_offset_t = tail_n_write_t = empty_i32
        ragged_pool_pages_t = ragged_q_len_t = empty_i32
        cu_pages_excl_t = cu_q_len_excl_t = empty_i32

    # Compress side reads ``full_real_page_table``; local view reads
    # ``local_real_page_table`` / ``local_seqlens_expanded``. They diverge
    # under CP round-robin-split: K is gathered before compress (so the
    # compress plan is full-seq), but q stays rank-local for topk.

    if n_pool > 0:
        # Compressed pool slots are packed `page_size`-per-page (DeepGEMM's
        # fp8_paged_mqa_logits historically expects 64-token pages). Each
        # logical pooled-K id maps to `packed_page * slots_per_page +
        # pool_id % slots_per_page`, with `packed_page` looked up via the
        # per-batch page table row chosen by `pool_batch_idx_t`.
        pool_page_group = torch.div(
            pool_pool_id_t, slots_per_page, rounding_mode="floor"
        )
        packed_page = full_real_page_table[pool_batch_idx_t, pool_page_group].to(
            torch.int64
        )
        pool_write_locs = packed_page * slots_per_page + torch.remainder(
            pool_pool_id_t, slots_per_page
        )
    else:
        pool_write_locs = torch.empty((0,), dtype=torch.int64, device=device)

    pooled_seq_lens_expanded = torch.div(
        local_seqlens_expanded, pool_size, rounding_mode="floor"
    ).to(torch.int32)

    if n_rag > 0:
        # Single Triton kernel writes concat_page_table / ragged_q_ks / ragged_q_ke.
        # Per-batch prefix sums (cu_pages_excl, cu_q_len_excl) were built on CPU
        # and arrived via the i32 H2D above, so no GPU cumsum is needed here.
        (
            ragged_concat_page_table,
            ragged_q_ks,
            ragged_q_ke,
        ) = kpool_build_ragged_layout(
            full_page_table=local_real_page_table,
            cu_pages_excl=cu_pages_excl_t,
            ragged_pool_pages=ragged_pool_pages_t,
            cu_q_len_excl=cu_q_len_excl_t,
            ragged_q_len=ragged_q_len_t,
            pooled_seq_lens_expanded=pooled_seq_lens_expanded,
            slots_per_page=slots_per_page,
            total_pool_pages=total_pool_pages,
            total_q=pooled_seq_lens_expanded.shape[0],
        )
    else:
        empty_i32_dev = torch.empty((0,), dtype=torch.int32, device=device)
        ragged_concat_page_table = empty_i32_dev
        ragged_q_ks = empty_i32_dev
        ragged_q_ke = empty_i32_dev

    # Build once per forward to avoid an O(B*layers) Python loop in the indexer.
    ragged_paged_page_table = None
    if need_paged:
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        # ``local_max_seq_len`` is the per-rank max already on CPU; saves a
        # GPU sync over ``local_seqlens_expanded.max()``. Consumer masks
        # via per-row ``lengths``.
        req_pool_indices_per_q = torch.repeat_interleave(
            local_req_pool_indices.to(torch.int64), ragged_q_len_t
        )
        ragged_paged_page_table = req_to_token[
            req_pool_indices_per_q, :local_max_seq_len
        ].to(torch.int32)

    # Layer-shared scratch (all NSA layers share these (total_k_rows, head_dim) buffers).
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
        seq_lens_expanded=local_seqlens_expanded,
        ragged_concat_page_table=ragged_concat_page_table,
        ragged_q_ks=ragged_q_ks,
        ragged_q_ke=ragged_q_ke,
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
    if not nsa_use_prefill_cp(forward_batch):
        return None

    cp_size = get_attention_cp_size()
    if cp_size <= 1:
        return None

    cp_rank = get_attention_cp_rank()
    if n_pool > 0:
        owner = torch.arange(n_pool, dtype=torch.int32, device=device) % cp_size
        local_write_mask = owner == cp_rank
    else:
        owner = torch.empty((0,), dtype=torch.int32, device=device)
        local_write_mask = torch.empty((0,), dtype=torch.bool, device=device)
    return KPoolCpInfo(
        size=cp_size,
        rank=cp_rank,
        owner_rank=owner,
        local_write_mask=local_write_mask,
    )


def init_kpool_extend_metadata(
    metadata: "NSAMetadata",
    forward_batch: "ForwardBatch",
    *,
    pool_size: int,
    real_page_size: int,
    topk_transform_method: "TopkTransformMethod",
    full_real_page_table: torch.Tensor,
    full_seqlens_expanded: torch.Tensor,
    local_real_page_table: Optional[torch.Tensor] = None,
    local_seqlens_expanded: Optional[torch.Tensor] = None,
    local_extend_seq_lens_cpu: Optional[List[int]] = None,
    local_seq_lens_cpu: Optional[List[int]] = None,
    local_req_pool_indices: Optional[torch.Tensor] = None,
    local_max_seq_len: Optional[int] = None,
) -> "NSAMetadata":
    """Build the layer-invariant kpool extend plan once per forward.

    Two views:
      * Compress (full-seq) always reads ``full_real_page_table`` --
        K is gathered before compress under CP, so write_locs index the
        full pool table.
      * Local view defaults to the full layout; under CP round-robin-split
        the caller passes the rank-local ``local_*`` tensors and per-batch
        counts so topk runs on this rank's q slice without a q all_gather.

    Returns input unchanged when the gating fails (pool_size > 1,
    extend_without_speculative, page_size == 64, and 64 % pool_size == 0).
    """
    if (
        pool_size <= 1
        or not forward_batch.forward_mode.is_extend_without_speculative()
        or forward_batch.extend_seq_lens_cpu is None
        or forward_batch.seq_lens_cpu is None
        or real_page_size != 64
        or real_page_size % pool_size != 0
    ):
        return metadata

    if local_real_page_table is None:
        local_real_page_table = full_real_page_table
    if local_seqlens_expanded is None:
        local_seqlens_expanded = full_seqlens_expanded
    if local_req_pool_indices is None:
        local_req_pool_indices = forward_batch.req_pool_indices
    if local_max_seq_len is None:
        local_max_seq_len = int(forward_batch.seq_lens_cpu.max().item())

    cpu = _kpool_cpu_plan(
        forward_batch,
        pool_size,
        real_page_size,
        local_extend_seq_lens_cpu=local_extend_seq_lens_cpu,
        local_seq_lens_cpu=local_seq_lens_cpu,
    )
    plan = _kpool_plan_to_gpu(
        cpu,
        forward_batch,
        full_real_page_table,
        local_real_page_table,
        local_seqlens_expanded,
        local_req_pool_indices,
        local_max_seq_len,
        pool_size,
        real_page_size,
        topk_transform_method,
    )
    return dataclasses.replace(metadata, kpool_extend_plan=plan)


def init_pooled_paged_mqa_metadata(
    metadata: "NSAMetadata",
    seqlens_32: torch.Tensor,
    forward_mode: "ForwardMode",
    *,
    pool_size: int,
    real_page_size: int,
) -> "NSAMetadata":
    """Build decode-side pooled cache seqlens + deep_gemm schedule.

    Returns the (possibly updated) metadata.
    """
    if (
        pool_size <= 1
        or not is_cuda()
        or not forward_mode.is_decode_or_idle()
        or real_page_size != 64
        or real_page_size % pool_size != 0
    ):
        return metadata

    pooled_cache_seqlens = torch.div(seqlens_32, pool_size, rounding_mode="floor").to(
        torch.int32
    )
    try:
        import deep_gemm

        slots_per_page = real_page_size // pool_size
        pooled_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            pooled_cache_seqlens.unsqueeze(-1),
            slots_per_page,
            deep_gemm.get_num_sms(),
        )
    except (ImportError, ModuleNotFoundError):
        pooled_schedule = None

    return dataclasses.replace(
        metadata,
        pooled_index_kpool=pool_size,
        pooled_cache_seqlens_int32=pooled_cache_seqlens,
        pooled_paged_mqa_schedule_metadata=pooled_schedule,
    )


def update_pooled_paged_mqa_metadata(
    metadata: "NSAMetadata",
    seqlens_32: torch.Tensor,
    forward_mode: "ForwardMode",
    *,
    pool_size: int,
    real_page_size: int,
) -> None:
    """In-place refresh of decode-side pooled metadata (cuda-graph replay).

    Precondition: the pooled buffers were either both allocated (and
    pooled_index_kpool set to pool_size) or both left None during cuda-graph
    capture under the same gating conditions. Replay must observe the same
    gating, otherwise the captured graph's tensor addresses are invalid --
    so we either copy_ into the existing buffers or no-op; we never alloc /
    reset fields here.
    """
    if (
        pool_size <= 1
        or not is_cuda()
        or not forward_mode.is_decode_or_idle()
        or real_page_size != 64
        or real_page_size % pool_size != 0
    ):
        return

    pool_seqlens = torch.div(seqlens_32, pool_size, rounding_mode="floor").to(
        torch.int32
    )
    metadata.pooled_cache_seqlens_int32[: pool_seqlens.shape[0]].copy_(pool_seqlens)

    try:
        import deep_gemm

        new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            pool_seqlens.unsqueeze(-1),
            real_page_size // pool_size,
            deep_gemm.get_num_sms(),
        )
        metadata.pooled_paged_mqa_schedule_metadata.copy_(new_schedule)
    except (ImportError, ModuleNotFoundError):
        # deep_gemm availability at replay must match capture; if it was
        # absent at capture the schedule buffer is None and nothing to do.
        pass
