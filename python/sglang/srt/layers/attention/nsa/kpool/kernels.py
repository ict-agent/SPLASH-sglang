import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128
KPOOL_SCORE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def gather_index_k_scale_prefix_into(
    pool,
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    k_out: torch.Tensor,
    scale_out: torch.Tensor,
) -> None:
    assert buf.dtype == torch.uint8
    assert page_indices.dtype in (torch.int32, torch.int64)
    assert k_out.dtype == torch.uint8
    assert scale_out.dtype == torch.float32
    assert k_out.shape[0] >= seq_len
    assert k_out.shape[1] == INDEX_HEAD_DIM
    assert scale_out.shape[0] >= seq_len
    assert buf.is_contiguous()
    assert page_indices.is_contiguous()
    assert k_out.is_contiguous()
    assert scale_out.is_contiguous()
    if seq_len == 0:
        return

    slots_per_page = pool.slots_per_page
    _gather_index_k_scale_prefix_into_kernel[(seq_len,)](
        buf,
        buf.view(torch.float32),
        page_indices,
        k_out,
        scale_out,
        SLOTS_PER_PAGE=slots_per_page,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE=slots_per_page * INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


@triton.jit
def _gather_index_k_scale_prefix_into_kernel(
    buf_u8_ptr,
    buf_fp32_ptr,
    page_indices_ptr,
    k_out_ptr,
    scale_out_ptr,
    SLOTS_PER_PAGE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    page_idx = token_id // SLOTS_PER_PAGE
    token_offset_in_page = token_id % SLOTS_PER_PAGE
    page = tl.load(page_indices_ptr + page_idx)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    src_k_offsets = page * BUF_NUMEL_PER_PAGE + token_offset_in_page * HEAD_DIM + offs
    dst_k_offsets = token_id * HEAD_DIM + offs
    k = tl.load(buf_u8_ptr + src_k_offsets, mask=mask)
    tl.store(k_out_ptr + dst_k_offsets, k, mask=mask)

    src_s_offset = (
        page * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + token_offset_in_page
    )
    scale = tl.load(buf_fp32_ptr + src_s_offset)
    tl.store(scale_out_ptr + token_id, scale)


def kpool_build_ragged_layout(
    full_page_table: torch.Tensor,
    cu_pages_excl: torch.Tensor,
    ragged_pool_pages: torch.Tensor,
    cu_q_len_excl: torch.Tensor,
    ragged_q_len: torch.Tensor,
    pooled_seq_lens_expanded: torch.Tensor,
    slots_per_page: int,
    total_pool_pages: int,
    total_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the ragged-topk layout build: gather padded page-table rows into
    a flat ``concat_page_table`` and broadcast per-batch K offsets to
    per-q ``q_ks`` / ``q_ke``.

    Replaces ~8 small torch ops (cumsum / arange / repeat_interleave /
    index_select / add) with a single kernel; the prefix sums come from
    the CPU planner via the existing i32 H2D channel so no GPU cumsum is
    needed.
    """
    assert full_page_table.ndim == 2
    assert cu_pages_excl.dtype == torch.int32
    assert ragged_pool_pages.dtype == torch.int32
    assert cu_q_len_excl.dtype == torch.int32
    assert ragged_q_len.dtype == torch.int32
    assert pooled_seq_lens_expanded.dtype == torch.int32

    device = full_page_table.device
    n_rag = cu_pages_excl.shape[0]
    concat_page_table = torch.empty(
        (total_pool_pages,), dtype=full_page_table.dtype, device=device
    )
    q_ks = torch.empty((total_q,), dtype=torch.int32, device=device)
    q_ke = torch.empty((total_q,), dtype=torch.int32, device=device)
    if n_rag == 0:
        return concat_page_table, q_ks, q_ke

    max_pool_pages = full_page_table.shape[1]
    _kpool_build_ragged_layout_kernel[(n_rag,)](
        full_page_table,
        cu_pages_excl,
        ragged_pool_pages,
        cu_q_len_excl,
        ragged_q_len,
        pooled_seq_lens_expanded,
        concat_page_table,
        q_ks,
        q_ke,
        max_pool_pages,
        slots_per_page,
        BLOCK_PAGE=128,
        BLOCK_Q=128,
    )
    return concat_page_table, q_ks, q_ke


@triton.jit
def _kpool_build_ragged_layout_kernel(
    full_page_table_ptr,
    cu_pages_excl_ptr,
    ragged_pool_pages_ptr,
    cu_q_len_excl_ptr,
    ragged_q_len_ptr,
    pooled_seq_lens_ptr,
    concat_page_table_ptr,
    q_ks_ptr,
    q_ke_ptr,
    MAX_POOL_PAGES,
    SLOTS_PER_PAGE: tl.constexpr,
    BLOCK_PAGE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    k = tl.program_id(0)
    page_start = tl.load(cu_pages_excl_ptr + k)
    n_pages = tl.load(ragged_pool_pages_ptr + k)
    q_start = tl.load(cu_q_len_excl_ptr + k)
    q_count = tl.load(ragged_q_len_ptr + k)
    ks_val = page_start * SLOTS_PER_PAGE

    for p_off in tl.range(0, BLOCK_PAGE * tl.cdiv(n_pages, BLOCK_PAGE), BLOCK_PAGE):
        p_offs = p_off + tl.arange(0, BLOCK_PAGE)
        p_mask = p_offs < n_pages
        pages = tl.load(
            full_page_table_ptr + k * MAX_POOL_PAGES + p_offs,
            mask=p_mask,
            other=0,
        )
        tl.store(concat_page_table_ptr + page_start + p_offs, pages, mask=p_mask)

    for q_off in tl.range(0, BLOCK_Q * tl.cdiv(q_count, BLOCK_Q), BLOCK_Q):
        q_offs = q_off + tl.arange(0, BLOCK_Q)
        q_mask = q_offs < q_count
        plen = tl.load(pooled_seq_lens_ptr + q_start + q_offs, mask=q_mask, other=0)
        tl.store(
            q_ks_ptr + q_start + q_offs,
            tl.full([BLOCK_Q], ks_val, tl.int32),
            mask=q_mask,
        )
        tl.store(q_ke_ptr + q_start + q_offs, ks_val + plen, mask=q_mask)


def update_kpool_decode_cuda_graph_page_tables(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table_1: torch.Tensor,
    real_page_table: torch.Tensor,
    max_len: int,
    pool_size: int,
    page_size: int,
) -> None:
    """Update CUDA graph decode page tables for kpool in one pass."""
    assert req_to_token.ndim == 2
    assert req_pool_indices.ndim == 1
    assert page_table_1.ndim == 2
    assert real_page_table.ndim == 2
    assert req_to_token.dtype == torch.int32
    assert req_pool_indices.dtype in (torch.int32, torch.int64)
    assert page_table_1.dtype == torch.int32
    assert real_page_table.dtype == torch.int32
    assert page_size % pool_size == 0
    assert page_table_1.is_contiguous()
    assert real_page_table.is_contiguous()

    bs = req_pool_indices.shape[0]
    if bs == 0 or max_len == 0:
        return

    real_cols = triton.cdiv(max_len, page_size)
    assert page_table_1.shape[0] >= bs and page_table_1.shape[1] >= max_len
    assert real_page_table.shape[0] >= bs and real_page_table.shape[1] >= real_cols

    block_n = 1024
    block_p = block_n // page_size
    grid = (bs, triton.cdiv(max_len, block_n))
    _update_kpool_decode_cuda_graph_page_tables_kernel[grid](
        req_to_token,
        req_pool_indices,
        page_table_1,
        real_page_table,
        req_to_token.stride(0),
        page_table_1.stride(0),
        real_page_table.stride(0),
        max_len,
        real_cols,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
        BLOCK_P=block_p,
    )


@triton.jit
def _update_kpool_decode_cuda_graph_page_tables_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_1_ptr,
    real_page_table_ptr,
    req_to_token_stride_0,
    page_table_1_stride_0,
    real_page_table_stride_0,
    max_len,
    real_cols,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    req = tl.load(req_pool_indices_ptr + row)

    token_cols = block * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_cols < max_len
    token_values = tl.load(
        req_to_token_ptr + req * req_to_token_stride_0 + token_cols,
        mask=token_mask,
        other=0,
    ).to(tl.int32)
    tl.store(
        page_table_1_ptr + row * page_table_1_stride_0 + token_cols,
        token_values,
        mask=token_mask,
    )

    page_cols = block * BLOCK_P + tl.arange(0, BLOCK_P)
    page_token_cols = page_cols * PAGE_SIZE
    page_mask = page_cols < real_cols
    page_values = tl.load(
        req_to_token_ptr + req * req_to_token_stride_0 + page_token_cols,
        mask=page_mask,
        other=0,
    ).to(tl.int32)
    tl.store(
        real_page_table_ptr + row * real_page_table_stride_0 + page_cols,
        page_values // PAGE_SIZE,
        mask=page_mask,
    )


def expand_pooled_groups_to_topk(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected full-pool ids to a strict-width token topk tensor."""
    assert group_ids.ndim == 2
    assert group_valid.shape == group_ids.shape
    assert topk % pool_size == 0
    assert group_ids.shape[1] == topk // pool_size
    assert page_table is None or topk_offsets is None

    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )

    if page_table is not None:
        assert page_table.ndim == 2
        assert page_table.shape[0] == group_ids.shape[0]
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        assert topk_offsets.ndim == 1
        output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = token_ids.to(torch.int32)

    return torch.where(valid, output, torch.full_like(output, -1))


def append_kpool_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append non-pooled tail tokens after selected expanded-history tokens."""
    assert topk_result.dtype == torch.int32
    assert seq_lens.ndim == 1
    assert pool_lens.ndim == 1
    assert seq_lens.shape[0] == topk_result.shape[0]
    assert pool_lens.shape[0] == topk_result.shape[0]

    tail_pool = pool_size - 1
    if tail_pool == 0:
        return topk_result

    rows, n_cols = topk_result.shape
    out_cols = n_cols + tail_pool
    out = torch.empty(
        (rows, out_cols), dtype=topk_result.dtype, device=topk_result.device
    )

    if page_table is None:
        page_table = topk_result
        has_page_table = False
        page_table_cols = 1
    else:
        assert page_table.ndim == 2
        has_page_table = True
        page_table_cols = page_table.shape[1]

    if topk_offsets is None:
        topk_offsets = seq_lens
        has_topk_offsets = False
    else:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        assert topk_offsets.ndim == 1
        has_topk_offsets = True

    block_cols = triton.next_power_of_2(out_cols)
    _append_kpool_tail_to_topk_kernel[(rows,)](
        topk_result,
        seq_lens,
        pool_lens,
        page_table,
        topk_offsets,
        out,
        topk_result.stride(0),
        topk_result.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        out.stride(0),
        out.stride(1),
        N_COLS=n_cols,
        OUT_COLS=out_cols,
        PAGE_TABLE_COLS=page_table_cols,
        POOL_SIZE=pool_size,
        HAS_PAGE_TABLE=has_page_table,
        HAS_TOPK_OFFSETS=has_topk_offsets,
        BLOCK_COLS=block_cols,
    )
    return out


@triton.jit
def _append_kpool_tail_to_topk_kernel(
    topk_ptr,
    seq_lens_ptr,
    pool_lens_ptr,
    page_table_ptr,
    topk_offsets_ptr,
    out_ptr,
    topk_stride_0,
    topk_stride_1,
    page_table_stride_0,
    page_table_stride_1,
    out_stride_0,
    out_stride_1,
    N_COLS: tl.constexpr,
    OUT_COLS: tl.constexpr,
    PAGE_TABLE_COLS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HAS_PAGE_TABLE: tl.constexpr,
    HAS_TOPK_OFFSETS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    mask = cols < OUT_COLS

    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    pool_len = tl.load(pool_lens_ptr + row).to(tl.int32)
    tail_start = pool_len * POOL_SIZE
    history_len = tl.minimum(tail_start, N_COLS)
    tail_count = seq_len % POOL_SIZE

    is_history = cols < history_len
    safe_history_cols = tl.minimum(cols, N_COLS - 1)
    history_value = tl.load(
        topk_ptr + row * topk_stride_0 + safe_history_cols * topk_stride_1,
        mask=mask & is_history,
        other=-1,
    )

    tail_offset = cols - history_len
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    tail_raw = tail_start + tail_offset
    tail_value = tail_raw
    if HAS_PAGE_TABLE:
        safe_tail = tl.minimum(tl.maximum(tail_raw, 0), PAGE_TABLE_COLS - 1)
        tail_value = tl.load(
            page_table_ptr
            + row * page_table_stride_0
            + safe_tail * page_table_stride_1,
            mask=mask & is_tail,
            other=-1,
        ).to(tl.int32)
    if HAS_TOPK_OFFSETS:
        offset = tl.load(topk_offsets_ptr + row).to(tl.int32)
        tail_value = tail_raw + offset

    value = tl.where(is_history, history_value, -1)
    value = tl.where(is_tail, tail_value, value)
    tl.store(out_ptr + row * out_stride_0 + cols * out_stride_1, value, mask=mask)


def topk_from_pooled_history_logits(
    logits: torch.Tensor,
    group_lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    out_rows: int | None = None,
) -> torch.Tensor:
    """Select full-pool groups, expand to tokens, and optionally append tail.

    ``row_starts`` lets the caller pass a single flat ``[total_q, K]``
    logits matrix where each row's valid scores live at columns
    ``[row_starts[i], row_starts[i] + group_lengths[i])`` instead of at
    column 0. Used by the kpool ragged-extend path to do one fused
    topk over concatenated batches.

    ``out_rows`` lets the caller request more output rows than
    ``logits.shape[0]``; the trailing rows are filled with -1. On the
    fused (group_topk in 128..512) path this happens in-kernel; on the
    slow ``fast_topk_v2`` path it falls back to a host-side ``torch.full``
    pad. Used when q is right-padded for mlp-sync (TP/CP) and the topk
    output should be sized against the padded q row count.
    """
    assert logits.ndim == 2
    assert group_lengths.ndim == 1
    assert logits.shape[0] == group_lengths.shape[0]
    assert topk > 0
    assert topk % pool_size == 0

    _, cols = logits.shape
    group_topk = topk // pool_size
    if topk_offsets is not None and topk_offsets.ndim == 2:
        assert topk_offsets.shape[1] == 1
        topk_offsets = topk_offsets.squeeze(1)

    if group_topk not in (128, 160, 192, 224, 256, 512, 2048):
        raise NotImplementedError(
            "index_kpool topk only supports pooled group_topk in "
            f"(128, 160, 192, 224, 256, 512, 2048), got {group_topk} "
            f"(topk={topk}, pool_size={pool_size})."
        )
    if not logits.is_cuda or logits.dtype != torch.float32:
        raise NotImplementedError(
            "index_kpool topk requires CUDA float32 logits; PyTorch topk fallback "
            f"is disabled. Got device={logits.device}, dtype={logits.dtype}."
        )

    if group_topk in (128, 160, 192, 224, 256, 512):
        from sgl_kernel import fast_kpool_topk_transform_fused

        return fast_kpool_topk_transform_fused(
            score=logits,
            lengths=group_lengths.to(torch.int32),
            pool_size=pool_size,
            topk=topk,
            page_table=page_table,
            topk_indices_offset=topk_offsets,
            row_starts=row_starts,
            seq_lens=seq_lens.to(torch.int32) if seq_lens is not None else None,
            out_rows=out_rows,
        )

    # Slow path: ``fast_topk_v2`` + manual expand. The fused fast_kpool
    # kernel handles ``out_rows`` in-kernel; here we replicate it with a
    # host-side -1 pad so callers see the same contract regardless of
    # which path runs.
    from sgl_kernel import fast_topk_v2

    selected_groups = fast_topk_v2(
        logits,
        group_lengths.to(torch.int32),
        group_topk,
        row_starts=row_starts,
    )

    rank = torch.arange(group_topk, device=logits.device, dtype=torch.int32)
    max_valid_groups = min(cols, group_topk)
    valid_counts = torch.minimum(
        group_lengths.to(torch.int32),
        torch.full_like(group_lengths.to(torch.int32), max_valid_groups),
    )
    group_valid = rank.unsqueeze(0) < valid_counts.unsqueeze(1)
    expanded = expand_pooled_groups_to_topk(
        selected_groups.contiguous(),
        group_valid,
        topk=topk,
        pool_size=pool_size,
        page_table=page_table,
        topk_offsets=topk_offsets,
    )
    if seq_lens is None:
        result = expanded
    else:
        result = append_kpool_tail_to_topk(
            expanded,
            seq_lens=seq_lens,
            pool_lens=group_lengths,
            pool_size=pool_size,
            page_table=page_table,
            topk_offsets=topk_offsets,
        )

    if out_rows is None or out_rows == result.shape[0]:
        return result
    assert (
        out_rows >= result.shape[0]
    ), f"out_rows ({out_rows}) must be >= topk rows ({result.shape[0]})"
    padded = torch.full(
        (out_rows, result.shape[1]),
        -1,
        dtype=result.dtype,
        device=result.device,
    )
    padded[: result.shape[0]] = result
    return padded


def kpool_assemble_softmax_rotate_write_cache(
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = False,
) -> None:
    """Fused gather + softmax + Hadamard rotate + fp8 quant + cache write.

    For each output pool row r, slots are gathered from tail or chunk
    based on n_from_tail[r] (see _kpool_assemble_softmax_rotate_write_cache_kernel).
    Avoids materializing the (n_pools, pool_size, head_dim) intermediate
    """
    assert chunk_k.dim() == 2 and chunk_k.shape[1] == INDEX_HEAD_DIM
    assert chunk_score.shape == chunk_k.shape
    assert tail_k.dim() == 3 and tail_k.shape[2] == INDEX_HEAD_DIM
    assert tail_score.shape == tail_k.shape
    assert chunk_k.dtype == torch.bfloat16
    assert chunk_score.dtype in KPOOL_SCORE_DTYPES
    assert tail_k.dtype == torch.bfloat16
    assert tail_score.dtype in KPOOL_SCORE_DTYPES
    pool_size = tail_k.shape[1]
    n_pools = req_pool_idx.shape[0]
    assert n_from_tail.shape == (n_pools,)
    assert chunk_src_start.shape == (n_pools,)
    assert ape.shape == (pool_size, INDEX_HEAD_DIM)
    assert ape.dtype == torch.float32
    assert buf.dtype == torch.uint8
    assert pool.index_head_dim == INDEX_HEAD_DIM
    assert loc.dtype == torch.int64
    assert loc.shape == (n_pools,)

    if n_pools == 0:
        return

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=chunk_k.device)
        has_write_mask = False
    else:
        assert write_mask.shape == (n_pools,)
        write_mask = write_mask.contiguous()
        has_write_mask = True

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)

    _kpool_assemble_softmax_rotate_write_cache_kernel[(n_pools,)](
        buf_fp8,
        buf_fp32,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        ape,
        loc,
        write_mask,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        ape.stride(0),
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=pool_size,
        HEAD_DIM=INDEX_HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE=pool.slots_per_page * pool.index_head_dim,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        SLOTS_PER_PAGE=pool.slots_per_page,
    )


def kpool_decode_update_and_maybe_write_cache(
    pool,
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    round_scale: bool = False,
) -> None:
    assert tail_k.ndim == 3
    assert tail_score.shape == tail_k.shape
    assert tail_k.shape[1] == ape.shape[0]
    assert tail_k.shape[2] == INDEX_HEAD_DIM
    assert key.ndim == 2 and key.shape[1] == INDEX_HEAD_DIM
    assert slot_score.shape == key.shape
    assert ape.shape == tail_k.shape[1:]
    assert tail_k.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    assert tail_score.dtype in KPOOL_SCORE_DTYPES
    assert slot_score.dtype == tail_score.dtype
    assert ape.dtype == torch.float32
    assert buf.dtype == torch.uint8
    assert pool.index_head_dim == INDEX_HEAD_DIM
    assert tail_k.is_contiguous()
    assert tail_score.is_contiguous()

    batch = key.shape[0]
    if batch == 0:
        return

    key = key.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    positions = positions.contiguous()
    seq_lens = seq_lens.contiguous()
    out_cache_loc = out_cache_loc.contiguous()

    assert req_pool_indices.shape[0] >= batch
    assert positions.shape[0] >= batch
    assert seq_lens.shape[0] >= batch
    assert out_cache_loc.shape[0] >= batch
    assert block_tables.ndim == 2
    assert block_tables.shape[0] >= batch

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)
    _kpool_decode_update_and_maybe_write_cache_kernel[(batch,)](
        buf_fp8,
        buf_fp32,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        key.stride(0),
        slot_score.stride(0),
        ape.stride(0),
        block_tables.stride(0),
        block_tables.stride(1),
        TAIL_SIZE=tail_k.shape[0],
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=tail_k.shape[1],
        HEAD_DIM=tail_k.shape[2],
        S_OFFSET_NBYTES_IN_PAGE=pool.slots_per_page * pool.index_head_dim,
        ROUND_SCALE=round_scale,
        BLOCK_D=triton.next_power_of_2(tail_k.shape[2]),
        SLOTS_PER_PAGE=pool.slots_per_page,
    )


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _hadamard_quantize_fp8(acc, denom, do_write, ROUND_SCALE: tl.constexpr):
    """Normalize -> bf16 round-trip -> Hadamard rotate -> bf16 round-trip
    -> fp8-e4m3 quantize. Returns (quantized, scale)."""
    x = acc / denom
    x = tl.where(do_write, x, 0.0).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    fp8_max_inv = 1.0 / 448.0
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv

    quantized = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    return quantized, scale


@triton.jit
def _kpool_assemble_softmax_rotate_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    n_from_tail_ptr,
    chunk_src_start_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr = 64,
):
    """Per-pool program. Slot src dispatch:
    if slot < n_from_tail[row]:
        src = tail[req_pool_idx[row], slot]
    else:
        chunk_off = chunk_src_start[row] + (slot - n_from_tail[row])
        src = chunk[chunk_off]
    """
    row = tl.program_id(0)
    do_write = True
    if HAS_WRITE_MASK:
        do_write = tl.load(write_mask_ptr + row)

    offs = tl.arange(0, BLOCK_D)
    mask = (offs < HEAD_DIM) & do_write

    n_tail = tl.load(n_from_tail_ptr + row)
    req = tl.load(req_pool_idx_ptr + row)
    chunk_src = tl.load(chunk_src_start_ptr + row)

    # Online softmax over POOL_SIZE slots; (max, denom, acc) rescaled
    # in place to halve the load traffic vs a two-pass version.
    #
    # `from_tail = slot < n_tail` is a scalar runtime bool (n_tail is a
    # per-row scalar). Using a real `if/else` here -- instead of the
    # previous `tl.where(from_tail, load_tail, load_chunk)` -- means
    # only the active branch's loads are emitted. The dead pointer
    # arithmetic (e.g. ``tail_ptr + chunk_off``) is never computed, so
    # we cannot read into unmapped memory even when max_running_requests
    # is small.
    m = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        if slot < n_tail:
            off = req * tail_stride_0 + slot * tail_stride_1 + offs
            score = tl.load(tail_score_ptr + off, mask=mask, other=0.0).to(tl.float32)
            k = tl.load(tail_k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        else:
            off = (chunk_src + (slot - n_tail)) * chunk_stride_0 + offs
            score = tl.load(chunk_score_ptr + off, mask=mask, other=0.0).to(tl.float32)
            k = tl.load(chunk_k_ptr + off, mask=mask, other=0.0).to(tl.float32)

        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )

        new_m = tl.maximum(m, score)
        rescale = tl.exp(m - new_m)
        prob = tl.exp(score - new_m)
        denom = denom * rescale + prob
        acc = acc * rescale + k * prob
        m = new_m

    quantized, scale = _hadamard_quantize_fp8(acc, denom, do_write, ROUND_SCALE)

    loc = tl.load(loc_ptr + row, mask=do_write, other=0)
    loc_page_index = loc // SLOTS_PER_PAGE
    loc_token_offset_in_page = loc % SLOTS_PER_PAGE
    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE + loc_token_offset_in_page * HEAD_DIM + offs
    )
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=mask)
    tl.store(buf_fp32_ptr + out_s_offset, scale, mask=do_write)


@triton.jit
def _kpool_decode_update_and_maybe_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_k_ptr,
    tail_score_ptr,
    key_ptr,
    slot_score_ptr,
    ape_ptr,
    block_tables_ptr,
    req_pool_indices_ptr,
    positions_ptr,
    seq_lens_ptr,
    out_cache_loc_ptr,
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    key_stride_0,
    slot_score_stride_0,
    ape_stride_0,
    block_tables_stride_0,
    block_tables_stride_1,
    TAIL_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr = 64,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    req_raw = tl.load(req_pool_indices_ptr + row)
    req_valid = (req_raw >= 0) & (req_raw < TAIL_SIZE)
    req = tl.minimum(tl.maximum(req_raw, 0), TAIL_SIZE - 1)

    pos = tl.load(positions_ptr + row)
    safe_pos = tl.maximum(pos, 0)
    seq_len = tl.load(seq_lens_ptr + row)
    cache_loc = tl.load(out_cache_loc_ptr + row)
    pos_valid = req_valid & (cache_loc != 0) & (pos >= 0) & (pos < seq_len)

    slot = safe_pos % POOL_SIZE

    # key/score_current are needed for the tail store at the end on every
    # pos_valid row, so load them outside the do_write branch.
    key = tl.load(
        key_ptr + row * key_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    score_current = tl.load(
        slot_score_ptr + row * slot_score_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    # Only the last slot of each pool finalizes a compressed entry --
    # 1/POOL_SIZE of programs do real work, the rest just defer-update tail.
    do_write = pos_valid & (slot == POOL_SIZE - 1)

    if do_write:
        # Online softmax over POOL_SIZE slots. The current slot reads from
        # key/score_current (new token); other slots read tail_k/tail_score,
        # which still hold pre-update values because the tail store is
        # deferred to the end of the kernel (avoids RAW).
        m = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        acc = tl.full((BLOCK_D,), 0.0, tl.float32)
        denom = tl.full((BLOCK_D,), 0.0, tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + pool_slot * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            k_buf = tl.load(
                tail_k_ptr + req * tail_k_stride_0 + pool_slot * tail_k_stride_1 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            k = tl.where(is_current, key, k_buf)
            new_m = tl.maximum(m, score)
            rescale = tl.exp(m - new_m)
            prob = tl.exp(score - new_m)
            denom = denom * rescale + prob
            acc = acc * rescale + k * prob
            m = new_m

        quantized, scale = _hadamard_quantize_fp8(acc, denom, True, ROUND_SCALE)

        pool_id = safe_pos // POOL_SIZE
        pool_page_group = pool_id // SLOTS_PER_PAGE
        # 16 (anchor: pool=16, slots=64) or 1 (dense: pool=16, slots=4).
        token_pages_per_pool_group: tl.constexpr = (
            POOL_SIZE * SLOTS_PER_PAGE // PAGE_SIZE
        )
        token_page_row = pool_page_group * token_pages_per_pool_group
        packed_page = tl.load(
            block_tables_ptr
            + row * block_tables_stride_0
            + token_page_row * block_tables_stride_1,
        )
        loc_page_index = packed_page.to(tl.int64)
        loc_token_offset_in_page = pool_id % SLOTS_PER_PAGE
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + loc_token_offset_in_page * HEAD_DIM
            + offs
        )
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4
            + S_OFFSET_NBYTES_IN_PAGE // 4
            + loc_token_offset_in_page
        )

        tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=dim_mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale)

    # Deferred tail store (see online-softmax comment above for RAW).
    tail_k_offset = req * tail_k_stride_0 + slot * tail_k_stride_1 + offs
    tail_score_offset = req * tail_score_stride_0 + slot * tail_score_stride_1 + offs
    update_mask = dim_mask & pos_valid
    tl.store(tail_k_ptr + tail_k_offset, key, mask=update_mask)
    tl.store(tail_score_ptr + tail_score_offset, score_current, mask=update_mask)


def scatter_kpool_tail_updates(
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_offset: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    """Per-batch tail buffer writes folded into one kernel.

    For each batch row r:
        tail[req_pool_idx[r], dst_offset[r] : dst_offset[r] + n_write[r]] <-
            chunk[chunk_src_start[r] : chunk_src_start[r] + n_write[r]]
    """
    assert chunk_k.dim() == 2 and chunk_k.shape[1] == INDEX_HEAD_DIM
    assert chunk_score.shape == chunk_k.shape
    pool_size = tail_k.shape[1]
    n_rows = req_pool_idx.shape[0]
    if n_rows == 0:
        return

    _scatter_kpool_tail_updates_kernel[(n_rows, pool_size)](
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_offset,
        chunk_src_start,
        n_write,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        POOL_SIZE=pool_size,
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


@triton.jit
def _scatter_kpool_tail_updates_kernel(
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    dst_offset_ptr,
    chunk_src_start_ptr,
    n_write_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)

    n_w = tl.load(n_write_ptr + row)
    if slot >= n_w:
        return

    req = tl.load(req_pool_idx_ptr + row)
    dst_off = tl.load(dst_offset_ptr + row)
    src_off = tl.load(chunk_src_start_ptr + row) + slot

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    k = tl.load(chunk_k_ptr + src_off * chunk_stride_0 + offs, mask=mask)
    s = tl.load(chunk_score_ptr + src_off * chunk_stride_0 + offs, mask=mask)

    dst = req * tail_stride_0 + (dst_off + slot) * tail_stride_1 + offs
    tl.store(tail_k_ptr + dst, k, mask=mask)
    tl.store(tail_score_ptr + dst, s, mask=mask)


@triton.jit
def _pack_pool_slots_to_payload_kernel(
    buf_ptr,  # uint8 [num_pages, page_bytes]
    locs_ptr,  # int64 [N]
    payload_ptr,  # uint8 [N, payload_bytes]
    payload_bytes: tl.constexpr,
    slots_per_page: tl.constexpr,
    head_dim: tl.constexpr,
    page_bytes: tl.constexpr,  # slots_per_page * head_dim + slots_per_page * 4
    scale_region_off: tl.constexpr,  # slots_per_page * head_dim
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    loc = tl.load(locs_ptr + row).to(tl.int64)
    page = loc // slots_per_page
    slot = loc % slots_per_page
    page_base = page * page_bytes

    # Copy head_dim fp8 bytes.
    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    src = page_base + slot * head_dim + offs
    val = tl.load(buf_ptr + src, mask=mask, other=0).to(tl.uint8)
    tl.store(payload_ptr + row * payload_bytes + offs, val, mask=mask)

    # Copy 4 fp32-scale bytes (positions head_dim .. head_dim+4).
    s_offs = tl.arange(0, 4)
    s_src = page_base + scale_region_off + slot * 4 + s_offs
    s_val = tl.load(buf_ptr + s_src).to(tl.uint8)
    tl.store(payload_ptr + row * payload_bytes + head_dim + s_offs, s_val)


@triton.jit
def _select_and_scatter_pool_slots_kernel(
    recv_ptr,  # uint8 [cp_size, N, payload_bytes]
    owner_ptr,  # int32 [N]
    locs_ptr,  # int64 [N]
    buf_ptr,  # uint8 [num_pages, page_bytes]
    cp_rank: tl.constexpr,
    payload_bytes: tl.constexpr,
    slots_per_page: tl.constexpr,
    head_dim: tl.constexpr,
    page_bytes: tl.constexpr,
    scale_region_off: tl.constexpr,
    n_total: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    owner = tl.load(owner_ptr + row).to(tl.int64)
    # Self-owned rows were already written by this rank's compress kernel
    # (the local writes that fed pack_pool_slots). Skipping them avoids a
    # redundant recv -> buf copy of bytes we already have correct in buf.
    if owner == cp_rank:
        return
    loc = tl.load(locs_ptr + row).to(tl.int64)
    page = loc // slots_per_page
    slot = loc % slots_per_page
    page_base = page * page_bytes

    recv_row_base = (owner * n_total + row) * payload_bytes

    # Scatter head_dim fp8 bytes back into the fp8 region.
    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    val = tl.load(recv_ptr + recv_row_base + offs, mask=mask, other=0).to(tl.uint8)
    dst = page_base + slot * head_dim + offs
    tl.store(buf_ptr + dst, val, mask=mask)

    # Scatter 4 fp32-scale bytes back into the scale region.
    s_offs = tl.arange(0, 4)
    s_val = tl.load(recv_ptr + recv_row_base + head_dim + s_offs).to(tl.uint8)
    s_dst = page_base + scale_region_off + slot * 4 + s_offs
    tl.store(buf_ptr + s_dst, s_val)


def all_gather_and_scatter_pool_slots(
    buf: torch.Tensor,
    local_locs: torch.Tensor,
    owner_rank: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    slots_per_page: int,
) -> None:
    """Replicate this layer's freshly written pool slots across all CP ranks.

    Each rank wrote a subset of N total pool slots (FP8 key + fp32 scale)
    at physical locations ``local_locs[owner_rank == cp_rank]`` into its
    own copy of ``buf``. To make every rank see the same post-write cache
    state, we pack each slot's (fp8 key + fp32 scale) into a contiguous
    payload, all-gather across ranks, then scatter the owner's payload
    for each row back into buf.

    Layout of ``buf`` (per page, see NSATokenToKVPool init):
        buf[p, 0 : slots_per_page * head_dim]                  uint8 fp8 keys
        buf[p, slots_per_page * head_dim : ].view(fp32)        fp32 scales

    Two triton kernels (pack and select-scatter) replace four advanced-
    indexing launches and avoid the (cp_size, N, payload) intermediate
    contiguous gather buffer copy.

    buf:               (num_pages, slots_per_page * head_dim + slots_per_page * 4) uint8
    local_locs:        (N,) int64. Physical slot locations for all pool
                       rows in the plan (identical on every rank).
    owner_rank:        (N,) int32. Which CP rank owns each row (identical
                       on every rank). Derived from a row-index-mod-cp_size
                       scheme in ``_init_kpool_extend_metadata``.
    cp_size:           Number of CP ranks.
    """
    from sglang.srt.layers.dp_attention import attn_cp_all_gather_into_tensor

    assert buf.dtype == torch.uint8
    assert buf.is_contiguous()
    assert local_locs.dtype == torch.int64
    assert owner_rank.dtype == torch.int32
    assert local_locs.shape == owner_rank.shape

    n_total = local_locs.shape[0]
    if n_total == 0 or cp_size <= 1:
        return

    head_dim = INDEX_HEAD_DIM
    payload_bytes = head_dim + 4
    scale_region_off = slots_per_page * head_dim
    page_bytes = buf.shape[1]
    device = buf.device

    send_payload = torch.empty(
        (n_total, payload_bytes), dtype=torch.uint8, device=device
    )
    _pack_pool_slots_to_payload_kernel[(n_total,)](
        buf,
        local_locs,
        send_payload,
        payload_bytes=payload_bytes,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
        page_bytes=page_bytes,
        scale_region_off=scale_region_off,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )

    recv = torch.empty(
        (cp_size, n_total, payload_bytes), dtype=torch.uint8, device=device
    )
    attn_cp_all_gather_into_tensor(
        recv.view(cp_size * n_total, payload_bytes), send_payload
    )

    _select_and_scatter_pool_slots_kernel[(n_total,)](
        recv,
        owner_rank,
        local_locs,
        buf,
        cp_rank=cp_rank,
        payload_bytes=payload_bytes,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
        page_bytes=page_bytes,
        scale_region_off=scale_region_off,
        n_total=n_total,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
