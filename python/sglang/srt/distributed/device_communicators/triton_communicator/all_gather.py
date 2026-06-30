# Adapted from https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py

"""NVIDIA multimem all-gather (bf16) over symmetric memory.

Provides the plain ``all_gather`` collective plus ``all_gather_rerange`` -- a
fused all-gather + rerange + concat used by context-parallel attention.
"""

from typing import List

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.triton_communicator.comm_state import (
    _MULTIMEM_BLOCK_THREADS,
    _MULTIMEM_MIN_BLOCKS,
    _WARP_SIZE,
    TritonMultimemState,
    fits_comm_buffer,
    get_even_token_distribution,
    get_launch_config,
    get_symm_mem_handle,
    get_token_partition,
    is_even_token_distribution,
    mark_workspace_used,
    view_comm_buffer,
    wait_for_workspace,
)
from sglang.srt.distributed.device_communicators.triton_communicator.triton_barrier import (
    blockwise_barrier,
)
from sglang.srt.distributed.device_communicators.triton_communicator.triton_utils import (
    get_flat_tid,
    local_ld_128,
    multimem_st_128,
    sync_threads,
)


@triton.jit
def all_gather_kernel(
    input_ptr,
    multicast_ptr,
    signal_pad_ptr,
    numel,
    in_offset,
    out_offset,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
) -> None:
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="relaxed")
    sync_threads()

    numel = numel // NUMEL_PER_THREAD
    pid = tl.program_id(axis=0)
    tid = get_flat_tid()
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        thread_offset = block_start + tid
        mask = thread_offset < numel
        # The load is local, so input_ptr can be any tensor (e.g. the caller's
        # shard read in place) -- it need not live in symmetric memory. in_offset
        # / out_offset are decoupled so the source is read from its own start
        # while the multicast store lands in this rank's slot of the buffer.
        in_ptr = (
            input_ptr.to(tl.pointer_type(tl.uint64))
            + (in_offset // NUMEL_PER_THREAD + thread_offset) * 2
        )
        out_ptr = (
            multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64))
            + (out_offset // NUMEL_PER_THREAD + thread_offset) * 2
        )
        x, y, z, w = local_ld_128(in_ptr, mask)
        multimem_st_128(out_ptr, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="acq_rel")


def all_gather(
    state: TritonMultimemState,
    hidden_states: torch.Tensor,
    tp_num_tokens: int = None,
    token_list_in_group: List[int] = None,
    safe: bool = True,
) -> torch.Tensor:
    """Multimem all-gather of ``hidden_states`` over the comm group.

    ``hidden_states`` is this rank's ``[local_num_tokens, hidden_dim]`` bf16
    shard; the output is the full ``[total_num_tokens, hidden_dim]`` tensor on
    every rank. Either ``tp_num_tokens`` (even split) or an explicit
    ``token_list_in_group`` must be provided. With ``safe=False`` the result
    aliases the comm buffer and is only valid until the next collective.

    The shard is read in place by the kernel (the all-gather load is local), so
    no copy into the comm buffer is needed; ``hidden_states`` must therefore be
    contiguous and 16-byte aligned.

    A hidden dim is taken from ``hidden_states``; returns ``None`` when the
    gathered ``total_num_tokens * hidden`` does not fit ``state.max_numel`` so
    the caller can fall back to NCCL.
    """
    assert (
        tp_num_tokens is not None or token_list_in_group is not None
    ), "Either tp_num_tokens or token_list_in_group must be provided"
    if token_list_in_group is None:
        token_list_in_group = get_even_token_distribution(state, tp_num_tokens)
    assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported for now"
    assert hidden_states.dim() == 2, "hidden_states must be 2-D"
    total_num_tokens, local_num_tokens, local_token_offset = get_token_partition(
        state, token_list_in_group
    )
    hidden = hidden_states.shape[-1]
    assert (
        hidden_states.shape[0] == local_num_tokens
    ), f"{hidden_states.shape=}|{local_num_tokens=}|{hidden_states.device=} Mismatched shape"
    # The kernel reads hidden_states in place with 128-bit local loads instead
    # of copying it into the comm buffer first, so it must be contiguous and
    # 16-byte aligned. is_contiguous() alone does not imply 16-byte data_ptr
    # alignment (a contiguous slice of a wider tensor can start at a 2-byte
    # offset), so check both.
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert hidden_states.data_ptr() % 16 == 0, (
        f"hidden_states.data_ptr()={hex(hidden_states.data_ptr())} must be "
        f"16-byte aligned for 128-bit loads; copy through a fresh allocation "
        f"if needed"
    )

    if not fits_comm_buffer(state, total_num_tokens, hidden):
        return None
    # View the flat comm buffer as [total, hidden] for this call.
    comm_buffer = view_comm_buffer(state, total_num_tokens, hidden)

    # Uniform split: every rank contributes the same row count, so the gather is
    # plain rank-major -- hand it to torch's maintained multimem all_gather
    # (writes the gathered result straight into the symm-mem comm buffer). The
    # triton kernel below is kept only for the non-uniform token_list case, which
    # torch's op does not support.
    if is_even_token_distribution(token_list_in_group):
        wait_for_workspace(state)
        torch.ops.symm_mem.multimem_all_gather_out(
            hidden_states, state.group.group_name, comm_buffer
        )
        output = comm_buffer.clone() if safe else comm_buffer
        mark_workspace_used(state)
        return output

    # No copy_ into comm_buffer: the all-gather load is local, so the kernel reads
    # this rank's shard straight from hidden_states and multicast-stores it into
    # every rank's comm_buffer slot.
    num_elts = local_num_tokens * hidden
    assert (local_token_offset * hidden) % 8 == 0, (
        "all_gather output offset must be 16-byte aligned, got "
        f"{local_token_offset=} {hidden=}"
    )
    num_blocks, block_size, num_warps, numel_per_thread = get_launch_config(num_elts)
    wait_for_workspace(state)
    symm_mem_handle = get_symm_mem_handle(state)
    all_gather_kernel[(num_blocks, 1, 1)](
        input_ptr=hidden_states,
        multicast_ptr=symm_mem_handle.multicast_ptr,
        signal_pad_ptr=symm_mem_handle.signal_pad_ptrs_dev,
        numel=num_elts,
        in_offset=0,
        out_offset=local_token_offset * hidden,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_handle.rank,
        WORLD_SIZE=symm_mem_handle.world_size,
        num_warps=num_warps,
    )
    output = comm_buffer.clone() if safe else comm_buffer
    mark_workspace_used(state)
    return output


@triton.jit
def all_gather_rerange_kernel(
    in0_ptr,
    in1_ptr,
    multicast_ptr,
    signal_pad_ptr,
    num_vecs,
    n_a,
    base_a,
    stride_a,
    base_b,
    stride_b,
    IN0_VECS: tl.constexpr,
    IN0_ROW_STRIDE_VECS: tl.constexpr,
    IN1_ROW_STRIDE_VECS: tl.constexpr,
    OUT_VECS: tl.constexpr,
    HAS_IN1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
) -> None:
    """Fused all-gather + rerange + concat, in one multimem collective.

    Reads this rank's local rows straight from the source tensors (no copy into
    comm_buffer, and no host-side ``cat`` of the sources) and multicast-stores each
    128-bit vector to its *natural-sequential* destination row. After the
    collective every rank's comm_buffer already holds the gathered, concatenated
    tensor in natural order, so the input ``cat``, the ``copy_`` and the rerange
    permute all disappear.

    A vec is one 128-bit transfer unit = 8 bf16 elements.

    Concat: each output token is ``OUT_VECS`` vectors; the first
    ``IN0_VECS`` come from ``in0_ptr`` (e.g. key) and the rest from ``in1_ptr``
    (e.g. gate_score). Source boundaries are vec-aligned (both widths are
    multiples of 8 bf16), so no vec straddles the seam. A single source is
    expressed as IN0_VECS == OUT_VECS (in1 unused).

    Strided sources: ``IN0_ROW_STRIDE_VECS`` / ``IN1_ROW_STRIDE_VECS`` are the per-row
    strides (in 128-bit vectors) of in0 / in1. For a contiguous source they equal
    its own width in vecs; for a slice view of a wider tensor (e.g.
    ``latent_cache = qkv[..., q_lora:]``) they are the FULL row width, so the
    kernel reads the non-contiguous view in place -- no host ``.contiguous()``
    copy. Any column offset is already folded into the tensor's data_ptr.

    Rerange: the destination row is piecewise-affine in (at most) two segments --
    all both CP layouts need, so no per-token index table is materialized on the
    host:
      token i < n_a:  dst_row = base_a + i        * stride_a
      token i >= n_a: dst_row = base_b + (i-n_a)  * stride_b
    round-robin: one segment, base_a=cp_rank, stride_a=cp_size, n_a=local_n.
    in-seq-split: two zigzag segments, both stride 1, bases = natural offsets.
    """

    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="relaxed")
    sync_threads()

    pid = tl.program_id(axis=0)
    tid = get_flat_tid()
    block_start = pid * BLOCK_SIZE

    while block_start < num_vecs:
        thread_offset = block_start + tid
        mask = thread_offset < num_vecs

        local_tok = thread_offset // OUT_VECS
        out_vec_idx = thread_offset % OUT_VECS
        in_seg_a = local_tok < n_a
        dst = tl.where(
            in_seg_a,
            base_a + local_tok * stride_a,
            base_b + (local_tok - n_a) * stride_b,
        )

        if HAS_IN1:
            # pick source: first IN0_VECS vecs -> in0, remainder -> in1.
            # Each source addressed by its own per-row stride (handles slice
            # views of wider tensors without a host-side contiguous copy).
            use0 = out_vec_idx < IN0_VECS
            use1 = out_vec_idx >= IN0_VECS
            # Clamp each per-source vec index so the *masked-off* load never
            # addresses out of bounds (its values are discarded by the where).
            l0 = tl.where(use0, out_vec_idx, 0)
            l1 = tl.where(use1, out_vec_idx - IN0_VECS, 0)
            w0 = local_tok * IN0_ROW_STRIDE_VECS + l0
            w1 = local_tok * IN1_ROW_STRIDE_VECS + l1

            p0 = in0_ptr.to(tl.pointer_type(tl.uint64)) + w0 * 2
            p1 = in1_ptr.to(tl.pointer_type(tl.uint64)) + w1 * 2
            x0, y0, z0, t0 = local_ld_128(p0, mask & use0)
            x1, y1, z1, t1 = local_ld_128(p1, mask & use1)
            x = tl.where(use0, x0, x1)
            y = tl.where(use0, y0, y1)
            z = tl.where(use0, z0, z1)
            t = tl.where(use0, t0, t1)
        else:
            w0 = local_tok * IN0_ROW_STRIDE_VECS + out_vec_idx
            p0 = in0_ptr.to(tl.pointer_type(tl.uint64)) + w0 * 2
            x, y, z, t = local_ld_128(p0, mask)

        out_vec = dst * OUT_VECS + out_vec_idx
        out_ptr = multicast_ptr.to(tl.pointer_type(tl.uint64)) + out_vec * 2
        multimem_st_128(out_ptr, x, y, z, t, mask)

        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="acq_rel")


def all_gather_rerange_supported(sources: List[torch.Tensor]) -> bool:
    """Whether ``sources`` satisfy ``all_gather_rerange``'s kernel constraints,
    as a non-raising predicate so callers can fall back to NCCL on a miss
    instead of tripping the asserts inside ``all_gather_rerange``:

      - 1 or 2 sources (the kernel has fixed in0/in1 operands);
      - each bf16, 2-D, sharing the first dim (same local token count);
      - inner dim contiguous, width a multiple of 8 bf16 (16B vec boundary);
      - row stride a multiple of 8 (strided slice views ok), data_ptr 16B-aligned.

    Does NOT check comm-buffer capacity -- that depends on the gathered
    ``total * width`` the caller derives; use ``fits_comm_buffer`` for it.
    """
    if not 1 <= len(sources) <= 2:
        return False
    n_local = sources[0].shape[0]
    return all(
        s.dtype == torch.bfloat16
        and s.dim() == 2
        and s.shape[0] == n_local
        and s.shape[1] > 0
        and s.shape[1] % 8 == 0
        and s.stride(1) == 1
        and s.stride(0) % 8 == 0
        and s.data_ptr() % 16 == 0
        for s in sources
    )


def all_gather_rerange(
    state: TritonMultimemState,
    sources: List[torch.Tensor],
    total_num_tokens: int,
    n_a: int,
    base_a: int,
    stride_a: int,
    base_b: int = 0,
    stride_b: int = 1,
    safe: bool = True,
) -> torch.Tensor:
    """Fused all-gather + rerange + concat in a single multimem collective.

    Versus ``all_gather`` + a host-side ``cat`` + permute this removes ALL of:
      * the input ``torch.cat(sources)`` (the kernel reads each source in place
        and writes them concatenated per output token),
      * the ``comm_buffer[slot].copy_`` (sources are the kernel's local operands
        directly), and
      * the rerange permute (each token is multicast-stored to its
        natural-sequential destination row).

    ``sources`` is 1 or 2 same-N bf16 tensors (e.g. [key, gate_score]); each
    width must be 16B-aligned (multiple of 8 bf16) so the concat seam lands on a
    128-bit vector boundary. They are laid out per output token as ``[src0|src1]``.

    Sources need NOT be fully contiguous: a row-major slice view of a wider
    tensor (inner dim contiguous, row stride a multiple of 8 bf16) is read in
    place via its per-row stride -- so callers like ``rebuild_cp_kv_cache``
    (where ``latent_cache`` is ``qkv[..., q_lora:]``) avoid a ``.contiguous()``
    copy. Any column offset is already encoded in the view's data_ptr.

    The destination row is piecewise-affine in two segments -- enough for both CP
    layouts, so NO per-token index tensor is built on the host:
      token i < n_a:  dst = base_a + i        * stride_a
      token i >= n_a: dst = base_b + (i-n_a)  * stride_b
    round-robin: single segment -> n_a=local_n, base_a=cp_rank, stride_a=cp.
    in-seq-split: two zigzag segments, both stride 1, bases=natural offsets.

    Returns a view (or clone if ``safe``) of comm_buffer[:total_num_tokens].
    """

    assert all_gather_rerange_supported(
        sources
    ), "sources do not satisfy all_gather_rerange constraints (count/dtype/shape/alignment)"
    numel_per_thread = 8  # 8 bf16 per 128-bit vector
    local_n = sources[0].shape[0]
    widths = [s.shape[1] for s in sources]
    row_stride_vecs = [s.stride(0) // numel_per_thread for s in sources]
    width = sum(widths)
    assert 0 <= n_a <= local_n, f"bad segment split {n_a=} {local_n=}"
    assert total_num_tokens >= 0, f"bad {total_num_tokens=}"
    if n_a > 0:
        assert stride_a > 0, f"bad {stride_a=}"
        last_a = base_a + (n_a - 1) * stride_a
        assert 0 <= base_a <= last_a < total_num_tokens, (
            f"segment A writes out of bounds: {base_a=}, {last_a=}, "
            f"{total_num_tokens=}"
        )
    n_b = local_n - n_a
    if n_b > 0:
        assert stride_b > 0, f"bad {stride_b=}"
        last_b = base_b + (n_b - 1) * stride_b
        assert 0 <= base_b <= last_b < total_num_tokens, (
            f"segment B writes out of bounds: {base_b=}, {last_b=}, "
            f"{total_num_tokens=}"
        )
    # View the flat comm buffer as the gathered [total, width] for this call
    # (also asserts total*width fits state.max_numel).
    comm_buffer = view_comm_buffer(state, total_num_tokens, width)

    out_vecs = width // numel_per_thread
    in0_vecs = widths[0] // numel_per_thread
    num_vecs = local_n * out_vecs
    in0 = sources[0]
    in1 = sources[1] if len(sources) == 2 else sources[0]  # unused if 1 src
    in0_row_stride_vecs = row_stride_vecs[0]
    in1_row_stride_vecs = (
        row_stride_vecs[1] if len(sources) == 2 else row_stride_vecs[0]
    )
    has_in1 = len(sources) == 2

    wait_for_workspace(state)
    symm_mem_handle = get_symm_mem_handle(state)

    block_size = _MULTIMEM_BLOCK_THREADS
    num_blocks = _MULTIMEM_MIN_BLOCKS
    all_gather_rerange_kernel[(num_blocks, 1, 1)](
        in0_ptr=in0,
        in1_ptr=in1,
        multicast_ptr=symm_mem_handle.multicast_ptr,
        signal_pad_ptr=symm_mem_handle.signal_pad_ptrs_dev,
        num_vecs=num_vecs,
        n_a=n_a,
        base_a=base_a,
        stride_a=stride_a,
        base_b=base_b,
        stride_b=stride_b,
        IN0_VECS=in0_vecs,
        IN0_ROW_STRIDE_VECS=in0_row_stride_vecs,
        IN1_ROW_STRIDE_VECS=in1_row_stride_vecs,
        OUT_VECS=out_vecs,
        HAS_IN1=has_in1,
        BLOCK_SIZE=block_size,
        RANK=symm_mem_handle.rank,
        WORLD_SIZE=symm_mem_handle.world_size,
        num_warps=block_size // _WARP_SIZE,
    )

    output = comm_buffer.clone() if safe else comm_buffer
    mark_workspace_used(state)
    return output
