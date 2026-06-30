# Adapted from https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py

"""NVIDIA multimem reduce-scatter (bf16) over symmetric memory."""

from typing import List

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.triton_communicator.comm_state import (
    _MULTIMEM_BLOCK_THREADS,
    _MULTIMEM_MAX_BLOCKS,
    _MULTIMEM_MIN_BLOCKS,
    _MULTIMEM_NUMEL_PER_THREAD,
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
    local_st_128,
    multimem_ld_reduce_128,
    sync_threads,
)


@triton.jit
def reduce_scatter_kernel(
    output_ptr,
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
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="acq_rel")
    sync_threads()

    numel = numel // NUMEL_PER_THREAD
    pid = tl.program_id(axis=0)
    tid = get_flat_tid()
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        thread_offset = block_start + tid
        mask = thread_offset < numel
        in_ptr = (
            multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64))
            + (in_offset // NUMEL_PER_THREAD + thread_offset) * 2
        )
        out_ptr = (
            output_ptr.to(tl.pointer_type(tl.uint64))
            + (out_offset // NUMEL_PER_THREAD + thread_offset) * 2
        )
        x, y, z, w = multimem_ld_reduce_128(in_ptr, mask)
        local_st_128(out_ptr, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="acq_rel")


def reduce_scatter_num_blocks(token_list_in_group: List[int], hidden_size: int) -> int:
    """Choose how many CTAs (grid blocks) the reduce-scatter kernel launches.

    The multimem kernel is a grid-stride loop, so the block count is a free
    tuning knob rather than dictated by the data: more CTAs expose more
    parallelism on a large payload, fewer avoid cross-CTA barrier cost on a
    small one. The count is sized from the busiest rank in the group
    (``max(token_list_in_group)``) because a collective must launch an identical
    grid on every rank for the kernel's per-CTA cross-rank barrier to pair up.

    Returns a power of two in ``[_MULTIMEM_MIN_BLOCKS, _MULTIMEM_MAX_BLOCKS]``:
    enough CTAs for roughly one grid-stride pass over the busiest rank's
    payload, floored and capped, then rounded up to a power of two. The cap is
    exactly what ``create_state`` reserves signal-pad slots for.
    """
    # Elements one CTA sweeps per grid-stride step (threads * elems-per-thread).
    numel_per_program = _MULTIMEM_BLOCK_THREADS * _MULTIMEM_NUMEL_PER_THREAD
    max_local_numel = max(token_list_in_group) * hidden_size
    needed_blocks = max(
        _MULTIMEM_MIN_BLOCKS, triton.cdiv(max_local_numel, numel_per_program)
    )
    return min(_MULTIMEM_MAX_BLOCKS, triton.next_power_of_2(needed_blocks))


def reduce_scatter(
    state: TritonMultimemState,
    hidden_states: torch.Tensor,
    tp_num_tokens: int = None,
    token_list_in_group: List[int] = None,
    safe: bool = True,
) -> torch.Tensor:
    """Multimem reduce-scatter of ``hidden_states`` over the comm group.

    ``hidden_states`` is the full ``[total_num_tokens, hidden_dim]`` bf16 tensor
    on every rank; the output is this rank's ``[local_num_tokens, hidden_dim]``
    reduced shard. Either ``tp_num_tokens`` (even split) or an explicit
    ``token_list_in_group`` must be provided. With ``safe=False`` the result
    aliases the comm buffer and is only valid until the next collective.

    Returns ``None`` when the full ``[total_num_tokens, hidden]`` does not fit
    ``state.max_numel`` so the caller can fall back to NCCL.
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
        hidden_states.shape[0] == total_num_tokens
    ), f"Mismatched shape, {hidden_states.shape[0]=} != {total_num_tokens=}"

    if not fits_comm_buffer(state, total_num_tokens, hidden):
        return None
    # The multimem ld_reduce reads across ranks from the symmetric buffer, so
    # the input must be staged into comm_buffer first (unlike all-gather, whose
    # load is local). View the flat buffer as [total, hidden].
    comm_buffer = view_comm_buffer(state, total_num_tokens, hidden)
    wait_for_workspace(state)
    comm_buffer.copy_(hidden_states)

    # Uniform split: plain rank-major scatter along the token dim -- hand it to
    # torch's maintained multimem reduce_scatter (reduces the staged comm buffer
    # across ranks into this rank's shard). The triton kernel below is kept only
    # for the non-uniform token_list case, which torch's op does not support.
    if is_even_token_distribution(token_list_in_group):
        # torch's reduce_scatter_out wants 1D in/out. The staged comm buffer is
        # row-major contiguous, so scattering the flat [total*hidden] vector
        # gives each rank a contiguous [local_num_tokens*hidden] chunk == its
        # [local_num_tokens, hidden] shard; view it back after.
        flat_in = comm_buffer.reshape(-1)
        flat_out = torch.empty(
            local_num_tokens * hidden,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.symm_mem.reduce_scatter_out(
            flat_in, state.group.group_name, False, flat_out
        )
        mark_workspace_used(state)
        return flat_out.view(local_num_tokens, hidden)

    num_elts = local_num_tokens * hidden
    in_offset = local_token_offset * hidden
    assert in_offset % _MULTIMEM_NUMEL_PER_THREAD == 0, (
        "reduce_scatter local shard offset must be 16-byte aligned, got "
        f"{local_token_offset=} {hidden=}"
    )
    num_blocks = reduce_scatter_num_blocks(token_list_in_group, hidden)
    num_blocks, block_size, num_warps, numel_per_thread = get_launch_config(
        num_elts, num_blocks=num_blocks
    )
    symm_mem_handle = get_symm_mem_handle(state)
    # safe=True: the kernel writes the reduced shard straight into a fresh
    # private output at offset 0, so no post-kernel clone of comm_buffer is
    # needed -- saves a full [local, hidden] device copy + a kernel launch on
    # the default path. safe=False: write in place into comm_buffer and alias it
    # (valid only until the next collective).
    if safe:
        output = torch.empty(
            (local_num_tokens, hidden),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        output_ptr = output
        out_offset = 0
    else:
        output = comm_buffer[
            local_token_offset : (local_token_offset + local_num_tokens), :
        ]
        output_ptr = state.comm_buffer
        out_offset = in_offset
    reduce_scatter_kernel[(num_blocks, 1, 1)](
        output_ptr=output_ptr,
        multicast_ptr=symm_mem_handle.multicast_ptr,
        signal_pad_ptr=symm_mem_handle.signal_pad_ptrs_dev,
        numel=num_elts,
        in_offset=in_offset,
        out_offset=out_offset,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_handle.rank,
        WORLD_SIZE=symm_mem_handle.world_size,
        num_warps=num_warps,
    )
    mark_workspace_used(state)
    return output
