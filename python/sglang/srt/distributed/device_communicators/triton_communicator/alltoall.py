# Adapted from https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py

"""NVIDIA symm-mem AllToAll helpers (bf16) used by KDA-TP qkvo head sharding.

The copy kernels push each chunk straight into the destination rank's symm-mem
comm buffer at the slot that matches the final layout, so the
transpose+contiguous trips in the NCCL fallback paths disappear.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl
import torch.distributed._symmetric_memory as torch_symm_mem
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.utils import (
    get_bool_env_var,
    get_int_env_var,
)
from sglang.srt.distributed.device_communicators.triton_communicator.comm_state import (
    TritonMultimemState,
    fits_comm_buffer,
    get_symm_mem_handle,
    mark_workspace_used,
    view_comm_buffer,
    wait_for_workspace,
)
from sglang.srt.distributed.device_communicators.triton_communicator.triton_barrier import (
    blockwise_barrier,
    sync_kernel,
)
from sglang.srt.distributed.device_communicators.triton_communicator.triton_utils import (
    sync_threads,
)


logger = logging.getLogger(__name__)

COPY_GRID_CLAMP_WARNED: set[str] = set()


def clamp_copy_grid_to_signal_pad(
    op_name: str,
    copy_grid: int,
    signal_pad_max_sync_blocks: int,
    env_var: str,
) -> int:
    if copy_grid <= signal_pad_max_sync_blocks:
        return copy_grid

    if op_name not in COPY_GRID_CLAMP_WARNED:
        logger.warning(
            "%s copy grid is adjusted from %s to %s because the "
            "symmetric-memory signal pad supports only %s sync blocks. This "
            "keeps the in-copy entry barrier within the signal pad. If %s was "
            "set manually, lower it to %s or less to avoid this warning.",
            op_name,
            copy_grid,
            signal_pad_max_sync_blocks,
            signal_pad_max_sync_blocks,
            env_var,
            signal_pad_max_sync_blocks,
        )
        COPY_GRID_CLAMP_WARNED.add(op_name)
    return signal_pad_max_sync_blocks


def alltoall_token_to_head(
    state: TritonMultimemState,
    core_attn_full: torch.Tensor,
    *,
    safe: bool = True,
) -> Optional[torch.Tensor]:
    """Symm-mem token->head fast path.

    Input  on rank r: ``(n_local, head_full)`` bf16, contiguous -- the
    full-head core-attention output produced by every rank.
    Output on rank r: compact ``(n_global, f_local)`` with
    ``f_local = head_full // world_size`` -- the head shard owned by this rank
    after the token<->head transpose.

    With ``safe=True`` (default) the result is cloned into a fresh contiguous
    tensor; with ``safe=False`` the direct AllToAll path aliases the compact
    comm buffer and is only valid until the next collective. ``torch.matmul``
    (o_proj) accepts non-contiguous inputs, so callers in the hot path can pass
    ``safe=False`` to skip the clone.

    This uses direct compact AllToAll for all batch sizes with a single tiled
    copy kernel; per-``n_local`` tile and worker-grid sizes come from the
    tuned tables below.

    Returns ``None`` when the required staging buffer does not fit
    ``state.max_numel`` so the caller can fall back to NCCL.
    """
    assert core_attn_full.dtype == torch.bfloat16, "Only bfloat16 is supported for now"
    assert core_attn_full.dim() == 2, "core_attn_full must be 2-D"
    assert core_attn_full.is_contiguous(), "core_attn_full must be contiguous"
    world_size = state.world_size
    n_local, head_full = core_attn_full.shape
    assert head_full % world_size == 0, (
        f"head_full={head_full} must be divisible by world_size={world_size}"
    )
    f_local = head_full // world_size
    n_global = n_local * world_size
    t_block, total_work, copy_grid = token_to_head_work_grid(
        world_size,
        n_local,
        head_full,
        core_attn_full.element_size(),
    )
    if total_work == 0:
        return core_attn_full.new_empty((0, f_local))
    if not fits_comm_buffer(state, n_global, f_local):
        return None
    comm_buffer = view_comm_buffer(state, n_global, f_local)

    signal_pad_max_sync_blocks = max(
        1, torch_symm_mem.get_signal_pad_size() // 4 // world_size
    )
    copy_grid = clamp_copy_grid_to_signal_pad(
        "alltoall_token_to_head",
        copy_grid,
        signal_pad_max_sync_blocks,
        "SGLANG_KDA_T2H_GRID",
    )
    use_two_pass = copy_grid >= KDA_T2H_TWOPASS_MIN_GRID

    wait_for_workspace(state)
    symm_mem_handle = get_symm_mem_handle(state)
    use_gdc = use_two_pass and KDA_T2H_USE_PDL and is_arch_support_pdl()

    grid = (copy_grid,)
    f_block = next_pow2(f_local)
    kda_a2a_token_to_head_kernel[grid](
        symm_mem_handle.buffer_ptrs_dev,
        symm_mem_handle.signal_pad_ptrs_dev,
        core_attn_full,
        core_attn_full.stride(0),
        core_attn_full.stride(1),
        comm_buffer.stride(0),
        comm_buffer.stride(1),
        N_LOCAL=n_local,
        F_LOCAL=f_local,
        F_BLOCK=f_block,
        T_BLOCK=t_block,
        COPY_GRID=copy_grid,
        DO_SYNC=not use_two_pass,
        USE_GDC=use_gdc,
        launch_pdl=use_gdc,
        rank=symm_mem_handle.rank,
        world_size=symm_mem_handle.world_size,
        num_warps=tuned_t2h_num_warps(n_local),
    )
    if use_two_pass:
        sync_kernel[(1,)](
            symm_mem_handle.signal_pad_ptrs_dev,
            USE_GDC=use_gdc,
            launch_pdl=use_gdc,
            rank=symm_mem_handle.rank,
            world_size=symm_mem_handle.world_size,
        )

    output = comm_buffer.clone() if safe else comm_buffer
    mark_workspace_used(state)
    return output


# The head->token copy has one logical chunk per (dst_rank, qkv_group, token).
# Tiling local tokens into one work item and assigning work items to a fixed
# worker grid reduces CTA count enough to improve coexistence with neighboring
# kernels. When the worker grid is still large, two-pass sync avoids one global
# signal/wait barrier per copy CTA; with the default small worker grid,
# single-pass sync avoids an extra post-copy kernel launch.
KDA_H2T_TWOPASS_MIN_GRID = get_int_env_var("SGLANG_KDA_H2T_TWOPASS_MIN_GRID", 128)
KDA_T2H_TWOPASS_MIN_GRID = get_int_env_var("SGLANG_KDA_T2H_TWOPASS_MIN_GRID", 128)
KDA_H2T_USE_PDL = get_bool_env_var("SGLANG_KDA_H2T_USE_PDL", "true")
KDA_T2H_USE_PDL = get_bool_env_var("SGLANG_KDA_T2H_USE_PDL", "true")
KDA_H2T_T_BLOCK = get_int_env_var("SGLANG_KDA_H2T_T_BLOCK", 0)
KDA_T2H_T_BLOCK = get_int_env_var("SGLANG_KDA_T2H_T_BLOCK", 0)
KDA_H2T_GRID = get_int_env_var("SGLANG_KDA_H2T_GRID", 0)
KDA_T2H_GRID = get_int_env_var("SGLANG_KDA_T2H_GRID", 0)
KDA_H2T_NUM_WARPS = get_int_env_var("SGLANG_KDA_H2T_NUM_WARPS", 0)
KDA_T2H_NUM_WARPS = get_int_env_var("SGLANG_KDA_T2H_NUM_WARPS", 0)
# Grid=0 means no override: use PyTorch multimem_all_gather_out's launch
# calculation with default max_num_blocks=8. Positive values force copy_grid
# directly for targeted experiments.
KDA_MULTIMEM_DEFAULT_GRID = 8
# T_BLOCK=0 means use the 8x H100 / bf16 sweep tables below. Positive values
# force a fixed token tile size for targeted experiments.
KDA_H2T_TUNED_T_BLOCK = (
    (96, 64),
    (2048, 32),
)
KDA_T2H_TUNED_T_BLOCK = (
    (1, 1),
    (2, 2),
    (4, 4),
    (8, 4),
    (16, 16),
    (32, 32),
    (64, 32),
    (512, 32),
    (2048, 64),
)
# Worker-grid cap for the T2H direct AllToAll. Unlike multimem all-gather
# (where the NVSwitch fans out one store to all peers and ~8 CTAs saturate),
# the direct path issues unicast stores, so large payloads need many more
# CTAs in flight to cover NVLink latency. Values from the 8x H100 sweep;
# grid >= KDA_T2H_TWOPASS_MIN_GRID flips to the two-pass barrier.
KDA_T2H_TUNED_MAX_GRID = (
    (64, 8),
    (128, 32),
    (256, 64),
    (2048, 128),
)
KDA_H2T_TUNED_NUM_WARPS = (
    (1, 4),
    (2, 16),
    (4, 16),
    (8, 16),
    (16, 16),
    (32, 4),
    (64, 16),
    (128, 16),
)
KDA_T2H_TUNED_NUM_WARPS = (
    (1, 8),
    (2, 16),
    (4, 16),
    (8, 8),
    (16, 8),
    (32, 32),
    (64, 8),
    (128, 16),
    (256, 32),
    (512, 8),
    (1024, 32),
    (2048, 16),
)


@triton.jit
def kda_a2a_head_to_token_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    # Input strides — (n_global, num_groups * F_local)
    stride_in_row: tl.constexpr,
    stride_in_col: tl.constexpr,
    # Output (symm-mem) strides — (n_local, num_groups * tp * F_local)
    stride_out_row: tl.constexpr,
    stride_out_col: tl.constexpr,
    N_LOCAL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    F_LOCAL: tl.constexpr,
    F_BLOCK: tl.constexpr,
    T_BLOCK: tl.constexpr,
    COPY_GRID: tl.constexpr,
    DO_SYNC: tl.constexpr,
    USE_GDC: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Fixed worker grid over (dst_rank, group, t_tile) copy chunks.

    Pushes the chunk directly into the destination rank's symm-mem output
    buffer at the column slot that matches the final per-token layout
    ``[g0_r0, g0_r1, ..., g0_r{T-1}, g1_r0, ...]``.
    """
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(N_LOCAL, T_BLOCK)
    total_work = world_size * NUM_GROUPS * n_tiles
    n_per_dst = n_tiles * NUM_GROUPS
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    offs_f = tl.arange(0, F_BLOCK)

    blockwise_barrier(symm_mem_signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    for work_id in tl.range(pid, total_work, COPY_GRID):
        dst_rank = work_id // n_per_dst
        rem = work_id % n_per_dst
        g = rem // n_tiles
        t_tile = rem % n_tiles

        offs_t = t_tile * T_BLOCK + tl.arange(0, T_BLOCK)
        mask = (offs_t[:, None] < N_LOCAL) & (offs_f[None, :] < F_LOCAL)

        t_global = dst_rank * N_LOCAL + offs_t
        src_cols = g * F_LOCAL + offs_f
        in_ptrs = (
            input_ptr
            + t_global[:, None] * stride_in_row
            + src_cols[None, :] * stride_in_col
        )
        val = tl.load(in_ptrs, mask=mask, other=0.0)

        remote_buffer = tl.load(buffer_ptrs + dst_rank).to(
            tl.pointer_type(input_ptr.dtype.element_ty)
        )
        remote_buffer = tl.multiple_of(remote_buffer, 16)

        dst_cols = g * (world_size * F_LOCAL) + rank * F_LOCAL + offs_f
        out_ptrs = (
            remote_buffer
            + offs_t[:, None] * stride_out_row
            + dst_cols[None, :] * stride_out_col
        )
        tl.store(out_ptrs, val, mask=mask)

    sync_threads()

    if USE_GDC and not DO_SYNC:
        gdc_launch_dependents()

    if DO_SYNC:
        blockwise_barrier(
            symm_mem_signal_pad_ptrs, None, rank, world_size, sem="acq_rel"
        )


@triton.jit
def kda_a2a_token_to_head_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    # Input strides - (n_local, tp * F_local)
    stride_in_row: tl.constexpr,
    stride_in_col: tl.constexpr,
    # Output (symm-mem) strides - compact (tp * n_local, F_local)
    stride_out_row: tl.constexpr,
    stride_out_col: tl.constexpr,
    N_LOCAL: tl.constexpr,
    F_LOCAL: tl.constexpr,
    F_BLOCK: tl.constexpr,
    T_BLOCK: tl.constexpr,
    COPY_GRID: tl.constexpr,
    DO_SYNC: tl.constexpr,
    USE_GDC: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Fixed worker grid over (dst_rank, t_tile) copy chunks.

    Pushes the head slice owned by ``dst_rank`` directly into that rank's
    compact comm buffer at rows ``rank * N_LOCAL + t``. This matches the NCCL
    fallback layout ``(n_global, F_LOCAL)`` instead of the legacy full-head
    all-gather + slice layout.
    """
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(N_LOCAL, T_BLOCK)
    total_work = world_size * n_tiles
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    offs_f = tl.arange(0, F_BLOCK)

    blockwise_barrier(symm_mem_signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    for work_id in tl.range(pid, total_work, COPY_GRID):
        dst_rank = work_id // n_tiles
        t_tile = work_id % n_tiles

        offs_t = t_tile * T_BLOCK + tl.arange(0, T_BLOCK)
        mask = (offs_t[:, None] < N_LOCAL) & (offs_f[None, :] < F_LOCAL)

        src_cols = dst_rank * F_LOCAL + offs_f
        in_ptrs = (
            input_ptr
            + offs_t[:, None] * stride_in_row
            + src_cols[None, :] * stride_in_col
        )
        val = tl.load(in_ptrs, mask=mask, other=0.0)

        remote_buffer = tl.load(buffer_ptrs + dst_rank).to(
            tl.pointer_type(input_ptr.dtype.element_ty)
        )
        remote_buffer = tl.multiple_of(remote_buffer, 16)

        dst_rows = rank * N_LOCAL + offs_t
        dst_cols = offs_f
        out_ptrs = (
            remote_buffer
            + dst_rows[:, None] * stride_out_row
            + dst_cols[None, :] * stride_out_col
        )
        tl.store(out_ptrs, val, mask=mask)

    sync_threads()

    if USE_GDC and not DO_SYNC:
        gdc_launch_dependents()

    if DO_SYNC:
        blockwise_barrier(
            symm_mem_signal_pad_ptrs, None, rank, world_size, sem="acq_rel"
        )


@triton.jit
def gdc_launch_dependents():
    tl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; // dummy $0",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def lookup_tuned_value(n_local: int, table: tuple[tuple[int, int], ...]) -> int:
    bucket = next_pow2(max(1, n_local))
    for max_n_local, value in table:
        if bucket <= max_n_local:
            return value
    return table[-1][1]


def lookup_tuned_by_n_local(n_local: int, table: tuple[tuple[int, int], ...]) -> int:
    n_local = max(1, n_local)
    for max_n_local, value in table:
        if n_local <= max_n_local:
            return value
    return table[-1][1]


def normalize_t_block(n_local: int, t_block: int) -> int:
    target_t_block = next_pow2(max(1, t_block))
    return min(target_t_block, next_pow2(max(1, n_local)))


def tuned_h2t_t_block(n_local: int) -> int:
    if KDA_H2T_T_BLOCK > 0:
        return normalize_t_block(n_local, KDA_H2T_T_BLOCK)
    return normalize_t_block(
        n_local, lookup_tuned_by_n_local(n_local, KDA_H2T_TUNED_T_BLOCK)
    )


def tuned_t2h_t_block(n_local: int) -> int:
    if KDA_T2H_T_BLOCK > 0:
        return normalize_t_block(n_local, KDA_T2H_T_BLOCK)
    return normalize_t_block(
        n_local, lookup_tuned_by_n_local(n_local, KDA_T2H_TUNED_T_BLOCK)
    )


def tuned_h2t_num_warps(n_local: int) -> int:
    if KDA_H2T_NUM_WARPS > 0:
        return KDA_H2T_NUM_WARPS
    return lookup_tuned_value(n_local, KDA_H2T_TUNED_NUM_WARPS)


def tuned_t2h_num_warps(n_local: int) -> int:
    if KDA_T2H_NUM_WARPS > 0:
        return KDA_T2H_NUM_WARPS
    return lookup_tuned_value(n_local, KDA_T2H_TUNED_NUM_WARPS)


def forced_h2t_grid() -> Optional[int]:
    return KDA_H2T_GRID if KDA_H2T_GRID > 0 else None


def forced_t2h_grid() -> Optional[int]:
    return KDA_T2H_GRID if KDA_T2H_GRID > 0 else None


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _multimem_like_alignment(num_bytes: int, element_size: int) -> int:
    min_alignment = max(4, element_size)
    for alignment in (16, 8, 4):
        if alignment >= min_alignment and num_bytes % alignment == 0:
            return alignment
    return min_alignment


def _multimem_like_copy_grid(
    *,
    input_bytes: int,
    output_bytes: int,
    element_size: int,
    total_work: int,
    max_blocks: int,
) -> int:
    if total_work == 0:
        return 0

    alignment = _multimem_like_alignment(output_bytes, element_size)
    aligned_bytes = _round_up(input_bytes, alignment)
    max_threads = 1024
    if aligned_bytes <= max_threads * alignment:
        copy_grid = 1
    else:
        copy_grid = min(
            triton.cdiv(aligned_bytes, max_threads * alignment),
            max(1, max_blocks),
        )
    return min(total_work, max(1, copy_grid))


def head_to_token_work_grid(
    world_size: int,
    n_local: int,
    num_qkv_groups: int,
    f_local: Optional[int] = None,
    element_size: Optional[int] = None,
):
    """Return H2T token tiling and a PyTorch multimem-like worker grid.

    When feature width is known, the effective copy grid mirrors
    ``torch.ops.symm_mem.multimem_all_gather_out`` launch sizing from the
    local rank payload byte count. ``SGLANG_KDA_H2T_GRID=0`` uses
    PyTorch's default ``max_num_blocks=8``; positive values force the
    launch ``copy_grid`` directly.
    """
    t_block = tuned_h2t_t_block(n_local)
    n_tiles = triton.cdiv(n_local, t_block)
    total_work = world_size * num_qkv_groups * n_tiles
    if total_work == 0:
        return t_block, total_work, 0

    forced_grid = forced_h2t_grid()
    if forced_grid is not None:
        return t_block, total_work, max(1, forced_grid)

    if f_local is None or element_size is None:
        copy_grid = min(total_work, KDA_MULTIMEM_DEFAULT_GRID)
        return t_block, total_work, copy_grid

    input_bytes = world_size * n_local * num_qkv_groups * f_local * element_size
    output_bytes = world_size * input_bytes
    copy_grid = _multimem_like_copy_grid(
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        element_size=element_size,
        total_work=total_work,
        max_blocks=KDA_MULTIMEM_DEFAULT_GRID,
    )
    return t_block, total_work, copy_grid


def token_to_head_work_grid(
    world_size: int,
    n_local: int,
    head_full: int,
    element_size: int,
):
    """Return T2H token tiling and a payload-scaled worker grid.

    Size the launch from the local input payload with ``max_num_threads=1024``
    like PyTorch's multimem launch calculation, but cap the CTA count with the
    n_local-bucketed ``KDA_T2H_TUNED_MAX_GRID`` sweep table instead of
    multimem's ``max_num_blocks=8``: the direct AllToAll issues unicast NVLink
    stores, so large payloads need far more CTAs in flight than a multimem
    fan-out. ``SGLANG_KDA_T2H_GRID=0`` uses the table; positive values force
    the launch ``copy_grid`` directly.
    """
    t_block = tuned_t2h_t_block(n_local)
    n_tiles = triton.cdiv(n_local, t_block)
    total_work = world_size * n_tiles
    if total_work == 0:
        return t_block, total_work, 0

    forced_grid = forced_t2h_grid()
    if forced_grid is not None:
        return t_block, total_work, max(1, forced_grid)

    input_bytes = n_local * head_full * element_size
    output_bytes = input_bytes
    copy_grid = _multimem_like_copy_grid(
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        element_size=element_size,
        total_work=total_work,
        max_blocks=lookup_tuned_by_n_local(n_local, KDA_T2H_TUNED_MAX_GRID),
    )
    return t_block, total_work, copy_grid


def alltoall_head_to_token(
    state: TritonMultimemState,
    qkv_global_shard: torch.Tensor,
    *,
    num_qkv_groups: int = 3,
    safe: bool = True,
) -> Optional[torch.Tensor]:
    """Tuned multimem AllToAll head->token with worker-grid + two-pass + PDL.

    Uses the fixed-grid copy kernel (``kda_a2a_head_to_token_kernel``) with
    per-call work-grid sizing and an optional two-pass post-copy barrier
    (``_kda_a2a_sync_kernel``). The two-pass path lets the copy CTAs scale
    independently of the cross-rank sync CTAs and enables PDL/GDC overlap on
    arch >= sm_90.
    """
    assert qkv_global_shard.dtype == torch.bfloat16, (
        "Only bfloat16 is supported for now"
    )
    assert qkv_global_shard.dim() == 2, (
        f"expected (n_global, num_qkv_groups*F_local), got "
        f"{tuple(qkv_global_shard.shape)}"
    )
    world_size = state.world_size
    n_global, feat = qkv_global_shard.shape
    assert n_global % world_size == 0, (
        f"n_global={n_global} must be divisible by world_size={world_size}"
    )
    assert feat % num_qkv_groups == 0, (
        f"feat={feat} must be divisible by num_qkv_groups={num_qkv_groups}"
    )
    n_local = n_global // world_size
    f_local = feat // num_qkv_groups
    feat_out = num_qkv_groups * world_size * f_local

    if not fits_comm_buffer(state, n_local, feat_out):
        return None
    comm_buffer = view_comm_buffer(state, n_local, feat_out)

    t_block, total_work, copy_grid = head_to_token_work_grid(
        world_size,
        n_local,
        num_qkv_groups,
        f_local,
        qkv_global_shard.element_size(),
    )
    if total_work == 0:
        # Nothing to copy; the comm buffer view is meaningless to a caller.
        # Match the safe contract of the main path so no caller holds a stale
        # alias of the comm buffer.
        return comm_buffer.clone() if safe else comm_buffer

    wait_for_workspace(state)
    symm_mem_handle = get_symm_mem_handle(state)

    # Each cross-rank sync CTA consumes 4 bytes per peer rank in the symm-mem
    # signal pad. Keep the copy grid within the pad row capacity so the in-copy
    # entry barrier is always safe, even when env vars force a larger grid.
    signal_pad_max_sync_blocks = max(
        1, torch_symm_mem.get_signal_pad_size() // 4 // world_size
    )
    copy_grid = clamp_copy_grid_to_signal_pad(
        "alltoall_head_to_token",
        copy_grid,
        signal_pad_max_sync_blocks,
        "SGLANG_KDA_H2T_GRID",
    )
    use_two_pass = copy_grid >= KDA_H2T_TWOPASS_MIN_GRID
    use_gdc = use_two_pass and KDA_H2T_USE_PDL and is_arch_support_pdl()

    grid = (copy_grid,)
    f_block = next_pow2(f_local)
    kda_a2a_head_to_token_kernel[grid](
        symm_mem_handle.buffer_ptrs_dev,
        symm_mem_handle.signal_pad_ptrs_dev,
        qkv_global_shard,
        qkv_global_shard.stride(0),
        qkv_global_shard.stride(1),
        comm_buffer.stride(0),
        comm_buffer.stride(1),
        N_LOCAL=n_local,
        NUM_GROUPS=num_qkv_groups,
        F_LOCAL=f_local,
        F_BLOCK=f_block,
        T_BLOCK=t_block,
        COPY_GRID=copy_grid,
        DO_SYNC=not use_two_pass,
        USE_GDC=use_gdc,
        launch_pdl=use_gdc,
        rank=symm_mem_handle.rank,
        world_size=symm_mem_handle.world_size,
        num_warps=tuned_h2t_num_warps(n_local),
    )
    if use_two_pass:
        sync_kernel[(1,)](
            symm_mem_handle.signal_pad_ptrs_dev,
            USE_GDC=use_gdc,
            launch_pdl=use_gdc,
            rank=symm_mem_handle.rank,
            world_size=symm_mem_handle.world_size,
        )

    output = comm_buffer.clone() if safe else comm_buffer
    mark_workspace_used(state)
    return output
