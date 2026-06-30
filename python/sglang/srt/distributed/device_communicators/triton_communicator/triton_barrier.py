# Adapted from https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py

"""Device-side cross-rank signal barriers over the symmetric-memory signal
pad, used to fence the multimem collectives at entry and exit."""

import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.triton_communicator.triton_utils import (
    get_flat_bid,
    get_flat_tid,
)


@triton.jit
def send_signal(addrs, sem: tl.constexpr):
    if sem == "relaxed":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.relaxed.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")


@triton.jit
def wait_signal(addrs, sem: tl.constexpr):
    if sem == "relaxed":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    atom.global.sys.relaxed.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    atom.global.sys.acquire.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")


@triton.jit
def blockwise_barrier(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
):
    if block_id is None:
        block_id = get_flat_bid()
    flat_tid = get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if flat_tid < world_size:
        send_signal(send_addrs, sem)
        wait_signal(wait_addrs, sem)


@triton.jit
def send_signal_to_peers(
    signal_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    for peer in tl.static_range(0, world_size):
        remote_signal = tl.load(signal_ptrs + peer).to(tl.pointer_type(tl.uint32))
        send_addr = remote_signal + block_id * world_size + rank
        send_old = tl.full((), 1, tl.int32)
        while send_old != 0:
            send_old = tl.atomic_cas(send_addr, 0, 1, sem="release", scope="sys")


@triton.jit
def wait_signal_from_peers(
    local_signal,
    block_id,
    world_size: tl.constexpr,
):
    for peer in tl.static_range(0, world_size):
        wait_addr = local_signal + block_id * world_size + peer
        wait_old = tl.full((), 0, tl.int32)
        while wait_old != 1:
            wait_old = tl.atomic_cas(wait_addr, 1, 0, sem="acquire", scope="sys")


@triton.jit
def symm_mem_barrier(
    signal_pad_ptrs_dev,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    signal_ptrs = signal_pad_ptrs_dev.to(tl.pointer_type(tl.uint64))
    local_signal = tl.load(signal_ptrs + rank).to(tl.pointer_type(tl.uint32))
    send_signal_to_peers(signal_ptrs, block_id, rank, world_size)
    wait_signal_from_peers(local_signal, block_id, world_size)
