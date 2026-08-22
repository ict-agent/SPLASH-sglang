"""Exercise Mooncake duplicate registration and the EPD registration guards.

Run inside a container with Mooncake, RDMA devices, and this checkout mounted:

    python test/manual/kv_transfer/test_mooncake_rdma_registration.py \
        --hostname 10.3.14.14 --ib-device shca_0 --mode fixed

Use ``--mode raw`` only as an isolated control: it intentionally registers the
same region twice and expects the second registration to fail.
"""

import argparse

import torch

from sglang.srt.disaggregation.encode_receiver import (
    RdmaBufferPool,
    RdmaRegRefcount,
    rdma_pool_enabled,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)


def test_raw_overlap(engine) -> None:
    tensor = torch.empty(1024 * 1024, dtype=torch.uint8)
    first = engine.register(tensor.data_ptr(), tensor.nbytes)
    try:
        second = engine.register(tensor.data_ptr(), tensor.nbytes)
        assert first == 0, f"first registration failed: {first}"
        assert second != 0, "raw duplicate registration unexpectedly succeeded"
        print(f"raw duplicate registration reproduced: second_ret={second}")
    finally:
        if first == 0:
            engine.deregister(tensor.data_ptr())


def test_fixed_registration(engine) -> None:
    register_calls = []
    deregister_calls = []
    original_register = engine.register
    original_deregister = engine.deregister

    def counted_register(addr, nbytes):
        register_calls.append((addr, nbytes))
        return original_register(addr, nbytes)

    def counted_deregister(addr):
        deregister_calls.append(addr)
        return original_deregister(addr)

    engine.register = counted_register
    engine.deregister = counted_deregister

    tensor = torch.empty(1024 * 1024, dtype=torch.uint8)
    registry = RdmaRegRefcount(engine)
    first_addr = registry.acquire(tensor)
    second_addr = registry.acquire(tensor)
    assert first_addr == second_addr
    assert len(register_calls) == 1, register_calls
    registry.release(first_addr)
    assert deregister_calls == []
    registry.release(second_addr)
    assert deregister_calls == [first_addr]

    assert rdma_pool_enabled(), "set both SGLANG_MC_RDMA_POOL_* limits"
    register_calls.clear()
    deregister_calls.clear()
    pool = RdmaBufferPool(engine)
    first_buffer = pool.acquire(1024)
    pool.release(first_buffer)
    second_buffer = pool.acquire(1024)
    assert first_buffer.data_ptr() == second_buffer.data_ptr()
    assert len(register_calls) == 1, register_calls
    pool.discard(second_buffer)
    assert deregister_calls == [second_buffer.data_ptr()]
    print("fixed registration passed: refcount=ok receiver_pool_reuse=ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--ib-device", default="")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--mode", choices=("raw", "fixed"), default="fixed")
    args = parser.parse_args()

    engine = MooncakeTransferEngine(
        hostname=args.hostname,
        gpu_id=args.gpu_id,
        ib_device=args.ib_device,
    )
    if args.mode == "raw":
        test_raw_overlap(engine)
    else:
        test_fixed_registration(engine)


if __name__ == "__main__":
    main()
