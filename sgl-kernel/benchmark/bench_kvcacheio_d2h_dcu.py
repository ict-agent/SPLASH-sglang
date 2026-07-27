import itertools

import torch
import triton.testing
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer_mla,
    transfer_kv_all_layer_mla_lf_lf_D2H_dcu,
)

from sglang.srt.mem_cache.memory_pool_host import kernel_accessible_host_ptr
from sglang.srt.utils import is_hip


def make_case(num_layers: int, num_items: int, item_size: int):
    total_items = 2 * num_items + 2
    src_pool = torch.randint(
        0,
        256,
        (num_layers, total_items, item_size),
        dtype=torch.uint8,
        device="cuda",
    )
    dst_pool = torch.zeros(
        (num_layers, total_items, item_size),
        dtype=torch.uint8,
        pin_memory=True,
    )
    src_layers = torch.tensor(
        [src_pool[layer_id].data_ptr() for layer_id in range(num_layers)],
        dtype=torch.uint64,
        device="cuda",
    )
    dst_layers = torch.tensor(
        [
            kernel_accessible_host_ptr(dst_pool[layer_id])
            for layer_id in range(num_layers)
        ],
        dtype=torch.uint64,
        device="cuda",
    )
    src_indices = (torch.arange(num_items, device="cuda", dtype=torch.int64) * 2)
    dst_indices = src_indices + 1
    return src_layers, dst_layers, src_indices, dst_indices


def run_one(provider: str, num_layers: int, num_items: int, item_size: int):
    src_layers, dst_layers, src_indices, dst_indices = make_case(
        num_layers, num_items, item_size
    )
    if provider == "optimized":
        fn = lambda: transfer_kv_all_layer_mla_lf_lf_D2H_dcu(
            src_layers,
            dst_layers,
            src_indices,
            dst_indices,
            item_size=item_size,
            num_layers=num_layers,
        )
    else:
        fn = lambda: transfer_kv_all_layer_mla(
            src_layers,
            dst_layers,
            src_indices,
            dst_indices,
            item_size=item_size,
            num_layers=num_layers,
        )

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    moved_bytes = num_layers * num_items * item_size
    gib_per_s = moved_bytes / (ms / 1000.0) / (1024**3)
    return ms * 1000.0, gib_per_s


def main():
    if not is_hip():
        raise SystemExit("This benchmark requires HIP/DCU.")

    print("provider,layers,items,item_size,latency_us,GiB_per_s")
    for num_layers, num_items, item_size in itertools.product(
        [1, 11], [1, 17, 128, 1024], [528]
    ):
        for provider in ["generic", "optimized"]:
            latency_us, gib_per_s = run_one(
                provider, num_layers, num_items, item_size
            )
            print(
                f"{provider},{num_layers},{num_items},{item_size},"
                f"{latency_us:.3f},{gib_per_s:.3f}"
            )


if __name__ == "__main__":
    main()
