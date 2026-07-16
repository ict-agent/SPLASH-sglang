import itertools

import torch
import triton
import triton.testing

import sgl_kernel
from sglang.srt.layers.attention.nsa.kpool.kernels import (
    update_kpool_write_plan_cuda_graph,
)


def _make_inputs(bs: int, num_draft_tokens: int, has_per_q: bool):
    device = "cuda"
    pool_size = 4
    slots_per_page = 16
    max_pages = 8192
    write_start = torch.randint(0, 32768, (bs,), dtype=torch.int32, device=device)
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)
    real_page_table = torch.randint(
        0,
        32768,
        (bs * num_draft_tokens, max_pages),
        dtype=torch.int32,
        device=device,
    )
    req_out = torch.empty(bs, dtype=torch.int64, device=device)
    write_start_out = torch.empty(bs, dtype=torch.int32, device=device)
    tail_logical_start_out = torch.empty(bs, dtype=torch.int32, device=device)
    write_loc_out = torch.empty(bs, dtype=torch.int64, device=device)
    if has_per_q:
        pool_seqlens_per_q_out = torch.empty(
            bs * num_draft_tokens, dtype=torch.int32, device=device
        )
        seqlens_per_q_out = torch.empty(
            bs * num_draft_tokens, dtype=torch.int32, device=device
        )
    else:
        pool_seqlens_per_q_out = None
        seqlens_per_q_out = None
    return dict(
        write_start=write_start,
        req_pool_indices=req_pool_indices,
        real_page_table=real_page_table,
        req_out=req_out,
        write_start_out=write_start_out,
        tail_logical_start_out=tail_logical_start_out,
        write_loc_out=write_loc_out,
        pool_seqlens_per_q_out=pool_seqlens_per_q_out,
        seqlens_per_q_out=seqlens_per_q_out,
        pool_size=pool_size,
        num_draft_tokens=num_draft_tokens,
        slots_per_page=slots_per_page,
    )


CONFIGS = [
    config
    for config in itertools.product([1, 8, 32, 128], [1, 2], [False, True])
    if not (config[1] == 1 and config[2])
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "num_draft_tokens", "has_per_q"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=["aot", "triton"],
        line_names=["AOT sgl-kernel", "Triton"],
        styles=[("green", "-"), ("blue", "--")],
        ylabel="us",
        plot_name="kpool-write-plan",
        args={},
    )
)
def benchmark(bs, num_draft_tokens, has_per_q, provider):
    kwargs = _make_inputs(bs, num_draft_tokens, has_per_q)
    if provider == "aot":
        fn = lambda: sgl_kernel.kpool_write_plan(**kwargs)
    else:
        fn = lambda: update_kpool_write_plan_cuda_graph(**kwargs)
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
