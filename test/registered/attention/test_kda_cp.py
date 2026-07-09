"""Context-parallel KDA prefill correctness test.

Exercises the parallel all-gather + merge path (``chunk_kda_cp`` +
``fla/cp`` + the plain-split metadata / context helpers) against a
single-process reference (``chunk_kda`` over the full sequence), verifying both
the per-token output ``o`` and the per-request final SSM state.

Requires >= 2 CUDA GPUs; spawns one process per rank. Run directly:

    python -m pytest test/registered/attention/test_kda_cp.py -s
    # or a single scenario:
    python test/registered/attention/test_kda_cp.py

NOTE: authored on a CPU-only host; must be run on a multi-GPU box.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def _init_distributed(rank: int, world_size: int, port: str = "29555"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def _worker(rank, world_size, lengths, H, D, dtype, ratio, ok_flag):
    from sglang.srt.layers.attention.fla.kda import chunk_kda, chunk_kda_cp
    from sglang.srt.layers.attention.linear.kda_cp_utils import (
        build_kda_fla_cp_context,
        kda_cp_continuation_segment_index,
        kda_cp_owner_of_global_token,
        prepare_kda_prefill_cp_metadata,
    )

    try:
        _init_distributed(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        T = sum(lengths)
        N = len(lengths)

        # ---- global inputs (generated identically on every rank) ----
        torch.manual_seed(1234)
        q = torch.randn(1, T, H, D, device=device, dtype=dtype)
        k = torch.randn(1, T, H, D, device=device, dtype=dtype)
        v = torch.randn(1, T, H, D, device=device, dtype=dtype)
        g = F.logsigmoid(torch.randn(1, T, H, D, device=device, dtype=torch.float)).clamp_(
            min=-5.0
        )
        beta = torch.randn(1, T, H, device=device, dtype=dtype).sigmoid()

        cu = [0]
        for x in lengths:
            cu.append(cu[-1] + x)
        cu_seqlens_global = torch.tensor(cu, device=device, dtype=torch.long)

        # ---- single-process reference over the full sequence ----
        ref_state = torch.zeros(N, H, D, D, device=device, dtype=torch.float32)
        o_ref, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ref_state,
            initial_state_indices=torch.arange(N, device=device, dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens_global,
        )
        # ref_state now holds each sequence's final state (in-place).

        # ---- context-parallel run on this rank's token slice ----
        meta = prepare_kda_prefill_cp_metadata(
            total_tokens=T,
            extend_seq_lens_cpu=lengths,
            cp_rank=rank,
            cp_size=world_size,
            device=device,
        )
        ctx = build_kda_fla_cp_context(
            meta, dist.group.WORLD, conv1d_kernel_size=4
        )
        lo, hi = meta.local_start, meta.local_end
        n_local = len(meta.local_seq_lens_cpu)
        cont_idx = kda_cp_continuation_segment_index(meta)

        local_state = torch.zeros(
            max(n_local, 1), H, D, D, device=device, dtype=torch.float32
        )
        o_local, _ = chunk_kda_cp(
            q=q[:, lo:hi],
            k=k[:, lo:hi],
            v=v[:, lo:hi],
            g=g[:, lo:hi],
            beta=beta[:, lo:hi],
            initial_state=local_state,
            initial_state_indices=torch.arange(n_local, device=device, dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=meta.local_query_start_loc,
            cp_context=ctx,
            continuation_h0=None,
            continuation_index=cont_idx,
        )

        passed = True

        # (1) output: compare this rank's slice against the reference slice.
        if hi > lo:
            diff = (o_local.float() - o_ref[:, lo:hi].float()).abs()
            denom = o_ref[:, lo:hi].float().abs().mean().clamp_min(1e-6)
            rel = diff.mean() / denom
            if not torch.isfinite(rel) or rel.item() > ratio:
                print(f"[rank {rank}] OUTPUT rel err {rel.item():.4e} > {ratio}")
                passed = False

        # (2) final state: the owner of each sequence's last token checks it.
        for seq_idx in range(N):
            seq_end = cu[seq_idx + 1]
            owner = kda_cp_owner_of_global_token(seq_end - 1, T, world_size)
            if owner != rank:
                continue
            # find this rank's local segment for that sequence (its last one).
            local_seg = None
            for li, (rq, se) in enumerate(
                zip(meta.local_req_indices_cpu, meta.local_segment_global_ends_cpu)
            ):
                if rq == seq_idx and se == seq_end:
                    local_seg = li
                    break
            assert local_seg is not None, (rank, seq_idx)
            diff = (local_state[local_seg] - ref_state[seq_idx]).abs()
            denom = ref_state[seq_idx].abs().mean().clamp_min(1e-6)
            rel = diff.mean() / denom
            if not torch.isfinite(rel) or rel.item() > ratio:
                print(f"[rank {rank}] STATE seq {seq_idx} rel err {rel.item():.4e} > {ratio}")
                passed = False

        flag = torch.tensor([1 if passed else 0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        ok_flag.value = int(flag.item())
        dist.barrier()
    finally:
        _cleanup()


def _run(world_size, lengths, H=4, D=128, dtype=torch.bfloat16, ratio=2e-2):
    ok = mp.Manager().Value("i", 0)
    mp.start_processes(
        _worker,
        args=(world_size, lengths, H, D, dtype, ratio, ok),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    return ok.value == 1


@unittest.skipIf(torch.cuda.device_count() < 2, "needs >= 2 GPUs")
class TestKDAContextParallel(unittest.TestCase):
    def test_cp2_single_sequence(self):
        self.assertTrue(_run(2, [4096]))

    def test_cp2_boundary_aligned(self):
        self.assertTrue(_run(2, [2048, 2048]))

    def test_cp2_sequence_cut(self):
        self.assertTrue(_run(2, [1500, 2000, 596]))

    def test_cp4_single_long(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("needs >= 4 GPUs")
        self.assertTrue(_run(4, [8192]))


if __name__ == "__main__":
    unittest.main()
