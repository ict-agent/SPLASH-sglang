"""Compare FlashKDAKernel.extend against TritonKDAKernel.extend.

FlashKDAKernel applies the KDA gate activation
(``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``) and ``sigmoid(beta)``
inside the kernel and consumes RAW g / beta. TritonKDAKernel expects them
already-activated, so we mirror what ``KDAAttnBackend.forward_extend`` does
(call ``fused_kda_gate`` first) before invoking it. Both kernels write the
final per-batch state back to ``ssm_states[cache_indices]``; we compare both
the per-token output and that final state.

Tolerances: FlashKDA's bf16 chunk=64 recurrence vs Triton's chunk_kda
chunk=64 path drift by ~1 bf16 LSB per token along the time axis, which
accumulates into ~1e-2 over a ~128-token sequence and grows roughly with
sqrt(T). atol=2e-2 / state atol=5e-2 covers up to a few-thousand-token seq;
matches what we observe end-to-end (see
feedback_flashkda_moe_routing_floor.md). FlashKDA's own test_fwd uses
atol=0.005 against the Triton reference too.

Axis coverage (8 cases):
    production:  (B,T,H,D) ∈ {(4,128,8,128), (4,512,8,128), (2,2048,16,128)}
    batch sweep: B ∈ {1, 8} at (T=128, H=8, D=128)
    heads:       H ∈ {1, 32} at (B=4, T=128, D=128)
    varlen:      seq_lens=[131, 547, 1024, 271]       # non-chunk-aligned mix

Note: D is fixed at 128 because FlashKDA's CP preprocess
(triton_cp_compute_ht_mt → state_only_fwd) raises
``RuntimeError: currently only supports D == 128``. Head_dim sweep is
therefore out of scope for this equivalence test.
"""

import unittest
from typing import List, Union

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=130, suite="stage-b-test-1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFlashKDAExtendVsTriton(unittest.TestCase):
    OUT_ATOL = 2e-2
    OUT_RTOL = 1e-2
    STATE_ATOL = 5e-2
    STATE_RTOL = 1e-2
    SAFE_GATE_LOWER_BOUND = -5.0

    def _build_inputs(
        self,
        seq_lens: List[int],
        H: int,
        D: int,
        pool_size: int,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        device = "cuda"

        T_total = sum(seq_lens)
        B = len(seq_lens)

        cu = [0]
        for sl in seq_lens:
            cu.append(cu[-1] + sl)
        query_start_loc = torch.tensor(cu, dtype=torch.int32, device=device)
        seq_lens_cpu = list(seq_lens)

        # Pick B distinct slots from the cache pool to exercise gather/scatter.
        perm = torch.randperm(pool_size, device=device)[:B]
        cache_indices = perm.to(torch.int32)

        A_log = torch.randn(1, 1, H, 1, dtype=torch.float32, device=device)
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)

        q = torch.randn(1, T_total, H, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, T_total, H, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, T_total, H, D, dtype=torch.bfloat16, device=device)
        raw_g = torch.randn(
            1, T_total, H * D, dtype=torch.float32, device=device
        )
        raw_beta = torch.randn(1, T_total, H, dtype=torch.float32, device=device)

        # Non-zero initial state to make sure the kernel actually picks it up.
        ssm_states_init = (
            torch.randn(pool_size, H, D, D, dtype=torch.float32, device=device)
            * 0.05
        )

        return dict(
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            cache_indices=cache_indices,
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            raw_beta=raw_beta,
            ssm_states_init=ssm_states_init,
            H=H,
            D=D,
        )

    def _run_flash(self, ctx):
        from sglang.srt.layers.attention.linear.kernels.kda_flash import (
            FlashKDAKernel,
        )

        ssm_states = ctx["ssm_states_init"].clone()
        out, _ = FlashKDAKernel().extend(
            q=ctx["q"],
            k=ctx["k"],
            v=ctx["v"],
            g=ctx["raw_g"],
            beta=ctx["raw_beta"],
            ssm_states=ssm_states,
            cache_indices=ctx["cache_indices"],
            query_start_loc=ctx["query_start_loc"],
            A_log=ctx["A_log"],
            dt_bias=ctx["dt_bias"],
            seq_lens_cpu=ctx["seq_lens_cpu"],
            safe_gate_lower_bound=self.SAFE_GATE_LOWER_BOUND,
        )
        return out, ssm_states[ctx["cache_indices"]].clone()

    def _run_triton(self, ctx):
        from sglang.srt.layers.attention.fla.kda import fused_kda_gate
        from sglang.srt.layers.attention.linear.kernels.kda_triton import (
            TritonKDAKernel,
        )

        g_act, b_sig = fused_kda_gate(
            ctx["raw_g"],
            ctx["A_log"],
            ctx["D"],
            g_bias=ctx["dt_bias"],
            safe_gate=True,
            lower_bound=self.SAFE_GATE_LOWER_BOUND,
            beta=ctx["raw_beta"],
            beta_scale=1.0,
        )
        ssm_states = ctx["ssm_states_init"].clone()
        out, _ = TritonKDAKernel().extend(
            q=ctx["q"],
            k=ctx["k"],
            v=ctx["v"],
            g=g_act,
            beta=b_sig,
            ssm_states=ssm_states,
            cache_indices=ctx["cache_indices"],
            query_start_loc=ctx["query_start_loc"],
        )
        return out, ssm_states[ctx["cache_indices"]].clone()

    def _check(
        self,
        *,
        seq_lens: Union[int, List[int]],
        B: int = 1,
        H: int,
        D: int,
        pool_size: int = 16,
        seed: int = 0,
    ):
        # Accept either a single int (uniform B*T) or a per-sequence list.
        if isinstance(seq_lens, int):
            seq_list = [seq_lens] * B
        else:
            seq_list = list(seq_lens)
        # D must be a multiple of 64 (FlashKDA static_assert).
        self.assertEqual(D % 64, 0)

        ctx = self._build_inputs(
            seq_list, H=H, D=D, pool_size=pool_size, seed=seed
        )

        out_f, st_f = self._run_flash(ctx)
        out_t, st_t = self._run_triton(ctx)

        self.assertEqual(out_f.shape, out_t.shape)
        self.assertEqual(st_f.shape, st_t.shape)

        diff_o = (out_f.float() - out_t.float()).abs()
        diff_s = (st_f.float() - st_t.float()).abs()
        print(
            f"[B={len(seq_list)}, T_total={sum(seq_list)}, H={H}, D={D}] "
            f"out max={diff_o.max().item():.3e} mean={diff_o.mean().item():.3e}; "
            f"state max={diff_s.max().item():.3e} mean={diff_s.mean().item():.3e}"
        )

        self.assertTrue(
            torch.allclose(
                out_f.float(), out_t.float(),
                atol=self.OUT_ATOL, rtol=self.OUT_RTOL,
            ),
            f"output mismatch: max_diff={diff_o.max().item():.3e}",
        )
        self.assertTrue(
            torch.allclose(
                st_f, st_t,
                atol=self.STATE_ATOL, rtol=self.STATE_RTOL,
            ),
            f"state mismatch: max_diff={diff_s.max().item():.3e}",
        )

    # ------------------------------------------------------------------
    # Production-style configs (GLM5-Next: H=8/16, D=128)
    # ------------------------------------------------------------------
    def test_production_short(self):
        self._check(B=4, seq_lens=128, H=8, D=128)

    def test_production_medium(self):
        self._check(B=4, seq_lens=512, H=8, D=128)

    def test_production_long(self):
        self._check(B=2, seq_lens=2048, H=16, D=128, pool_size=8)

    # ------------------------------------------------------------------
    # Batch sweep
    # ------------------------------------------------------------------
    def test_batch_1(self):
        self._check(B=1, seq_lens=128, H=8, D=128)

    def test_batch_8(self):
        self._check(B=8, seq_lens=128, H=8, D=128, pool_size=32)

    # ------------------------------------------------------------------
    # Head count corners
    # ------------------------------------------------------------------
    def test_heads_1(self):
        self._check(B=4, seq_lens=128, H=1, D=128)

    def test_heads_32(self):
        self._check(B=4, seq_lens=128, H=32, D=128)

    # ------------------------------------------------------------------
    # Varlen — chunk-aligned and non-chunk-aligned mix (FlashKDA's own
    # test_fwd_varlen uses similar irregular layouts).
    # ------------------------------------------------------------------
    def test_varlen_mixed(self):
        self._check(seq_lens=[131, 547, 1024, 271], H=8, D=128)


if __name__ == "__main__":
    unittest.main()
