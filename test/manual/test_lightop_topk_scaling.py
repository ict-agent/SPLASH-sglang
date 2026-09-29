"""Regression for the GLM5 LightOP gate/runner scaling contract.

Run on a DCU with SGLANG_USE_LIGHTOP=1 and SGLANG_USE_AITER=0:
    python -m pytest -q test/manual/test_lightop_topk_scaling.py
"""

import sys

import pytest
import torch

from sglang.srt.layers.moe import topk as topk_module

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not topk_module._use_lightop
    or topk_module._use_aiter,
    reason="requires a DCU with SGLANG_USE_LIGHTOP=1 and SGLANG_USE_AITER=0",
)


@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_glm5_routed_scale_is_applied_once(tokens, dtype):
    from lightop import op

    torch.manual_seed(29)
    logits = torch.randn(tokens, 288, device="cuda", dtype=dtype)
    bias = torch.randn(288, device="cuda", dtype=torch.float32) * 0.1
    hidden_states = torch.empty(tokens, 4096, device="cuda", dtype=torch.bfloat16)
    scale = 2.5

    weights, ids = topk_module.biased_grouped_topk_gpu(
        hidden_states,
        logits,
        bias,
        topk=8,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        num_fused_shared_experts=0,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=False,
    )
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(tokens, device="cuda"), rtol=2e-6, atol=2e-6
    )

    # Model each selected expert as returning ones. The runner's reduction
    # must produce scale (2.5), not scale**2 (6.25).
    weighted_expert_outputs = (
        weights.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]).to(torch.bfloat16)
    ).contiguous()
    output = torch.empty_like(hidden_states)
    op.moe_sum(input=weighted_expert_outputs, output=output, factor=scale)
    torch.testing.assert_close(
        output, torch.full_like(output, scale), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize("apply_scale", [False, True])
def test_custom_op_forwards_the_explicit_scale_flag(apply_scale):
    logits = torch.zeros(1, 256, device="cuda", dtype=torch.float32)
    bias = torch.zeros(256, device="cuda", dtype=torch.float32)
    scale = 2.826
    weights, _ = torch.ops.sglang.moe_fused_gate_dcu(
        logits, bias, 1, 1, 8, 0, scale, apply_scale
    )
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.full((1,), scale if apply_scale else 1.0, device="cuda"),
        rtol=2e-6,
        atol=2e-6,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
