import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.attention.nsa.nsa_indexer as nsa
from sglang.srt.layers.attention.nsa.kpool.indexer import IndexerKPool
from sglang.srt.layers.attention.nsa.triton_kernel import fused_get_logits_head_gate_triton
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.test.ci.ci_register import register_dcu_ci

register_dcu_ci(est_time=60, suite="stage-b-test-1-gpu-small-dcu")


def make_indexer(kpool=False):
    kwargs = dict(
        hidden_size=4096, index_n_heads=32, index_head_dim=128,
        rope_head_dim=0, index_topk=2048, q_lora_rank=16,
        max_position_embeddings=1024, rope_theta=10000,
        layer_id=3, scale_fmt=None,
    )
    cls = IndexerKPool if kpool else nsa.Indexer
    if kpool:
        kwargs["config"] = SimpleNamespace(index_kpool=4)
    with (
        set_default_torch_dtype(torch.bfloat16),
        patch.object(nsa, "is_nsa_enable_prefill_cp", return_value=False),
        patch.object(nsa, "get_global_server_args",
                     return_value=SimpleNamespace(pp_size=1, device="cuda")),
    ):
        return cls(**kwargs).cuda()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None, "DCU is required"
)
class TestIndexerWeightsProjFP32(unittest.TestCase):
    @torch.inference_mode()
    def test_constructor_loader_and_projection(self):
        torch.manual_seed(908)
        for kpool in (False, True):
            indexer = make_indexer(kpool)
            weight = indexer.weights_proj.weight
            self.assertEqual(weight.dtype, torch.float32)
            checkpoint_weight = torch.randn_like(weight).to(torch.bfloat16)
            weight.weight_loader(weight, checkpoint_weight)
            torch.testing.assert_close(weight, checkpoint_weight.float(), rtol=0, atol=0)
            for dtype in (torch.bfloat16, torch.float32):
                x = torch.randn(24, 4096, device="cuda", dtype=dtype)
                actual = indexer._weights_proj_bf16_in_fp32_out(x)
                expected = F.linear(x.float(), checkpoint_weight.float())
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_kpool_head_scale_keeps_fp32(self):
        torch.manual_seed(909)
        indexer = make_indexer(kpool=True)
        indexer.weights_proj.weight.normal_(std=0.02)
        x = torch.randn(24, 4096, device="cuda", dtype=torch.bfloat16)
        projected = F.linear(x.float(), indexer.weights_proj.weight)
        q_scale = torch.rand(24, 32, 1, device="cuda")
        expected = (projected * 32**-0.5).unsqueeze(-1)
        actual = indexer._get_logits_head_gate(x, q_scale)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(
            actual, expected * q_scale * 128**-0.5, rtol=0, atol=0
        )
        torch.testing.assert_close(
            indexer._get_bf16_logits_head_gate(x),
            expected * 128**-0.5, rtol=0, atol=0,
        )
        fused = fused_get_logits_head_gate_triton(projected, q_scale, 32, 128**-0.5)
        torch.testing.assert_close(fused, actual, rtol=1e-6, atol=1e-7)

    @torch.inference_mode()
    def test_aiter_tuple_extraction(self):
        indexer = make_indexer()
        indexer.weights_proj.weight.normal_(std=0.02)
        x = torch.randn(6, 4096, device="cuda", dtype=torch.bfloat16)
        with (
            patch.object(nsa, "_use_aiter", True),
            patch.object(nsa, "_is_gfx95_supported", True),
        ):
            actual = indexer._weights_proj_bf16_in_fp32_out((None, None, x))
        torch.testing.assert_close(
            actual, F.linear(x.float(), indexer.weights_proj.weight), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
