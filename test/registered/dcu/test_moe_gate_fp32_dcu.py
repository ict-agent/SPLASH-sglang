import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.model_executor.model_runner import _model_load_weights_direct
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_dcu_ci

register_dcu_ci(est_time=60, suite="stage-b-test-1-gpu-small-dcu")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None, "DCU is required"
)
class TestMoEGateFP32(unittest.TestCase):
    @staticmethod
    def make_gate(router_dtype="float32", is_nextn=False):
        config = SimpleNamespace(
            n_routed_experts=288,
            hidden_size=4096,
            topk_method="noaux_tc",
        )
        if router_dtype is not None:
            config.router_dtype = router_dtype
        with (
            set_default_torch_dtype(torch.bfloat16),
            patch.dict(os.environ, {"SGLANG_MOE_ROUTER_USE_CONFIG_DTYPE": "1"}),
            patch.object(deepseek_v2, "is_nsa_enable_prefill_cp", return_value=False),
        ):
            gate = deepseek_v2.MoEGate(config, None, is_nextn=is_nextn).cuda()
        with torch.no_grad():
            gate.weight.normal_(std=0.02)
        return gate

    @torch.inference_mode()
    def test_fp32_projection_and_checkpoint_loading(self):
        torch.manual_seed(908)
        for router_dtype in ("float32", "fp32"):
            for checkpoint_dtype in (torch.bfloat16, torch.float32):
                with self.subTest(
                    router_dtype=router_dtype, checkpoint_dtype=checkpoint_dtype
                ):
                    gate = self.make_gate(router_dtype)
                    checkpoint = torch.randn(
                        288, 4096, device="cuda", dtype=checkpoint_dtype
                    ) * 0.02
                    default_weight_loader(gate.weight, checkpoint)
                    self.assertEqual(gate.weight.dtype, torch.float32)
                    torch.testing.assert_close(
                        gate.weight, checkpoint.float(), rtol=0, atol=0
                    )
                    x = torch.randn(24, 4096, device="cuda", dtype=torch.bfloat16)
                    expected = F.linear(x.float(), checkpoint.float())
                    output = gate(x)
                    self.assertEqual(output.dtype, torch.float32)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    self.assertNotIn("_weight_fp32", gate.state_dict())

    @torch.inference_mode()
    def test_weight_updates_after_forward(self):
        torch.manual_seed(909)
        for is_nextn in (False, True):
            for update_dtype in (torch.bfloat16, torch.float32):
                with self.subTest(is_nextn=is_nextn, update_dtype=update_dtype):
                    gate = self.make_gate(is_nextn=is_nextn)
                    x = torch.randn(24, 4096, device="cuda", dtype=torch.bfloat16)
                    original_output = gate(x).clone()
                    update = torch.randn_like(gate.weight, dtype=update_dtype) * 0.02
                    # Exercise the direct tensor update helper used by ModelRunner.
                    _model_load_weights_direct(gate, [("weight", update)])
                    output = gate(x)
                    self.assertEqual(gate.weight.dtype, torch.float32)
                    torch.testing.assert_close(
                        output, F.linear(x.float(), update.float()), rtol=0, atol=0
                    )
                    self.assertFalse(torch.equal(output, original_output))
                    _model_load_weights_direct(
                        gate, [("weight", torch.zeros_like(update))]
                    )
                    torch.testing.assert_close(
                        gate(x), torch.zeros_like(output), rtol=0, atol=0
                    )

    @torch.inference_mode()
    def test_fp32_main_and_mtp_across_forward_modes(self):
        for is_nextn in (False, True):
            for deterministic, prefill_cp in ((False, False), (True, False), (False, True)):
                with self.subTest(
                    is_nextn=is_nextn, deterministic=deterministic, prefill_cp=prefill_cp
                ):
                    gate = self.make_gate(is_nextn=is_nextn)
                    x = torch.randn(6, 4096, device="cuda", dtype=torch.bfloat16)
                    args = SimpleNamespace(enable_deterministic_inference=deterministic)
                    with (
                        patch.object(deepseek_v2, "get_global_server_args", return_value=args),
                        patch.object(deepseek_v2, "nsa_use_prefill_cp", return_value=prefill_cp),
                    ):
                        output = gate(x, forward_batch=object())
                    self.assertEqual(output.dtype, torch.float32)
                    torch.testing.assert_close(
                        output, F.linear(x.float(), gate.weight.float()), rtol=0, atol=0
                    )

    @torch.inference_mode()
    def test_default_bf16_ignores_fp32_config(self):
        for is_nextn in (False, True):
            for enabled_value in (None, "0"):
                with self.subTest(is_nextn=is_nextn, enabled_value=enabled_value):
                    with patch.dict(os.environ):
                        key = "SGLANG_MOE_ROUTER_USE_CONFIG_DTYPE"
                        if enabled_value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = enabled_value
                        config = SimpleNamespace(
                            n_routed_experts=288, hidden_size=4096,
                            topk_method="noaux_tc", router_dtype="float32",
                        )
                        with (
                            set_default_torch_dtype(torch.float32),
                            patch.object(deepseek_v2, "is_nsa_enable_prefill_cp", return_value=False),
                        ):
                            gate = deepseek_v2.MoEGate(config, None, is_nextn=is_nextn).cuda()
                        gate.weight.normal_(std=0.02)
                    self.assertFalse(gate._use_fp32_router)
                    self.assertEqual(gate.weight.dtype, torch.bfloat16)
                    for input_dtype in (torch.bfloat16, torch.float32):
                        x = torch.randn(6, 4096, device="cuda", dtype=input_dtype)
                        args = SimpleNamespace(enable_deterministic_inference=True)
                        with patch.object(deepseek_v2, "get_global_server_args", return_value=args):
                            output = gate(x)
                        self.assertEqual(output.dtype, torch.bfloat16)
                        torch.testing.assert_close(
                            output, F.linear(x.bfloat16(), gate.weight), rtol=0, atol=0
                        )

    @torch.inference_mode()
    def test_legacy_router_dtype_is_preserved(self):
        for router_dtype in (None, "bfloat16"):
            with self.subTest(router_dtype=router_dtype):
                gate = self.make_gate(router_dtype)
                x = torch.randn(6, 4096, device="cuda", dtype=torch.bfloat16)
                args = SimpleNamespace(enable_deterministic_inference=True)
                with patch.object(deepseek_v2, "get_global_server_args", return_value=args):
                    output = gate(x)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(gate.weight.dtype, torch.bfloat16)
                torch.testing.assert_close(
                    output, F.linear(x, gate.weight), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
