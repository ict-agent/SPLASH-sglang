import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.test_utils import CustomTestCase

register_dcu_ci(est_time=5, suite="stage-b-test-1-gpu-small-dcu")


CKPT_LAYER_PREFIX = "model.layers.45"
RUNTIME_LAYER_PREFIX = "model.decoder"
CKPT_KV_B_PROJ = f"{CKPT_LAYER_PREFIX}.self_attn.kv_b_proj"
RUNTIME_KV_B_PROJ = f"{RUNTIME_LAYER_PREFIX}.self_attn.kv_b_proj"


def make_nextn_mapper():
    from sglang.srt.models.utils import WeightsMapper

    return WeightsMapper(
        orig_to_new_substr={
            CKPT_LAYER_PREFIX: RUNTIME_LAYER_PREFIX,
            f"{CKPT_LAYER_PREFIX}.eh_proj": "model.eh_proj",
            f"{CKPT_LAYER_PREFIX}.enorm": "model.enorm",
            f"{CKPT_LAYER_PREFIX}.hnorm": "model.hnorm",
            f"{CKPT_LAYER_PREFIX}.shared_head.norm": "model.shared_head.norm",
        }
    )


class TestW8A8Int8NextNWeightNameMapping(CustomTestCase):
    def test_ignore_mapping_is_deduplicating_and_idempotent(self):
        from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

        quant_config = W8A8Int8Config(
            {
                "ignore": [
                    CKPT_KV_B_PROJ,
                    CKPT_KV_B_PROJ,
                    f"{CKPT_LAYER_PREFIX}.self_attn.indexer.wk",
                    f"{CKPT_LAYER_PREFIX}.eh_proj",
                    f"{CKPT_LAYER_PREFIX}.shared_head.norm",
                ]
            }
        )
        mapper = make_nextn_mapper()

        quant_config.apply_weight_name_mapper(mapper)
        expected = [
            RUNTIME_KV_B_PROJ,
            f"{RUNTIME_LAYER_PREFIX}.self_attn.indexer.wk",
            "model.eh_proj",
            "model.shared_head.norm",
        ]
        self.assertEqual(quant_config.ignore, expected)

        quant_config.apply_weight_name_mapper(mapper)
        self.assertEqual(quant_config.ignore, expected)

    def test_layer45_kv_b_proj_is_unquantized_before_nextn_model_build(self):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.quantization.w8a8_int8 import (
            W8A8Int8Config,
            W8A8Int8LinearMethod,
        )
        from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN

        quant_config = W8A8Int8Config(
            {
                "ignore": [
                    CKPT_KV_B_PROJ,
                    f"{CKPT_LAYER_PREFIX}.eh_proj",
                ]
            }
        )
        config = SimpleNamespace(
            num_hidden_layers=45,
            num_nextn_predict_layers=1,
            vocab_size=128,
            hidden_size=16,
        )

        def build_nextn_model(_config, received_quant_config, prefix):
            self.assertIs(received_quant_config, quant_config)
            self.assertEqual(prefix, "model")
            self.assertEqual(
                received_quant_config.ignore,
                [RUNTIME_KV_B_PROJ, "model.eh_proj"],
            )
            return nn.Identity()

        class FakeDeepseekModelNextN(nn.Identity):
            @staticmethod
            def _get_nextn_layer_name(_config):
                return "decoder"

            def __new__(cls, _config, received_quant_config, prefix):
                return build_nextn_model(_config, received_quant_config, prefix)

        module = "sglang.srt.models.deepseek_nextn"
        server_args = SimpleNamespace(
            enable_dp_lm_head=False,
            model_path="/target",
            speculative_draft_model_path="/draft",
        )
        with (
            patch.object(
                DeepseekV3ForCausalLMNextN,
                "determine_num_fused_shared_experts",
            ),
            patch(f"{module}.get_tensor_model_parallel_world_size", return_value=1),
            patch(f"{module}.get_pp_group", return_value=SimpleNamespace()),
            patch(f"{module}.is_deepseek_nsa", return_value=False),
            patch(f"{module}.is_nsa_enable_prefill_cp", return_value=False),
            patch(f"{module}.DeepseekModelNextN", new=FakeDeepseekModelNextN),
            patch(f"{module}.ParallelLMHead", return_value=nn.Identity()),
            patch(f"{module}.LogitsProcessor", return_value=nn.Identity()),
            patch(f"{module}.get_global_server_args", return_value=server_args),
        ):
            model = DeepseekV3ForCausalLMNextN(config, quant_config)

        self.assertIs(model.quant_config, quant_config)

        dummy_linear = LinearBase(input_size=1, output_size=1)
        method = quant_config.get_quant_method(dummy_linear, RUNTIME_KV_B_PROJ)
        self.assertIsInstance(method, UnquantizedLinearMethod)

        method = quant_config.get_quant_method(
            dummy_linear, f"{RUNTIME_LAYER_PREFIX}.self_attn.q_b_proj"
        )
        self.assertIsInstance(method, W8A8Int8LinearMethod)


if __name__ == "__main__":
    unittest.main()
