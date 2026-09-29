import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import dp_attention
from sglang.srt.layers.mla_only_dp_transfer import (
    _get_transfer_token_counts,
    _pack_dp_rows_to_tp_shards_torch,
    _pack_two_tp_head_shards_torch,
    _use_fused_q_transfer,
    _use_pair_transfer_kernel,
    _unpack_two_tp_head_shards_torch,
    _unpack_tp_head_shards_torch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMlaOnlyDpTransferLayout(unittest.TestCase):
    def test_pair_transfer_gate_uses_per_rank_rows(self):
        class FakeTensor:
            is_cuda = True
            device = "cuda:0"
            dtype = torch.bfloat16
            shape = (32768, 16, 512)

            def stride(self, dim):
                return 1

        fake_q = FakeTensor()
        fake_q_pe = FakeTensor()
        fake_q_pe.shape = (32768, 16, 64)

        patches = [
            patch(
                "sglang.srt.layers.mla_only_dp_transfer._transfer_kernel_is_available",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.mla_only_dp_transfer._pack_two_tp_head_shards_kernel",
                object(),
            ),
            patch(
                "sglang.srt.layers.mla_only_dp_transfer._unpack_two_tp_head_shards_kernel",
                object(),
            ),
            patch.dict(
                "os.environ",
                {
                    "SGLANG_MLA_ONLY_DP_FUSED_Q_TRANSFER_MAX_ROWS": "32",
                    "SGLANG_MLA_ONLY_DP_FUSED_Q_TRANSFER_MIN_ROWS": "512",
                },
            ),
        ]
        with patches[0], patches[1], patches[2], patches[3]:
            self.assertTrue(_use_pair_transfer_kernel(fake_q, fake_q_pe))
            self.assertTrue(_use_fused_q_transfer([32] * 8, fake_q, fake_q_pe))
            self.assertFalse(_use_fused_q_transfer([128] * 8, fake_q, fake_q_pe))
            self.assertTrue(_use_fused_q_transfer([512] * 8, fake_q, fake_q_pe))

    def test_transfer_token_counts_require_aligned_tp_and_dp(self):
        batch = SimpleNamespace(global_num_tokens_cpu=[3, 0, 5])
        parallel = SimpleNamespace(tp_rank=1, attn_dp_rank=1)
        with patch(
            "sglang.srt.layers.mla_only_dp_transfer.get_parallel",
            return_value=parallel,
        ):
            self.assertEqual(_get_transfer_token_counts(batch, 3), [3, 0, 5])
            with self.assertRaises(RuntimeError):
                _get_transfer_token_counts(batch, 2)

        parallel.attn_dp_rank = 2
        with patch(
            "sglang.srt.layers.mla_only_dp_transfer.get_parallel",
            return_value=parallel,
        ):
            with self.assertRaises(RuntimeError):
                _get_transfer_token_counts(batch, 3)

    def test_pack_unpack_round_trip(self):
        tp_size = 3
        local_rows = 5
        local_heads = 2
        head_dim = 4
        full_heads = torch.arange(
            local_rows * tp_size * local_heads * head_dim, dtype=torch.float32
        ).reshape(local_rows, tp_size * local_heads, head_dim)

        packed = torch.empty(tp_size, local_rows, local_heads, head_dim)
        _pack_dp_rows_to_tp_shards_torch(full_heads, packed)

        unpacked = torch.empty_like(full_heads)
        _unpack_tp_head_shards_torch(packed, unpacked)
        torch.testing.assert_close(unpacked, full_heads)

    def test_zero_row_rank_participates(self):
        tp_size = 8
        local_heads = 1
        head_dim = 16
        full_heads = torch.empty(0, tp_size * local_heads, head_dim)
        packed = torch.empty(tp_size, 0, local_heads, head_dim)

        _pack_dp_rows_to_tp_shards_torch(full_heads, packed)
        unpacked = torch.empty_like(full_heads)
        _unpack_tp_head_shards_torch(packed, unpacked)

        self.assertEqual(packed.numel(), 0)
        self.assertEqual(unpacked.shape, full_heads.shape)

    def test_fused_q_nope_q_pe_pack_unpack_round_trip(self):
        tp_size = 3
        local_rows = 5
        local_heads = 2
        dim_nope = 7
        dim_pe = 3
        fused = torch.arange(
            tp_size * local_rows * local_heads * (dim_nope + dim_pe),
            dtype=torch.float32,
        ).reshape(tp_size, local_rows, local_heads, dim_nope + dim_pe)

        out_nope = torch.empty(local_rows, tp_size * local_heads, dim_nope)
        out_pe = torch.empty(local_rows, tp_size * local_heads, dim_pe)
        _unpack_two_tp_head_shards_torch(fused, out_nope, out_pe, dim_nope)

        expected = fused.permute(1, 0, 2, 3).reshape(
            local_rows, tp_size * local_heads, dim_nope + dim_pe
        )
        torch.testing.assert_close(out_nope, expected[..., :dim_nope])
        torch.testing.assert_close(out_pe, expected[..., dim_nope:])

        packed = torch.empty(
            local_rows, tp_size * local_heads, dim_nope + dim_pe
        )
        _pack_two_tp_head_shards_torch(out_nope, out_pe, packed)
        torch.testing.assert_close(packed, expected)

    def test_mla_only_dp_enables_gatherv_by_default(self):
        parallel = SimpleNamespace(mla_only_dp=True)
        patches = [
            patch.dict(os.environ, {}, clear=True),
            patch(
                "sglang.srt.layers.dp_attention.world_dp_gather_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attn_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_tensor_model_parallel_world_size",
                return_value=8,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_dp_size",
                return_value=8,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.dp_attention._DpGatheredBufferWrapper.is_dp_max_padding",
                return_value=False,
            ),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            self.assertTrue(dp_attention.is_dp_gatherv_active())

    def test_mla_only_dp_gatherv_env_override_wins(self):
        parallel = SimpleNamespace(mla_only_dp=True)
        patches = [
            patch.dict(os.environ, {"SGLANG_DP_USE_GATHERV": "0"}, clear=True),
            patch(
                "sglang.srt.layers.dp_attention.world_dp_gather_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attn_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_tensor_model_parallel_world_size",
                return_value=8,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_dp_size",
                return_value=8,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.dp_attention._DpGatheredBufferWrapper.is_dp_max_padding",
                return_value=False,
            ),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            self.assertFalse(dp_attention.is_dp_gatherv_active())


if __name__ == "__main__":
    unittest.main()
