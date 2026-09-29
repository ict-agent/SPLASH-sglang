import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.nsa.kpool.indexer import (
    IndexerKPool,
    _should_split_dcu_mqa_by_request,
)
from sglang.srt.layers.attention.nsa.kpool.planner import (
    _build_ragged_request_slices,
)
from sglang.srt.layers.attention.nsa.nsa_indexer import (
    _get_dcu_mqa_logits_max_rows,
    _run_dcu_mqa_logits_with_workspace,
)
from sglang.test.ci.ci_register import register_dcu_ci

register_dcu_ci(est_time=60, suite="stage-b-test-1-gpu-small-dcu")


@unittest.skipUnless(torch.cuda.is_available(), "DCU is required")
class TestNSAKPoolMQARequestSplit(unittest.TestCase):
    @staticmethod
    def _make_indexer(topk: int = 512) -> IndexerKPool:
        indexer = object.__new__(IndexerKPool)
        indexer.index_kpool = 4
        indexer.index_topk = topk
        return indexer

    def test_ragged_request_slice_layout(self):
        slices = _build_ragged_request_slices(
            q_starts=[0, 3, 132],
            q_lens=[3, 129, 257],
            page_starts=[0, 0, 17],
            pool_pages=[0, 17, 9],
            slots_per_page=16,
        )
        self.assertEqual(
            slices,
            (
                (0, 3, 0, 0),
                (3, 132, 0, 272),
                (132, 389, 272, 416),
            ),
        )

    def test_request_split_heuristic(self):
        self.assertFalse(
            _should_split_dcu_mqa_by_request(((0, 896, 0, 512),))
        )

        short_k_slices = tuple(
            (
                request * 256,
                (request + 1) * 256,
                request * 512,
                (request + 1) * 512,
            )
            for request in range(16)
        )
        self.assertFalse(_should_split_dcu_mqa_by_request(short_k_slices))

        long_k_slices = tuple(
            (
                request * 896,
                (request + 1) * 896,
                request * 21_760,
                (request + 1) * 21_760,
            )
            for request in range(4)
        )
        self.assertTrue(_should_split_dcu_mqa_by_request(long_k_slices))

    def test_dcu_prefill_default_keeps_bf16_query(self):
        indexer = self._make_indexer()
        indexer._get_bf16_logits_head_gate = MagicMock(
            return_value=torch.empty((3, 32, 1), device="cuda")
        )
        indexer._get_logits_head_gate = MagicMock()
        query = torch.randn((3, 32, 128), dtype=torch.bfloat16, device="cuda")
        x = torch.randn((3, 16), dtype=torch.bfloat16, device="cuda")

        with (
            patch(
                "sglang.srt.layers.attention.nsa.kpool.indexer.per_token_quant_int8"
            ) as mock_quant,
            patch(
                "sglang.srt.layers.attention.nsa.kpool.indexer._DCU_USE_INT8_MQA_LOGITS",
                False,
            ),
        ):
            q_index, weights = (
                indexer._prepare_dcu_prefill_q_and_logits_head_gate(query, x)
            )

        self.assertIs(q_index, query)
        self.assertIs(weights, indexer._get_bf16_logits_head_gate.return_value)
        mock_quant.assert_not_called()
        indexer._get_bf16_logits_head_gate.assert_called_once_with(x)
        indexer._get_logits_head_gate.assert_not_called()

    def test_dcu_int8_mqa_switch_quantizes_query(self):
        indexer = self._make_indexer()
        q_index = torch.empty((3, 32, 128), dtype=torch.int8, device="cuda")
        q_scale = torch.empty((3, 32, 1), dtype=torch.float32, device="cuda")
        weights = torch.empty((3, 32, 1), dtype=torch.float32, device="cuda")
        indexer._get_bf16_logits_head_gate = MagicMock()
        indexer._get_logits_head_gate = MagicMock(return_value=weights)
        query = torch.randn((3, 32, 128), dtype=torch.bfloat16, device="cuda")
        x = torch.randn((3, 16), dtype=torch.bfloat16, device="cuda")

        with (
            patch(
                "sglang.srt.layers.attention.nsa.kpool.indexer._DCU_USE_INT8_MQA_LOGITS",
                True,
            ),
            patch(
                "sglang.srt.layers.attention.nsa.kpool.indexer.per_token_quant_int8",
                return_value=(q_index, q_scale),
            ) as mock_quant,
        ):
            actual_q, actual_weights = (
                indexer._prepare_dcu_prefill_q_and_logits_head_gate(query, x)
            )

        self.assertIs(actual_q, q_index)
        self.assertIs(actual_weights, weights)
        mock_quant.assert_called_once_with(query)
        indexer._get_logits_head_gate.assert_called_once_with(x, q_scale)
        indexer._get_bf16_logits_head_gate.assert_not_called()

    def test_empty_history_topk(self):
        indexer = self._make_indexer(topk=2048)
        logits = torch.empty((3, 0), dtype=torch.float32, device="cuda")
        pool_lens = torch.zeros(3, dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([1, 3, 7], dtype=torch.int32, device="cuda")

        server_args = SimpleNamespace(enable_deterministic_inference=False)
        with patch(
            "sglang.srt.layers.attention.nsa.kpool.indexer.get_global_server_args",
            return_value=server_args,
        ):
            result = indexer._topk_from_kpool_logits(
                logits,
                pool_lens,
                seq_lens=seq_lens,
                allow_lightop_topk=True,
            )

        self.assertEqual(tuple(result.shape), (3, 2051))
        self.assertEqual((result >= 0).sum(dim=1).cpu().tolist(), [1, 3, 3])

    def test_request_local_bf16_logits_match_global_with_q_chunks(self):
        torch.manual_seed(20260826)
        request_slices = (
            (0, 3, 0, 0),
            (3, 132, 0, 257),
            (132, 389, 257, 386),
        )
        q = torch.randn((389, 32, 128), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((386, 128), dtype=torch.bfloat16, device="cuda")
        weights = torch.randn((389, 32), dtype=torch.float32, device="cuda")
        global_ks = torch.empty(389, dtype=torch.int32, device="cuda")
        global_ke = torch.empty_like(global_ks)
        for q_start, q_end, k_start, k_end in request_slices:
            global_ks[q_start:q_end] = k_start
            global_ke[q_start:q_end] = k_end

        global_workspace = torch.empty(
            512 * 512, dtype=torch.float32, device="cuda"
        )
        global_logits = _run_dcu_mqa_logits_with_workspace(
            q,
            k,
            weights,
            global_ks,
            global_ke,
            None,
            global_workspace,
        ).clone()

        chunk_workspace = torch.empty(
            128 * 384, dtype=torch.float32, device="cuda"
        )
        chunk_shapes = []
        for q_start, q_end, k_start, k_end in request_slices:
            request_k_rows = k_end - k_start
            if request_k_rows == 0:
                continue
            max_rows = _get_dcu_mqa_logits_max_rows(
                chunk_workspace.numel(),
                q_end - q_start,
                request_k_rows,
                q.dtype,
            )
            for start in range(q_start, q_end, max_rows):
                end = min(start + max_rows, q_end)
                rows = end - start
                local_logits = _run_dcu_mqa_logits_with_workspace(
                    q[start:end],
                    k[k_start:k_end],
                    weights[start:end],
                    torch.zeros(rows, dtype=torch.int32, device="cuda"),
                    torch.full(
                        (rows,),
                        request_k_rows,
                        dtype=torch.int32,
                        device="cuda",
                    ),
                    None,
                    chunk_workspace,
                ).clone()
                expected = global_logits[start:end, k_start:k_end]
                torch.testing.assert_close(
                    local_logits,
                    expected,
                    rtol=5e-4,
                    atol=2e-5,
                )
                chunk_shapes.append((rows, request_k_rows))

        self.assertEqual(
            chunk_shapes,
            [(128, 257), (1, 257), (128, 129), (128, 129), (1, 129)],
        )

    def test_int8_logits_match_dequantized_bf16_reference(self):
        torch.manual_seed(20260826)
        num_q, num_k, num_heads, head_dim = 128, 1024, 32, 128
        q = torch.randn(
            (num_q, num_heads, head_dim), dtype=torch.bfloat16, device="cuda"
        )
        k = torch.randn((num_k, head_dim), dtype=torch.bfloat16, device="cuda")
        weights = torch.randn((num_q, num_heads), dtype=torch.float32, device="cuda")

        q_scale = q.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127
        k_scale = k.float().abs().amax(dim=-1).clamp_min(1e-10) / 127
        q_int8 = torch.round(q.float() / q_scale).clamp(-127, 127).to(torch.int8)
        k_int8 = torch.round(k.float() / k_scale.unsqueeze(-1)).clamp(-127, 127).to(
            torch.int8
        )
        int8_weights = weights * q_scale.squeeze(-1)
        ks = torch.zeros(num_q, dtype=torch.int32, device="cuda")
        ke = torch.full((num_q,), num_k, dtype=torch.int32, device="cuda")
        workspace = torch.empty(num_q * num_k, dtype=torch.float32, device="cuda")

        int8_logits = _run_dcu_mqa_logits_with_workspace(
            q_int8,
            k_int8.unsqueeze(1),
            int8_weights,
            ks,
            ke,
            k_scale,
            workspace,
        ).clone()
        q_dequant = (q_int8.float() * q_scale).to(torch.bfloat16)
        k_dequant = (k_int8.float() * k_scale.unsqueeze(-1)).to(torch.bfloat16)
        bf16_logits = _run_dcu_mqa_logits_with_workspace(
            q_dequant,
            k_dequant.unsqueeze(1),
            weights,
            ks,
            ke,
            None,
            workspace,
        ).clone()

        abs_error = (int8_logits - bf16_logits).abs()
        self.assertLess(abs_error.mean().item(), 0.1)
        self.assertLess(abs_error.max().item(), 0.75)

    def test_request_local_topk_matches_global_ragged_and_paged_mapping(self):
        torch.manual_seed(20260826)
        indexer = self._make_indexer()
        request_slices = ((0, 2, 0, 130), (2, 5, 130, 259))
        logits = torch.full((5, 259), float("-inf"), device="cuda")
        for q_start, q_end, k_start, k_end in request_slices:
            logits[q_start:q_end, k_start:k_end] = torch.randn(
                (q_end - q_start, k_end - k_start),
                dtype=torch.float32,
                device="cuda",
            )

        pool_lens = torch.tensor([130, 130, 129, 129, 129], device="cuda")
        row_starts = torch.tensor([0, 0, 130, 130, 130], device="cuda")
        topk_offsets = torch.tensor([1000, 1000, 2000, 2000, 2000], device="cuda")
        page_table_row_index = torch.tensor([0, 0, 1, 1, 1], device="cuda")
        page_table = torch.arange(2 * 1024, dtype=torch.int32, device="cuda").view(
            2, 1024
        )
        server_args = SimpleNamespace(enable_deterministic_inference=True)

        with patch(
            "sglang.srt.layers.attention.nsa.kpool.indexer.get_global_server_args",
            return_value=server_args,
        ):
            global_ragged = indexer._topk_from_kpool_logits(
                logits,
                pool_lens,
                topk_offsets=topk_offsets,
                row_starts=row_starts,
            )
            global_paged = indexer._topk_from_kpool_logits(
                logits,
                pool_lens,
                page_table=page_table,
                row_starts=row_starts,
                page_table_row_index=page_table_row_index,
            )

            local_ragged = torch.empty_like(global_ragged)
            local_paged = torch.empty_like(global_paged)
            for q_start, q_end, k_start, k_end in request_slices:
                local_logits = logits[q_start:q_end, k_start:k_end]
                local_ragged[q_start:q_end] = indexer._topk_from_kpool_logits(
                    local_logits,
                    pool_lens[q_start:q_end],
                    topk_offsets=topk_offsets[q_start:q_end],
                )
                local_paged[q_start:q_end] = indexer._topk_from_kpool_logits(
                    local_logits,
                    pool_lens[q_start:q_end],
                    page_table=page_table,
                    page_table_row_index=page_table_row_index[q_start:q_end],
                )

        self.assertTrue(torch.equal(local_ragged, global_ragged))
        self.assertTrue(torch.equal(local_paged, global_paged))


if __name__ == "__main__":
    unittest.main()
