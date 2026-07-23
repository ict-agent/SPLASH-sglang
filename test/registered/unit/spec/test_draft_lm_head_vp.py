"""Unit tests for node-local EAGLE draft LM-head vocabulary parallelism."""

import unittest

import torch

from sglang.srt.speculative.draft_lm_head_vp import (
    build_node_local_vp_groups,
    select_global_top1_from_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestDraftLMHeadVocabParallelTop1(CustomTestCase):
    def test_builds_two_node_local_vp8_groups(self):
        groups = build_node_local_vp_groups(
            list(range(16)),
            vp_size=8,
            local_world_size=8,
        )
        self.assertEqual(groups, [list(range(8)), list(range(8, 16))])

    def test_rejects_cross_node_group(self):
        with self.assertRaisesRegex(ValueError, "crosses node boundaries"):
            build_node_local_vp_groups(
                list(range(16)),
                vp_size=16,
                local_world_size=8,
            )

    def test_selects_full_vocab_top1_with_lowest_id_tie_break(self):
        # Candidate layout: [vocab shard, hidden row, (score, local token id)].
        candidates = torch.tensor(
            [
                [[1.0, 3.0], [5.0, 3.0]],
                [[2.0, 0.0], [5.0, 0.0]],
                [[1.5, 1.0], [4.0, 2.0]],
            ],
            dtype=torch.float32,
        )

        scores, token_ids = select_global_top1_from_candidates(
            candidates,
            shard_size=4,
        )

        torch.testing.assert_close(scores, torch.tensor([2.0, 5.0]))
        # Row 0 comes from shard 1: 1 * 4 + 0 = 4.
        # Row 1 ties across shards 0 and 1, so the lower global id 3 wins.
        torch.testing.assert_close(token_ids, torch.tensor([4, 3]))


if __name__ == "__main__":
    unittest.main()
