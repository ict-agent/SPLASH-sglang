"""Unit tests for GLM4V mrope token-type grouping."""

import itertools
import random
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang.srt.layers.rotary_embedding.mrope_rope_index import (  # noqa: E402
    _glm4v_token_type_groups,
    get_rope_index_glm4v,
)

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

IMAGE = 154854
VIDEO_START = 154832
VIDEO_END = 154833


def reference_groups(ids):
    """Original per-token implementation, retained as a test oracle."""
    token_types = []
    in_video = False
    for token in ids.tolist():
        if token == VIDEO_START:
            in_video = True
        elif token == VIDEO_END:
            in_video = False

        if token == IMAGE:
            token_types.append("video" if in_video else "image")
        else:
            token_types.append("text")

    groups = []
    for name, group in itertools.groupby(enumerate(token_types), lambda item: item[1]):
        group = list(group)
        groups.append((name, group[0][0], group[-1][0] + 1))
    return groups


class TestGlm4vTokenTypeGroups(CustomTestCase):
    def assert_matches_reference(self, tokens):
        ids = torch.tensor(tokens, dtype=torch.long)
        self.assertEqual(
            _glm4v_token_type_groups(ids, IMAGE, VIDEO_START, VIDEO_END),
            reference_groups(ids),
        )

    def test_empty_and_single_token_sequences(self):
        for tokens in ([], [1], [IMAGE], [VIDEO_START], [VIDEO_END]):
            with self.subTest(tokens=tokens):
                self.assert_matches_reference(tokens)

    def test_image_and_video_boundaries(self):
        cases = [
            [IMAGE, IMAGE, 1, 2],
            [1, 2, IMAGE, IMAGE],
            [IMAGE, 1, IMAGE],
            [1, VIDEO_START, IMAGE, IMAGE, VIDEO_END, IMAGE, 2],
        ]
        for tokens in cases:
            with self.subTest(tokens=tokens):
                self.assert_matches_reference(tokens)

    def test_malformed_markers_preserve_last_writer_wins(self):
        cases = [
            [VIDEO_START, VIDEO_START, VIDEO_END, IMAGE],
            [VIDEO_END, IMAGE, 1],
            [VIDEO_START, IMAGE, 1, IMAGE],
            [VIDEO_START, IMAGE, VIDEO_END, VIDEO_END, IMAGE],
            [VIDEO_START, VIDEO_END, VIDEO_START, IMAGE],
        ]
        for tokens in cases:
            with self.subTest(tokens=tokens):
                self.assert_matches_reference(tokens)

    def test_random_sequences(self):
        rng = random.Random(1234)
        specials = [IMAGE, VIDEO_START, VIDEO_END]
        for trial in range(200):
            tokens = [
                rng.choice(specials) if rng.random() < 0.35 else rng.randint(1, 1000)
                for _ in range(rng.randint(1, 200))
            ]
            with self.subTest(trial=trial):
                self.assert_matches_reference(tokens)

    def test_get_rope_index_integration(self):
        config = types.SimpleNamespace(
            image_token_id=IMAGE,
            video_start_token_id=VIDEO_START,
            video_end_token_id=VIDEO_END,
            vision_config=types.SimpleNamespace(spatial_merge_size=2),
        )
        positions, deltas = get_rope_index_glm4v(
            torch.tensor([[1, IMAGE, IMAGE, 2]], dtype=torch.long),
            config,
            image_grid_thw=[[1, 2, 4]],
            video_grid_thw=[],
            attention_mask=None,
        )
        expected = torch.tensor(
            [[[0, 1, 1, 3]], [[0, 1, 1, 3]], [[0, 1, 2, 3]]],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(positions, expected))
        self.assertTrue(torch.equal(deltas, torch.tensor([[0]])))


if __name__ == "__main__":
    unittest.main()
