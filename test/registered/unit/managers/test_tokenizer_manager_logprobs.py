import unittest

import torch

from sglang.srt.managers.tokenizer_manager import TokenizerManager


class TestTokenizerManagerLogprobs(unittest.TestCase):
    def test_detokenize_top_logprobs_tokens_accepts_tensor_entries(self):
        manager = TokenizerManager.__new__(TokenizerManager)

        ret = TokenizerManager.detokenize_top_logprobs_tokens(
            manager,
            # -0.5/-0.25 are exactly representable in float32, so the
            # tolist() round-trip preserves equality for assertEqual.
            [torch.tensor([-0.5, -0.25], dtype=torch.float32), []],
            [torch.tensor([10, 11], dtype=torch.int64), []],
            decode_to_text=False,
        )

        self.assertEqual(ret[0], [(-0.5, 10, None), (-0.25, 11, None)])
        self.assertIsNone(ret[1])


if __name__ == "__main__":
    unittest.main()
