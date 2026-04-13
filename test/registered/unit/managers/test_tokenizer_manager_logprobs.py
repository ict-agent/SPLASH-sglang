import unittest

import torch

from sglang.srt.managers.tokenizer_manager import TokenizerManager


class TestTokenizerManagerLogprobs(unittest.TestCase):
    def test_detokenize_top_logprobs_tokens_accepts_tensor_entries(self):
        manager = TokenizerManager.__new__(TokenizerManager)

        ret = TokenizerManager.detokenize_top_logprobs_tokens(
            manager,
            [torch.tensor([-0.1, -0.2], dtype=torch.float32), []],
            [torch.tensor([10, 11], dtype=torch.int64), []],
            decode_to_text=False,
        )

        self.assertEqual(ret[0], [(-0.1, 10, None), (-0.2, 11, None)])
        self.assertIsNone(ret[1])


if __name__ == "__main__":
    unittest.main()
