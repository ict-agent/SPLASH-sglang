"""Unit tests for async tokenizer queue and execution timing."""

import asyncio
import contextlib

from sglang.srt.managers.async_dynamic_batch_tokenizer import (
    AsyncDynamicbatchTokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, list):
            return {"input_ids": [[len(prompt)] for prompt in prompts]}
        return {"input_ids": [len(prompts)]}


class TestAsyncTokenizerTiming(CustomTestCase):
    def test_queue_and_execution_timestamps_are_ordered(self):
        async def run_test():
            tokenizer = AsyncDynamicbatchTokenizer(FakeTokenizer())
            timing = {}
            try:
                result = await tokenizer.encode(
                    "hello", _tokenization_timing=timing
                )
                self.assertEqual(result["input_ids"], [5])
                self.assertLessEqual(timing["queue_entry"], timing["exec_start"])
                self.assertLessEqual(timing["exec_start"], timing["exec_finish"])
            finally:
                if tokenizer._batcher_task is not None:
                    tokenizer._batcher_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await tokenizer._batcher_task
                tokenizer._executor.shutdown(wait=True)

        asyncio.run(run_test())


if __name__ == "__main__":
    import unittest

    unittest.main()
