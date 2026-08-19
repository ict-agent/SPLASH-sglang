"""Unit tests for TokenizerManager's serialized tokenizer offload executor."""

import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import tokenizer_manager as tokenizer_manager_module
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    cap_torch_intraop_threads,
)

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class _FakeTokenizerManager:
    run_tokenizer_offload = TokenizerManager.run_tokenizer_offload

    def __init__(self, executor):
        self.tokenizer_offload_executor = executor


class TestTokenizerOffload(CustomTestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tokenizer_offload_test"
        )
        self.manager = _FakeTokenizerManager(self.executor)

    def tearDown(self):
        self.executor.shutdown(wait=True)

    def test_event_loop_remains_responsive(self):
        release = threading.Event()
        loop_ticks = []

        def blocking_conversion():
            release.wait(timeout=2)
            return "converted"

        async def main():
            conversion = asyncio.create_task(
                self.manager.run_tokenizer_offload(blocking_conversion)
            )
            for _ in range(10):
                loop_ticks.append(1)
                await asyncio.sleep(0.002)
            self.assertFalse(conversion.done())
            release.set()
            self.assertEqual(await conversion, "converted")

        asyncio.run(main())
        self.assertEqual(len(loop_ticks), 10)

    def test_calls_are_serialized_on_one_worker_thread(self):
        lock = threading.Lock()
        active = 0
        max_active = 0
        worker_threads = set()

        def conversion():
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            worker_threads.add(threading.get_ident())
            time.sleep(0.01)
            with lock:
                active -= 1
            return True

        async def main():
            return await asyncio.gather(
                *[self.manager.run_tokenizer_offload(conversion) for _ in range(4)]
            )

        self.assertEqual(asyncio.run(main()), [True] * 4)
        self.assertEqual(max_active, 1)
        self.assertEqual(len(worker_threads), 1)
        self.assertNotIn(threading.get_ident(), worker_threads)

    def test_exception_and_arguments_propagate(self):
        def raise_template_error():
            raise ValueError("template error")

        async def main():
            result = await self.manager.run_tokenizer_offload(
                lambda a, b=0: (a, b), 1, b=2
            )
            self.assertEqual(result, (1, 2))
            with self.assertRaisesRegex(ValueError, "template error"):
                await self.manager.run_tokenizer_offload(raise_template_error)

        asyncio.run(main())

    def test_disabled_executor_preserves_inline_behavior(self):
        manager = _FakeTokenizerManager(None)

        async def main():
            return await manager.run_tokenizer_offload(threading.get_ident)

        self.assertEqual(asyncio.run(main()), threading.get_ident())

    def test_slow_tokenizer_batch_uses_offload_thread(self):
        worker_threads = set()

        class SlowTokenizer:
            is_fast = False

            def encode(self, text):
                worker_threads.add(threading.get_ident())
                return [len(text)]

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = SlowTokenizer()
        manager.async_dynamic_batch_tokenizer = None
        manager.tokenizer_offload_executor = self.executor

        input_ids, token_type_ids = asyncio.run(
            manager._tokenize_texts(["a", "bb"], is_cross_encoder=False)
        )

        self.assertEqual(input_ids, [[1], [2]])
        self.assertIsNone(token_type_ids)
        self.assertEqual(len(worker_threads), 1)
        self.assertNotIn(threading.get_ident(), worker_threads)


class TestTokenizerTorchThreadCap(CustomTestCase):
    def test_configured_thread_cap_is_applied(self):
        with (
            patch.object(
                tokenizer_manager_module.envs.GLM_TOKENIZER_TORCH_NUM_THREADS,
                "get",
                return_value=4,
            ),
            patch("torch.set_num_threads") as set_num_threads,
        ):
            cap_torch_intraop_threads()

        set_num_threads.assert_called_once_with(4)

    def test_non_positive_thread_cap_keeps_torch_default(self):
        with (
            patch.object(
                tokenizer_manager_module.envs.GLM_TOKENIZER_TORCH_NUM_THREADS,
                "get",
                return_value=0,
            ),
            patch("torch.set_num_threads") as set_num_threads,
        ):
            cap_torch_intraop_threads()

        set_num_threads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
