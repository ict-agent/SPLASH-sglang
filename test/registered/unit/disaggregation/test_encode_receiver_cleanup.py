"""Unit tests for EPD encode_receiver failure cleanup and fail-fast.

Covers the leak fixes for the P-node memory growth issue:
- recv_mm_data cleans up the encode/recv tasks and the mooncake buffer on
  timeout / cancellation / encoder failure / unexpected exceptions.
- The encode task failure (e.g. encoder unreachable) fails the request fast
  instead of waiting out the full recv timeout.
- _abort_encode_and_cleanup drains already-done tasks so their exceptions
  are always retrieved.
"""

import asyncio
import concurrent.futures
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.encode_receiver import (  # noqa: E402
    EncoderError,
    MMReceiverHTTP,
    RdmaBufferPool,
    RdmaRegRefcount,
    rdma_pool_enabled,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_receiver():
    """Build a bare MMReceiverHTTP without running its __init__."""
    receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
    receiver.encode_urls = ["http://fake-encoder:8080"]
    receiver.host = "127.0.0.1"
    receiver.context = MagicMock()
    receiver.encoder_transfer_backend = "mooncake"
    receiver.embeddings_buffer = {}
    receiver.embeddings_engine = MagicMock()
    receiver.embeddings_engine.register = MagicMock(return_value=0)
    receiver.embeddings_engine.deregister = MagicMock()
    receiver._use_rdma_pool = False
    receiver._rdma_pool = None
    return receiver


class _FakeMooncakeEngine:
    def __init__(self, register_result=0):
        self.register_result = register_result
        self.register_calls = []
        self.deregister_calls = []
        self._lock = threading.Lock()

    def register(self, addr, nbytes):
        with self._lock:
            self.register_calls.append((addr, nbytes))
        return self.register_result

    def deregister(self, addr):
        with self._lock:
            self.deregister_calls.append(addr)


class TestRdmaRegistrationLifecycle(unittest.TestCase):
    def test_pool_requires_both_positive_limits(self):
        with envs.SGLANG_MC_RDMA_POOL_MAX_MB.override(
            1
        ), envs.SGLANG_MC_RDMA_POOL_MAX_BUFFERS.override(0):
            self.assertFalse(rdma_pool_enabled())
        with envs.SGLANG_MC_RDMA_POOL_MAX_MB.override(
            1
        ), envs.SGLANG_MC_RDMA_POOL_MAX_BUFFERS.override(1):
            self.assertTrue(rdma_pool_enabled())

    def test_shared_tensor_is_registered_once_for_concurrent_users(self):
        engine = _FakeMooncakeEngine()
        registry = RdmaRegRefcount(engine)
        tensor = torch.empty(1024, dtype=torch.uint8)
        acquired = threading.Barrier(3)
        release = threading.Event()

        def use_tensor():
            addr = registry.acquire(tensor)
            acquired.wait(timeout=5)
            release.wait(timeout=5)
            registry.release(addr)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(use_tensor) for _ in range(2)]
            acquired.wait(timeout=5)
            self.assertEqual(
                engine.register_calls, [(tensor.data_ptr(), tensor.nbytes)]
            )
            self.assertEqual(engine.deregister_calls, [])
            release.set()
            for future in futures:
                future.result(timeout=5)

        self.assertEqual(engine.deregister_calls, [tensor.data_ptr()])

    def test_receive_pool_reuses_registered_buffer(self):
        engine = _FakeMooncakeEngine()
        with envs.SGLANG_MC_RDMA_POOL_MAX_MB.override(
            2
        ), envs.SGLANG_MC_RDMA_POOL_MAX_BUFFERS.override(2):
            pool = RdmaBufferPool(engine)
            first = pool.acquire(100)
            pool.release(first)
            second = pool.acquire(100)

        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(len(engine.register_calls), 1)
        pool.discard(second)
        self.assertEqual(engine.deregister_calls, [second.data_ptr()])


class TestRecvMMDataCleanup(unittest.IsolatedAsyncioTestCase):
    """recv_mm_data must clean up tasks and buffers on every failure path."""

    def setUp(self):
        self.receiver = _make_receiver()
        self.socket_patcher = patch(
            "sglang.srt.disaggregation.encode_receiver.get_zmq_socket_on_host",
            return_value=(12345, MagicMock()),
        )
        self.socket_patcher.start()
        self.addCleanup(self.socket_patcher.stop)
        self.extract_patcher = patch.object(
            MMReceiverHTTP, "_extract_url_data", return_value=[]
        )
        self.extract_patcher.start()
        self.addCleanup(self.extract_patcher.stop)

    async def test_encoder_failure_fails_fast(self):
        """encode task failure must surface immediately, not after the
        180s recv timeout."""

        async def failing_encode(*args, **kwargs):
            raise EncoderError("Cannot connect to host", status_code=503)

        async def slow_recv(*args, **kwargs):
            await asyncio.sleep(3600)  # would hit the recv timeout

        with patch.object(MMReceiverHTTP, "encode", failing_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", slow_recv
        ):
            start = asyncio.get_event_loop().time()
            with self.assertRaises(EncoderError) as ctx:
                await self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p",
                )
            elapsed = asyncio.get_event_loop().time() - start
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertLess(elapsed, 5.0, "should fail fast, not wait out timeout")

    async def test_timeout_cleans_buffer_and_cancels_encode(self):
        encode_started = asyncio.Event()
        encode_cancelled = asyncio.Event()

        async def hanging_encode(self_, req_id, *args, **kwargs):
            # Simulate: buffer allocated, then /send hangs.
            self_.embeddings_buffer[req_id] = MagicMock(
                data_ptr=MagicMock(return_value=0)
            )
            encode_started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                encode_cancelled.set()
                raise

        async def slow_recv(*args, **kwargs):
            await asyncio.sleep(3600)

        with envs.SGLANG_ENCODER_RECV_TIMEOUT.override(0.2), patch.object(
            MMReceiverHTTP, "encode", hanging_encode
        ), patch.object(MMReceiverHTTP, "_recv_mm_data", slow_recv):
            result = await self.receiver.recv_mm_data(
                request_obj=MagicMock(_upstream_session_id=None),
                mm_processor=MagicMock(),
                prompt="p",
            )
        self.assertIsNone(result)  # timeout returns None (-> 504 upstream)
        self.assertTrue(encode_started.is_set())
        self.assertTrue(encode_cancelled.is_set(), "encode task must be cancelled")
        self.assertEqual(
            len(self.receiver.embeddings_buffer), 0, "buffer must be cleaned"
        )
        self.receiver.embeddings_engine.deregister.assert_called_once()

    async def test_cancellation_cleans_buffer(self):
        async def hanging_encode(self_, req_id, *args, **kwargs):
            self_.embeddings_buffer[req_id] = MagicMock(
                data_ptr=MagicMock(return_value=0)
            )
            await asyncio.sleep(3600)

        async def slow_recv(*args, **kwargs):
            await asyncio.sleep(3600)

        with patch.object(MMReceiverHTTP, "encode", hanging_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", slow_recv
        ):
            task = asyncio.create_task(
                self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p",
                )
            )
            await asyncio.sleep(0.1)  # let it start
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)

    async def test_unexpected_recv_exception_cleans_buffer(self):
        """Non-timeout errors in the recv path (pickle, mm_processor...) must
        also trigger cleanup instead of leaking the buffer."""

        async def ok_encode(self_, req_id, *args, **kwargs):
            self_.embeddings_buffer[req_id] = MagicMock(
                data_ptr=MagicMock(return_value=0)
            )

        async def broken_recv(*args, **kwargs):
            raise ValueError("pickle exploded")

        with patch.object(MMReceiverHTTP, "encode", ok_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", broken_recv
        ):
            with self.assertRaises(ValueError):
                await self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p",
                )
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)

    async def test_success_path_returns_result(self):
        async def ok_encode(self_, req_id, *args, **kwargs):
            return None

        async def ok_recv(*args, **kwargs):
            return {"mm": "inputs"}

        with patch.object(MMReceiverHTTP, "encode", ok_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", ok_recv
        ):
            result = await self.receiver.recv_mm_data(
                request_obj=MagicMock(_upstream_session_id=None),
                mm_processor=MagicMock(),
                prompt="p",
            )
        self.assertEqual(result, {"mm": "inputs"})


class TestAbortEncodeAndCleanup(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.receiver = _make_receiver()

    async def test_drains_already_done_task_exception(self):
        """A done task with an exception must be drained (no 'Task exception
        was never retrieved')."""

        async def boom():
            raise RuntimeError("late failure")

        task = asyncio.create_task(boom())
        await asyncio.sleep(0)  # let it finish
        self.assertTrue(task.done())
        # Must not raise, and must retrieve the exception.
        await self.receiver._abort_encode_and_cleanup(task, req_id=None)
        self.assertIsNotNone(task.exception())

    async def test_cleanup_deregisters_buffer(self):
        buf = MagicMock(data_ptr=MagicMock(return_value=42))
        self.receiver.embeddings_buffer["rid"] = buf
        await self.receiver._abort_encode_and_cleanup(None, req_id="rid")
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)
        self.receiver.embeddings_engine.deregister.assert_called_once_with(42)

    async def test_cleanup_is_idempotent(self):
        await self.receiver._abort_encode_and_cleanup(None, req_id="missing")
        self.receiver.embeddings_engine.deregister.assert_not_called()


class TestRaceEncodeAndRecv(unittest.IsolatedAsyncioTestCase):
    async def test_recv_wins(self):
        async def encode():
            await asyncio.sleep(3600)

        async def recv():
            return "data"

        encode_task = asyncio.create_task(encode())
        recv_task = asyncio.create_task(recv())
        try:
            result = await MMReceiverHTTP._race_encode_and_recv(
                encode_task, recv_task
            )
            self.assertEqual(result, "data")
        finally:
            encode_task.cancel()

    async def test_encode_success_then_recv(self):
        async def encode():
            return None

        async def recv():
            await asyncio.sleep(0.05)
            return "data"

        result = await MMReceiverHTTP._race_encode_and_recv(
            asyncio.create_task(encode()), asyncio.create_task(recv())
        )
        self.assertEqual(result, "data")

    async def test_encode_failure_raises(self):
        async def encode():
            raise EncoderError("boom", status_code=500)

        async def recv():
            await asyncio.sleep(3600)

        recv_task = asyncio.create_task(recv())
        try:
            with self.assertRaises(EncoderError):
                await MMReceiverHTTP._race_encode_and_recv(
                    asyncio.create_task(encode()), recv_task
                )
        finally:
            recv_task.cancel()


if __name__ == "__main__":
    unittest.main()
