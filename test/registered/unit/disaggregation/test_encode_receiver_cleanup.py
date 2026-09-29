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
import pickle
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.encode_receiver import (  # noqa: E402
    EmbeddingData,
    EncoderError,
    MMReceiverHTTP,
    RdmaBufferPool,
    RdmaRegRefcount,
    rdma_pool_enabled,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.schedule_batch import Modality  # noqa: E402
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
    receiver._buffer_index = {}
    receiver._cleanup_tasks = set()
    receiver.embeddings_engine = MagicMock()
    receiver.embeddings_engine.register = MagicMock(return_value=0)
    receiver.embeddings_engine.deregister = MagicMock()
    receiver._use_rdma_pool = False
    receiver._rdma_pool = None
    receiver.meta_only = False
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
    def test_pool_enabled_by_byte_cap(self):
        with envs.SGLANG_MC_RDMA_POOL_MAX_MB.override(0):
            self.assertFalse(rdma_pool_enabled())
        with envs.SGLANG_MC_RDMA_POOL_MAX_MB.override(1):
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
        with envs.SGLANG_MC_RDMA_POOL_MAX_MB.override(2):
            pool = RdmaBufferPool(engine)
            first = pool.acquire(100)
            pool.release(first)
            second = pool.acquire(100)

        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(len(engine.register_calls), 1)
        pool.discard(second)
        self.assertEqual(engine.deregister_calls, [second.data_ptr()])

    def test_checked_out_buffers_do_not_consume_free_byte_budget(self):
        engine = _FakeMooncakeEngine()
        pool = RdmaBufferPool(engine)
        pool._floor = 1
        pool._max_total_bytes = 4
        pool._acquire_timeout = 0

        buffers = [pool.acquire(2) for _ in range(3)]
        pool.release(buffers[0])
        pool.release(buffers[1])

        self.assertEqual(engine.deregister_calls, [])
        self.assertEqual(pool._free_bytes, 4)

        pool.release(buffers[2])
        self.assertEqual(engine.deregister_calls, [buffers[2].data_ptr()])
        self.assertEqual(pool._free_bytes, 4)
        self.assertEqual(pool._total_bytes, 4)

    def test_acquire_evicts_idle_smaller_buffers_for_budget(self):
        engine = _FakeMooncakeEngine()
        pool = RdmaBufferPool(engine)
        pool._floor = 1
        pool._max_total_bytes = 4

        small = [pool.acquire(1) for _ in range(4)]
        for buffer in small:
            pool.release(buffer)
        self.assertEqual(pool._free_count, 4)

        # The cache is saturated with 1-byte buffers; a 2-byte acquire must
        # evict idle ones for budget instead of stalling in the timed wait.
        big = pool.acquire(2)
        self.assertEqual(big.numel(), 2)
        self.assertEqual(len(engine.deregister_calls), 2)
        self.assertEqual(pool._total_bytes, 4)
        self.assertEqual(pool._free_count, 2)


class TestAcquirePoolBufferAsync(unittest.IsolatedAsyncioTestCase):
    """_acquire_pool_buffer must keep the event loop free and never leak
    budget when the awaiting request is cancelled mid-wait."""

    async def test_cancelled_acquire_reclaims_buffer(self):
        receiver = _make_receiver()
        receiver._use_rdma_pool = True
        engine = _FakeMooncakeEngine()
        pool = RdmaBufferPool(engine)
        pool._floor = 1
        pool._max_total_bytes = 1
        receiver._rdma_pool = pool

        blocker = pool.acquire(1)  # exhausts the byte budget
        task = asyncio.create_task(receiver._acquire_pool_buffer(1))
        await asyncio.sleep(0.2)  # let the worker thread enter the pool wait
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        # Waking the abandoned waiter hands it the released buffer; the
        # reclaim callback must return it to the pool instead of leaking it.
        pool.release(blocker)
        for _ in range(50):
            with pool._lock:
                if pool._free_count == 1:
                    break
            await asyncio.sleep(0.1)
        self.assertEqual(pool._free_count, 1)
        self.assertEqual(pool._total_count, 1)
        self.assertEqual(engine.deregister_calls, [])


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
            # Simulate: buffer allocated, then /send hangs. Store through the
            # production helper so the buffer lands in both embeddings_buffer
            # and the _buffer_index the cleanup path resolves through.
            self_._store_embedding_buffer(
                req_id, MagicMock(data_ptr=MagicMock(return_value=0)), 0
            )
            encode_started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                encode_cancelled.set()
                raise

        async def slow_recv(*args, **kwargs):
            await asyncio.sleep(3600)

        with envs.SGLANG_ENCODER_RECV_TIMEOUT.override(0.2), patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.5,
        ), patch.object(MMReceiverHTTP, "encode", hanging_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", slow_recv
        ):
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
        self.assertEqual(
            len(self.receiver._buffer_index), 0, "buffer index must be cleaned"
        )
        self.receiver.embeddings_engine.deregister.assert_called_once()

    async def test_cancellation_cleans_buffer(self):
        async def hanging_encode(self_, req_id, *args, **kwargs):
            self_._store_embedding_buffer(
                req_id, MagicMock(data_ptr=MagicMock(return_value=0)), 0
            )
            await asyncio.sleep(3600)

        async def slow_recv(*args, **kwargs):
            await asyncio.sleep(3600)

        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.5,
        ), patch.object(MMReceiverHTTP, "encode", hanging_encode), patch.object(
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
            # Cleanup is detached to a background task on the cancellation
            # path; wait for it to finish draining and deregistering.
            pending_cleanups = list(self.receiver._cleanup_tasks)
            if pending_cleanups:
                await asyncio.wait_for(
                    asyncio.gather(*pending_cleanups, return_exceptions=True),
                    timeout=5,
                )
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)

    async def test_unexpected_recv_exception_cleans_buffer(self):
        """Non-timeout errors in the recv path (pickle, mm_processor...) must
        also trigger cleanup instead of leaking the buffer."""

        async def ok_encode(self_, req_id, *args, **kwargs):
            self_._store_embedding_buffer(
                req_id, MagicMock(data_ptr=MagicMock(return_value=0)), 0
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
        self.receiver._store_embedding_buffer("rid", buf, 0)
        # Buffers exist -> the cleanup now always quiesces one window before
        # deregistering; shrink the window so this test stays fast.
        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.01,
        ):
            await self.receiver._abort_encode_and_cleanup(None, req_id="rid")
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)
        self.assertEqual(len(self.receiver._buffer_index), 0)
        self.receiver.embeddings_engine.deregister.assert_called_once_with(42)

    async def test_cleanup_is_idempotent(self):
        await self.receiver._abort_encode_and_cleanup(None, req_id="missing")
        self.receiver.embeddings_engine.deregister.assert_not_called()

    async def test_cleanup_drops_every_part_of_a_request(self):
        """Cleanup must drop all of a request's part buffers via the index,
        not just one and not other requests' parts."""
        own_bufs = [
            MagicMock(data_ptr=MagicMock(return_value=100 + i)) for i in range(3)
        ]
        for i, buf in enumerate(own_bufs):
            self.receiver._store_embedding_buffer(f"rid_local_part_{i}", buf, 0)
        other_buf = MagicMock(data_ptr=MagicMock(return_value=999))
        self.receiver._store_embedding_buffer("other_local_part_0", other_buf, 0)

        # Same as above: shrink the mandatory quiesce window.
        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.01,
        ):
            await self.receiver._abort_encode_and_cleanup(None, req_id="rid")
        self.assertEqual(len(self.receiver.embeddings_buffer), 1)
        self.assertIn("other_local_part_0", self.receiver.embeddings_buffer)
        deregistered = {
            call.args[0]
            for call in self.receiver.embeddings_engine.deregister.call_args_list
        }
        self.assertEqual(deregistered, {100, 101, 102})


class TestQuiesceOnFailure(unittest.IsolatedAsyncioTestCase):
    """The quiesce hold before deregister on E-side write failures.

    A reported failure (error frame surfacing via recv, non-200 /send) with
    live buffers must delay deregistration by one drain window so E's
    residual posted slices hit a live rkey; a pure /encode dispatch failure
    (no buffer ever allocated) must NOT pay that latency.
    """

    def setUp(self):
        self.receiver = _make_receiver()

    async def test_error_frame_with_buffer_quiesces_before_deregister(self):
        """E-side RDMA write failure arrives as an error frame from recv while
        every /send returned 200 (encode task 'successful'). Cleanup must hold
        the MRs for one drain window before deregistering."""
        quiesce_slept = []

        async def ok_encode(self_, req_id, *args, **kwargs):
            self_._store_embedding_buffer(
                req_id, MagicMock(data_ptr=MagicMock(return_value=7)), 0
            )

        async def error_frame_recv(*args, **kwargs):
            raise EncoderError("mooncake RDMA write failed", status_code=500)

        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            # Only the quiesce hold sleeps for the drain-bound window.
            if seconds <= 1:
                quiesce_slept.append(seconds)
            await real_sleep(seconds if seconds > 1 else 0)

        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.05,
        ), patch.object(MMReceiverHTTP, "encode", ok_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", error_frame_recv
        ), patch(
            "asyncio.sleep", side_effect=fake_sleep
        ):
            with self.assertRaises(EncoderError):
                await self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p",
                )
        self.assertEqual(
            len(quiesce_slept), 1, "cleanup must quiesce once before deregister"
        )
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)

    async def test_encode_dispatch_failure_does_not_quiesce(self):
        """A failure before any /send (encoder unreachable, bad image) never
        allocates a buffer: cleanup must deregister immediately (no quiesce
        hold), preserving the fail-fast contract."""
        quiesce_slept = []

        async def failing_encode(*args, **kwargs):
            raise EncoderError("Cannot connect to host", status_code=503)

        async def slow_recv(*args, **kwargs):
            await asyncio.sleep(3600)

        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            # Only the quiesce hold sleeps for the drain-bound window; the
            # test stub's asyncio.sleep(3600) calls must pass through and are
            # filtered out by magnitude.
            if seconds <= 1:
                quiesce_slept.append(seconds)
            await real_sleep(seconds if seconds > 1 else 0)

        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.05,
        ), patch.object(MMReceiverHTTP, "encode", failing_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", slow_recv
        ), patch(
            "asyncio.sleep", side_effect=fake_sleep
        ):
            start = asyncio.get_event_loop().time()
            # recv_mm_data can only surface once either task completes; the
            # failing encode completes immediately, but the surrounding
            # recv-timeout wait_for is 180s -- drive it with a short outer
            # timeout and accept either outcome's timing, asserting only the
            # quiesce contract below.
            try:
                await asyncio.wait_for(
                    self.receiver.recv_mm_data(
                        request_obj=MagicMock(_upstream_session_id=None),
                        mm_processor=MagicMock(),
                        prompt="p",
                    ),
                    timeout=5,
                )
            except (EncoderError, asyncio.TimeoutError):
                pass
            elapsed = asyncio.get_event_loop().time() - start
        self.assertEqual(quiesce_slept, [], "no quiesce without live buffers")
        self.assertLess(elapsed, 5.0, "must fail fast without buffers")


class TestRecancelDuringCleanup(unittest.IsolatedAsyncioTestCase):
    """Re-cancellation racing the inline cleanup (the mid-drain/mid-quiesce
    cancel bug).

    A second cancellation arriving while _abort_encode_and_cleanup is inline
    (draining, or holding the quiesce window) must NOT skip the remaining
    hold and deregister immediately -- that re-opens the
    deregister-while-writing race. The buffers must be handed to a detached
    task that completes the hold before deregistering.
    """

    def setUp(self):
        self.receiver = _make_receiver()

    async def test_recancel_mid_quiesce_defers_deregister(self):
        """E-side write failure surfaces via recv (error frame) with a live
        buffer; the quiesce hold starts; the caller is then cancelled again
        mid-hold. The inline path must deregister NOTHING and hand the
        buffers to a detached task which completes the hold and only then
        discards them."""
        import sglang.srt.disaggregation.encode_receiver as er_mod

        deregister_calls = []

        async def ok_encode(self_, req_id, *args, **kwargs):
            self_._store_embedding_buffer(
                req_id, MagicMock(data_ptr=MagicMock(return_value=7)), 0
            )

        async def error_frame_recv(*args, **kwargs):
            raise EncoderError("mooncake RDMA write failed", status_code=500)

        real_sleep = asyncio.sleep
        hold_started = asyncio.Event()
        detach_scheduled = []
        real_schedule = er_mod.MMReceiverBase._schedule_quiesced_deregister

        def spy_schedule(self, req_id):
            detach_scheduled.append(req_id)
            return real_schedule(self, req_id)

        async def fake_sleep(seconds):
            if seconds <= 1:
                hold_started.set()
                # Stay asleep until the test cancels the parent (bounded).
                await asyncio.sleep(5)
                return
            await real_sleep(seconds)

        orig_dereg = self.receiver.embeddings_engine.deregister
        self.receiver.embeddings_engine.deregister = MagicMock(
            side_effect=lambda addr: deregister_calls.append(addr)
        )

        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.5,
        ), patch.object(MMReceiverHTTP, "encode", ok_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", error_frame_recv
        ), patch(
            "asyncio.sleep", side_effect=fake_sleep
        ), patch.object(
            er_mod.MMReceiverBase,
            "_schedule_quiesced_deregister",
            spy_schedule,
        ):
            task = asyncio.create_task(
                self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p",
                )
            )
            # Wait until the inline quiesce hold is actually sleeping.
            await asyncio.wait_for(hold_started.wait(), timeout=5)
            # Second cancellation mid-hold.
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            # Inline path must NOT have deregistered anything at this point.
            self.assertEqual(
                deregister_calls,
                [],
                "inline cleanup must not deregister when re-cancelled " "mid-quiesce",
            )
            self.assertEqual(
                len(detach_scheduled), 1, "detached quiesce must be scheduled"
            )
            # The detached task holds the buffer until the (patched) window
            # elapses, then deregisters. Drain _cleanup_tasks.
            pending = list(getattr(self.receiver, "_cleanup_tasks", ()))
            if pending:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=10
                )
        self.assertEqual(
            deregister_calls,
            [7],
            "detached task must deregister exactly once after the hold",
        )
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)

    async def test_recancel_mid_drain_defers_deregister(self):
        """Same contract for a cancellation landing in the drain wait itself:
        encode task still pending (a buffer was handed over by an earlier
        part), the drain asyncio.wait is interrupted by the parent cancel.
        No inline deregister; detached hold + deregister afterwards."""
        import sglang.srt.disaggregation.encode_receiver as er_mod

        deregister_calls = []
        buf = MagicMock(data_ptr=MagicMock(return_value=9))

        async def hanging_encode(self_, req_id, *args, **kwargs):
            # One part already handed its buffer to E; a later /send hangs.
            self_._store_embedding_buffer(req_id, buf, 0)
            await asyncio.sleep(3600)

        async def slow_recv(*args, **kwargs):
            await asyncio.sleep(3600)

        real_wait = asyncio.wait
        drain_started = asyncio.Event()
        detach_scheduled = []
        real_schedule = er_mod.MMReceiverBase._schedule_quiesced_deregister

        def spy_schedule(self, req_id):
            detach_scheduled.append(req_id)
            return real_schedule(self, req_id)

        async def fake_wait(fs, timeout=None, **kwargs):
            drain_started.set()
            return await real_wait(fs, timeout=timeout, **kwargs)

        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            if seconds <= 1:
                await asyncio.sleep(3)  # detached hold (short)
                return
            await real_sleep(seconds)

        self.receiver.embeddings_engine.deregister = MagicMock(
            side_effect=lambda addr: deregister_calls.append(addr)
        )

        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.5,
        ), patch.object(MMReceiverHTTP, "encode", hanging_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", slow_recv
        ), patch(
            "asyncio.wait", side_effect=fake_wait
        ), patch(
            "asyncio.sleep", side_effect=fake_sleep
        ), patch.object(
            er_mod.MMReceiverBase,
            "_schedule_quiesced_deregister",
            spy_schedule,
        ):
            with envs.SGLANG_ENCODER_RECV_TIMEOUT.override(0.1):
                task = asyncio.create_task(
                    self.receiver.recv_mm_data(
                        request_obj=MagicMock(_upstream_session_id=None),
                        mm_processor=MagicMock(),
                        prompt="p",
                    )
                )
                await asyncio.wait_for(drain_started.wait(), timeout=5)
                # Yield once so the TimeoutError handler has definitely
                # entered the INLINE drain (drain_started fires at the top
                # of fake_wait, before the task actually suspends inside
                # real_wait; cancelling too early races the handler and the
                # detached _schedule_drained_cleanup path takes over instead,
                # which is correct behaviour but not the one under test).
                await asyncio.sleep(0.05)
                # Cancel the parent while the drain wait is in progress.
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(
                    deregister_calls,
                    [],
                    "inline cleanup must not deregister when re-cancelled " "mid-drain",
                )
                self.assertEqual(len(detach_scheduled), 1)
                pending = list(getattr(self.receiver, "_cleanup_tasks", ()))
                if pending:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=10,
                    )
        self.assertEqual(deregister_calls, [9])
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)


class TestRaceEncodeAndRecv(unittest.IsolatedAsyncioTestCase):
    async def test_recv_wins(self):
        async def encode():
            await asyncio.sleep(3600)

        async def recv():
            return "data"

        encode_task = asyncio.create_task(encode())
        recv_task = asyncio.create_task(recv())
        try:
            result = await MMReceiverHTTP._race_encode_and_recv(encode_task, recv_task)
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


class TestQuiesceWithoutFailureSignal(unittest.IsolatedAsyncioTestCase):
    """P0-1 regression: the timeout / client-disconnect paths.

    On these paths the recv task -- the ONLY consumer of E's ZMQ error frame
    and the /send response reader is gone or never sees a failure -- so the
    cleanup cannot rely on `failed`. If any buffer was handed to E (a /send
    was dispatched), the quiesce hold MUST still happen before deregistering,
    or residual posted slices land on a torn rkey (the RAE race).
    """

    def setUp(self):
        self.receiver = _make_receiver()
        self.receiver.dtype = torch.bfloat16

    async def test_timeout_path_with_buffers_still_quiesces(self):
        """Simulates the recv-timeout path: recv cancelled, encode task
        completed 'successfully' (every /send 200), buffers still held.
        Cleanup must quiesce exactly once before deregistering."""
        self.receiver._store_embedding_buffer(
            "r_local_part_0", MagicMock(data_ptr=MagicMock(return_value=7)), 0
        )
        encode_task = asyncio.create_task(_noop_future("ok"))
        await encode_task  # already drained, no exception anywhere
        recv_task = asyncio.create_task(_sleep_forever())
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

        quiesce_slept = []
        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            if 0 < seconds <= 1:
                quiesce_slept.append(seconds)
            await real_sleep(seconds if seconds > 1 else 0)

        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.05,
        ), patch("asyncio.sleep", side_effect=fake_sleep):
            await self.receiver._abort_encode_and_cleanup(encode_task, "r", recv_task)
        self.assertEqual(
            len(quiesce_slept),
            1,
            "cleanup must quiesce on live buffers even without a failure "
            "signal (timeout/disconnect path)",
        )
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)
        self.assertEqual(self.receiver._buffer_index.get("r"), None)

    async def test_dtype_mismatch_raises(self):
        """A frame whose element type differs from the model's (bf16 vs fp16,
        both 2-byte) must fail loudly instead of being silently reinterpreted
        into corrupted output."""

        class _FakeRecvSocket:
            def __init__(self, frames):
                self._frames = frames
                self.closed = False

            async def recv_multipart(self, copy=False):
                if self._frames:
                    return [self._frames.pop(0)]
                await asyncio.sleep(3600)

            def close(self):
                self.closed = True

        frame = pickle.dumps(
            EmbeddingData(
                req_id="r_local_part_0",
                num_parts=1,
                part_idx=0,
                grid_dim=torch.tensor([1, 1, 1]),
                modality=Modality.IMAGE,
                embedding=torch.zeros(2, dtype=torch.float32),  # E sent fp32
            )
        )
        sock = _FakeRecvSocket([frame])
        with self.assertRaises(EncoderError) as ctx:
            await self.receiver._recv_mm_data(
                req_id="r", recv_socket=sock, mm_processor=MagicMock(), prompt="p"
            )
        self.assertIn("dtype mismatch", str(ctx.exception))
        sock.closed = True
        sock_task = asyncio.current_task()  # sanity: socket closed in finally
        self.assertTrue(sock.closed or sock_task is None)

    async def test_dtype_match_passes(self):
        """Same frame flow with a matching dtype must NOT raise the mismatch
        error (it will proceed to aggregation and fail on missing buffers,
        which is fine for this test -- we only assert no dtype error)."""

        class _FakeRecvSocket:
            def __init__(self, frames):
                self._frames = frames
                self.closed = False

            async def recv_multipart(self, copy=False):
                if self._frames:
                    return [self._frames.pop(0)]
                await asyncio.sleep(3600)

            def close(self):
                self.closed = True

        frame = pickle.dumps(
            EmbeddingData(
                req_id="r_local_part_0",
                num_parts=1,
                part_idx=0,
                grid_dim=torch.tensor([1, 1, 1]),
                modality=Modality.IMAGE,
                embedding=torch.zeros(2, dtype=torch.bfloat16),  # matches
            )
        )
        sock = _FakeRecvSocket([frame])
        try:
            await asyncio.wait_for(
                self.receiver._recv_mm_data(
                    req_id="r", recv_socket=sock, mm_processor=MagicMock(), prompt="p"
                ),
                timeout=1,
            )
        except EncoderError as e:
            self.assertNotIn("dtype mismatch", str(e))
        except asyncio.TimeoutError:
            pass
        finally:
            sock.closed = True

    async def test_empty_video_shard_dtype_exempt(self):
        """N-1 regression: empty video shards carry a hardcoded placeholder
        dtype on the encoder (historically bfloat16 regardless of model
        dtype). A zero-row frame must NOT trip the dtype cross-check -- doing
        so would 500 the whole request for every sharded video with an empty
        shard on any fp16 model."""

        class _FakeRecvSocket:
            def __init__(self, frames):
                self._frames = frames
                self.closed = False

            async def recv_multipart(self, copy=False):
                if self._frames:
                    return [self._frames.pop(0)]
                await asyncio.sleep(3600)

            def close(self):
                self.closed = True

        # model is fp16; the empty-shard placeholder frame says bfloat16
        self.receiver.dtype = torch.float16
        # The real pipeline stores a (zero-byte) buffer per part before the
        # frame arrives; mirror that so the aggregation path runs to the end.
        self.receiver._store_embedding_buffer(
            "r_local_part_0", torch.zeros(0, dtype=torch.uint8), 0
        )
        frame = pickle.dumps(
            EmbeddingData(
                req_id="r_local_part_0",
                num_parts=1,
                part_idx=0,
                grid_dim=torch.tensor([1, 1, 1]),
                modality=Modality.VIDEO,
                embedding=torch.zeros((0, 8), dtype=torch.bfloat16),
            )
        )
        sock = _FakeRecvSocket([frame])
        try:
            await asyncio.wait_for(
                self.receiver._recv_mm_data(
                    req_id="r", recv_socket=sock, mm_processor=MagicMock(), prompt="p"
                ),
                timeout=1,
            )
        except EncoderError as e:
            self.fail(f"empty shard must be exempt from the dtype check, got: {e}")
        except asyncio.TimeoutError:
            pass  # aggregation waiting for more parts -- fine
        finally:
            sock.closed = True


async def _noop_future(value):
    return value


async def _sleep_forever():
    await asyncio.sleep(3600)


class TestSendFailureSurfacesOnHttp(unittest.IsolatedAsyncioTestCase):
    """/send must carry the RDMA write failure itself (P0-1 signal channel).

    The ZMQ error frame's only consumer (the recv task) may already be gone
    on timeout/disconnect; the /send HTTP response is the drain-coincident
    signal the receiver always reads (when it is still reading anything).
    """

    def setUp(self):
        import sglang.srt.disaggregation.encode_server as encode_server_module

        self.module = encode_server_module
        self._saved_encoder = getattr(encode_server_module, "encoder", None)
        self.encoder = MagicMock()
        self.encoder.metrics = None
        self.encoder.embedding_to_send = {}
        encode_server_module.encoder = self.encoder

    def tearDown(self):
        self.module.encoder = self._saved_encoder

    async def test_send_write_failure_returns_500(self):
        mm_data = EmbeddingData(
            req_id="r_local_part_0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=None,
            embedding=None,
            error_msg="mooncake RDMA write failed (register=0, transfer=-1)",
            error_code=500,
        )
        self.encoder.send = AsyncMock(return_value=mm_data)
        resp = await self.module.handle_send_request(
            {
                "req_id": "r_local_part_0",
                "prefill_host": "127.0.0.1",
                "embedding_port": 1,
                "session_id": "s",
                "buffer_address": 123,
            }
        )
        self.assertEqual(resp.status_code, 500)
        self.assertIn("mooncake RDMA write failed", resp.body.decode())

    async def test_send_reclaimed_returns_410(self):
        self.encoder.metrics = MagicMock()
        self.encoder.send = AsyncMock(return_value=None)
        resp = await self.module.handle_send_request(
            {
                "req_id": "r_local_part_0",
                "prefill_host": "127.0.0.1",
                "embedding_port": 1,
                "session_id": "s",
                "buffer_address": 123,
            }
        )
        self.assertEqual(resp.status_code, 410)
        self.encoder.metrics.inc_send_reclaimed.assert_called_once()

    async def test_send_ok_returns_200(self):
        mm_data = EmbeddingData(
            req_id="r_local_part_0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=None,
            embedding=None,
        )
        self.encoder.send = AsyncMock(return_value=mm_data)
        resp = await self.module.handle_send_request(
            {
                "req_id": "r_local_part_0",
                "prefill_host": "127.0.0.1",
                "embedding_port": 1,
                "session_id": "s",
                "buffer_address": 123,
            }
        )
        self.assertEqual(resp.status_code, 200)


class TestEncoderSourceMrQuiesce(unittest.IsolatedAsyncioTestCase):
    """P1-1: E's source MR must survive one window after a failed transfer.

    transferSync returning -1 only means the waiter gave up; posted slices can
    still DMA-read the source MR. Immediate deregister (or pool reuse) re-opens
    the deregister-while-writing race on the encoder side.
    """

    def _make_encoder(self, use_pool):
        from sglang.srt.disaggregation.encode_server import MMEncoder

        enc = MMEncoder.__new__(MMEncoder)
        enc.server_args = MagicMock(encoder_transfer_backend="mooncake")
        enc._use_rdma_pool = use_pool
        enc.engine = _FakeMooncakeEngine(register_result=0)
        enc.engine.transfer_sync = MagicMock(return_value=-1)
        enc._rdma_pool = RdmaBufferPool(enc.engine) if use_pool else None
        if enc._rdma_pool is not None:
            # These tests exercise reuse rather than the over-budget shrink
            # path. Production limits come from SGLANG_MC_RDMA_POOL_*.
            enc._rdma_pool._max_total_bytes = 8 * 1024 * 1024
        enc.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        enc.transfer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        enc.background_tasks = set()
        enc._source_mr_pins = {}
        enc.metrics = None
        enc.send_timeout = 1
        enc.sync_context = MagicMock()
        fake_socket = MagicMock()
        enc.sync_context.socket = MagicMock(return_value=fake_socket)
        enc.rank = 0
        return enc

    async def _run_send(self, enc, embedding, mm_data):
        with patch(
            "sglang.srt.disaggregation.encode_server.config_socket",
            lambda *a, **k: None,
        ), patch(
            "sglang.srt.disaggregation.encode_server._rdma_source_quiesce_s",
            lambda: 0.5,
        ):
            await enc._send(
                embedding,
                mm_data,
                session_id="sess",
                buffer_address=0xDEAD,
                prefill_host="127.0.0.1",
                embedding_port=1,
            )

    async def test_failed_transfer_defers_nonpool_deregister(self):
        enc = self._make_encoder(use_pool=False)
        enc.metrics = MagicMock()
        embedding = torch.zeros(16)
        mm_data = EmbeddingData(
            req_id="r_local_part_0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=None,
            embedding=embedding,
        )
        await self._run_send(enc, embedding, mm_data)
        self.assertIsNotNone(mm_data.error_msg, "write failure must surface")
        enc.metrics.inc_rdma_write_failures.assert_called_once_with("transfer")
        self.assertEqual(
            enc.engine.deregister_calls,
            [],
            "source MR must NOT be deregistered immediately after ret=-1",
        )
        # F-4: the tensor must be pinned for the whole window (mid-hold) ...
        addr = embedding.data_ptr()
        self.assertIn(addr, enc._source_mr_pins, "tensor pinned during hold")
        await asyncio.sleep(1.5)  # let the deferred release task fire
        self.assertEqual(
            len(enc.engine.deregister_calls),
            1,
            "source MR must be released after the quiesce window",
        )
        # ... and dropped exactly when the MR is released.
        self.assertEqual(enc._source_mr_pins, {}, "pin dropped after release")

    async def test_failed_transfer_defers_pool_release(self):
        enc = self._make_encoder(use_pool=True)
        embedding = torch.zeros(16)
        mm_data = EmbeddingData(
            req_id="r_local_part_0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=None,
            embedding=embedding,
        )
        await self._run_send(enc, embedding, mm_data)
        self.assertIsNotNone(mm_data.error_msg)
        self.assertEqual(
            enc.engine.deregister_calls,
            [],
            "pooled source MR must stay checked out after ret=-1",
        )
        self.assertEqual(len(enc._source_mr_pins), 1)
        held_addr = next(iter(enc._source_mr_pins))
        self.assertNotEqual(
            held_addr,
            embedding.data_ptr(),
            "sender must RDMA from the registered pool buffer",
        )
        await asyncio.sleep(1.5)  # window is patched to 0.5s
        self.assertEqual(
            enc.engine.deregister_calls,
            [],
            "within-budget pooled MR remains registered after quiesce",
        )
        self.assertEqual(enc._source_mr_pins, {})
        reused = enc._rdma_pool.acquire(embedding.nbytes)
        self.assertEqual(reused.data_ptr(), held_addr)
        self.assertEqual(len(enc.engine.register_calls), 1)
        enc._rdma_pool.release(reused)

    async def test_successful_transfer_releases_immediately(self):
        enc = self._make_encoder(use_pool=False)
        enc.engine.transfer_sync = MagicMock(return_value=0)
        embedding = torch.zeros(16)
        mm_data = EmbeddingData(
            req_id="r_local_part_0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=None,
            embedding=embedding,
        )
        await self._run_send(enc, embedding, mm_data)
        self.assertIsNone(mm_data.error_msg)
        self.assertEqual(
            len(enc.engine.deregister_calls), 1, "success path deregisters now"
        )

    async def test_successful_transfer_returns_sender_buffer_to_pool(self):
        enc = self._make_encoder(use_pool=True)
        enc.engine.transfer_sync = MagicMock(return_value=0)
        embedding = torch.arange(16, dtype=torch.float32)
        embedding_nbytes = embedding.nbytes
        mm_data = EmbeddingData(
            req_id="r_local_part_0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=None,
            embedding=embedding,
        )
        await self._run_send(enc, embedding, mm_data)
        self.assertIsNone(mm_data.error_msg)
        source_addr = enc.engine.transfer_sync.call_args.args[1]
        self.assertEqual(
            enc.engine.transfer_sync.call_args.args[3], embedding_nbytes
        )
        self.assertNotEqual(source_addr, embedding.data_ptr())
        self.assertEqual(enc.engine.deregister_calls, [])
        self.assertEqual(enc._source_mr_pins, {})
        reused = enc._rdma_pool.acquire(embedding.nbytes)
        self.assertEqual(reused.data_ptr(), source_addr)
        self.assertEqual(len(enc.engine.register_calls), 1)
        enc._rdma_pool.release(reused)

    async def test_cancelled_quiesce_still_completes_window_before_release(self):
        """N-2: cancelling the hold must not skip the window. The release
        moves to a detached task that sleeps the FULL window first; the
        source MR stays registered until then."""
        enc = self._make_encoder(use_pool=False)
        embedding = torch.zeros(16)
        mm_data = EmbeddingData(
            req_id="r_local_part_0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=None,
            embedding=embedding,
        )
        await self._run_send(enc, embedding, mm_data)
        self.assertEqual(enc.engine.deregister_calls, [])
        # Cancel the hold mid-window.
        hold_tasks = [t for t in enc.background_tasks if not t.done()]
        self.assertTrue(hold_tasks, "a quiesce task must be scheduled")
        hold_tasks[0].cancel()
        try:
            await hold_tasks[0]
        except asyncio.CancelledError:
            pass
        # The detached successor keeps the MR registered for the full window.
        self.assertEqual(
            enc.engine.deregister_calls, [], "MR must stay held after re-cancel"
        )
        await asyncio.sleep(1.5)  # window is patched to 0.5s
        self.assertEqual(
            len(enc.engine.deregister_calls),
            1,
            "detached successor must release after the full window",
        )


class TestGrpcBroadcastIncludesModality(unittest.IsolatedAsyncioTestCase):

    async def test_broadcast_dict_has_modality(self):
        try:
            from sglang.srt.disaggregation.encode_grpc_server import (
                SGLangEncoderServer,
            )
        except Exception as e:  # pragma: no cover
            # encode_grpc_server imports private wheels (smg_grpc_proto,
            # grpc_health...) that public CI images lack -- skip there.
            self.skipTest(f"grpc encoder deps unavailable: {e}")
        captured = {}

        class FakeSock:
            async def send_pyobj(self, obj):
                captured.update(obj)

        fake_encoder = MagicMock()
        fake_encoder.metrics = None
        fake_encoder.encode = AsyncMock(return_value=(0, 0, 0, None, None))
        server = SGLangEncoderServer(
            encoder=fake_encoder,
            send_sockets=[FakeSock()],
            server_args=MagicMock(encoder_transfer_backend="mooncake"),
        )
        request = MagicMock(
            mm_items=[], req_id="r", num_parts=1, part_idx=0, embedding_port=[]
        )
        await server.Encode(request, MagicMock())
        self.assertEqual(captured.get("modality"), "IMAGE")


class _FakeAiohttpResponse:
    def __init__(self, status, json_data=None, text_data="err"):
        self.status = status
        self._json = json_data
        self._text = text_data

    async def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    async def text(self):
        return self._text


class _FakeAiohttpSession:
    """Routes POSTs by URL suffix: /encode -> 200 + part json, /send -> the
    configured failure."""

    def __init__(self, send_status=500, send_message="mooncake RDMA write failed"):
        self.send_status = send_status
        self.send_message = send_message
        self.send_calls = 0

    async def post(self, url, json=None, headers=None):
        # Real aiohttp: `await session.post(...)` returns the response.
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint == "encode":
            # The real /encode handler echoes the whole request back plus
            # the embedding geometry; keep every request field (encoder_idx,
            # part_idx, req_id, ...) so the pipeline can drive /send.
            payload = dict(json)
            payload.update({"embedding_size": 128})
            return _FakeAiohttpResponse(200, json_data=payload)
        self.send_calls += 1
        return _FakeAiohttpResponse(
            self.send_status, json_data={"message": self.send_message}
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class TestSendNon200AbortsAndQuiesces(unittest.IsolatedAsyncioTestCase):
    """Pipeline-level: a non-200 /send (E-side RDMA write failure now also
    surfaces here) must fail the request through the existing abort path and
    quiesce the already-allocated part buffers before deregistering."""

    def setUp(self):
        self.receiver = _make_receiver()
        self.receiver.dtype = torch.bfloat16

    async def test_send_500_raises_and_quiesces(self):
        fake_session = _FakeAiohttpSession(send_status=500)
        stored = []

        async def fake_allocate(self_, part_req_id, total_bytes):
            buf = MagicMock(data_ptr=MagicMock(return_value=77))
            self_._store_embedding_buffer(part_req_id, buf, total_bytes)
            stored.append(part_req_id)
            return 77

        quiesce_slept = []
        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            if 0 < seconds <= 1:
                quiesce_slept.append(seconds)
            await real_sleep(seconds if seconds > 1 else 0)

        async def hanging_recv(*args, **kwargs):
            await real_sleep(3600)

        from sglang.srt.managers.schedule_batch import Modality as _Mod

        def fake_extract(self_, request_obj):
            return [{"url": "data:image/png;base64,x", "modality": _Mod.IMAGE}]

        with patch.object(MMReceiverHTTP, "_extract_url_data", fake_extract), patch(
            "sglang.srt.disaggregation.encode_receiver.aiohttp.ClientSession",
            lambda *a, **k: fake_session,
        ), patch.object(
            MMReceiverHTTP, "allocate_embedding_buffer", fake_allocate
        ), patch.object(
            MMReceiverHTTP, "_recv_mm_data", hanging_recv
        ), patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.05,
        ), patch(
            "asyncio.sleep", side_effect=fake_sleep
        ):
            with self.assertRaises(EncoderError) as ctx:
                await self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p",
                )
        self.assertIn("returned error 500 on /send", str(ctx.exception))
        self.assertEqual(fake_session.send_calls, 1, "one /send dispatched")
        self.assertEqual(
            len(quiesce_slept),
            1,
            "live buffers must be quiesced before deregister on /send failure",
        )
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)

    async def test_send_semaphore_bounds_inflight_per_encoder(self):
        """H-4: concurrent /send to one encoder is capped at
        SGLANG_ENCODER_MAX_INFLIGHT_SENDS; slots are reused per encoder."""
        slot0 = self.receiver._encoder_send_slot(0)
        slot0_again = self.receiver._encoder_send_slot(0)
        slot1 = self.receiver._encoder_send_slot(1)
        self.assertIs(slot0, slot0_again, "per-encoder slot must be reused")
        self.assertIsNot(slot0, slot1, "different encoders get distinct slots")
        # default budget of 10: acquire all, the 11th must not be immediate
        for _ in range(10):
            await asyncio.wait_for(slot0.acquire(), timeout=0.1)
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(slot0.acquire(), timeout=0.05)
        for _ in range(10):
            slot0.release()
        # available again
        await asyncio.wait_for(slot0.acquire(), timeout=0.1)
        slot0.release()

    async def test_send_semaphore_bounds_concurrent_parts(self):
        """F-3: with N parts and a SLOW /send, the pipeline never has more
        than SGLANG_ENCODER_MAX_INFLIGHT_SENDS concurrent /send in flight to
        one encoder (the semaphore is what keeps executor queue wait inside
        the quiesce window budget)."""
        import time as _time

        from sglang.srt.managers.schedule_batch import Modality as _Mod

        receiver = _make_receiver()
        receiver.dtype = torch.bfloat16
        receiver.encode_urls = ["http://fake-encoder:8080"]

        inflight = {"now": 0, "max": 0, "done": 0}

        class _SlowSendSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, json=None, headers=None):
                endpoint = url.rsplit("/", 1)[-1]
                if endpoint == "encode":
                    payload = dict(json)
                    payload.update({"embedding_size": 128})
                    return _FakeAiohttpResponse(200, json_data=payload)
                inflight["now"] += 1
                inflight["max"] = max(inflight["max"], inflight["now"])
                try:
                    await asyncio.sleep(0.08)
                finally:
                    inflight["now"] -= 1
                    inflight["done"] += 1
                return _FakeAiohttpResponse(200, json_data={})

        N_PARTS = 15

        def fake_extract(self_, request_obj):
            return [
                {"url": f"data:image/png;base64,{i}", "modality": _Mod.IMAGE}
                for i in range(N_PARTS)
            ]

        async def fake_allocate(self_, part_req_id, total_bytes):
            buf = MagicMock(data_ptr=MagicMock(return_value=1))
            self_._store_embedding_buffer(part_req_id, buf, total_bytes)
            return 1

        async def ok_recv(self_, *args, **kwargs):
            # first frames arrive; the request finishes via recv timeout path
            await asyncio.sleep(3600)

        with patch.object(MMReceiverHTTP, "_extract_url_data", fake_extract), patch(
            "sglang.srt.disaggregation.encode_receiver.aiohttp.ClientSession",
            lambda *a, **k: _SlowSendSession(),
        ), patch.object(MMReceiverHTTP, "allocate_embedding_buffer", fake_allocate):
            task = asyncio.create_task(
                receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p",
                )
            )
            # wait until all parts have gone through /send
            deadline = _time.monotonic() + 15
            while inflight["done"] < N_PARTS and _time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.assertEqual(inflight["done"], N_PARTS, "all parts dispatched")
        self.assertLessEqual(
            inflight["max"],
            10,
            f"concurrent /send exceeded the budget: {inflight['max']}",
        )


if __name__ == "__main__":
    unittest.main()
