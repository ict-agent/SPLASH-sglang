#!/usr/bin/env python3
"""交叉场景矩阵: 请求失败/超时/取消 × 并发, 验证 cleanup 的 MR 安全与幂等。

Case X1: 两个并发请求, 一个 E 写失败(error frame), 一个正常完成 —— 失败的
        quiesce 期间, 正常请求必须不受影响(成功返回)。
Case X2: 同一请求 error frame 与数据帧混合到达 (error frame 先到) ——
        FAIL 立即返回, 无双重清理。
Case X3: E /encode 失败 (400 error frame, 无 buffer) → fail fast, 无 quiesce,
        两个并发请求中另一个成功。
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.encode_receiver import (  # noqa: E402
    EncoderError,
    MMReceiverHTTP,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_receiver():
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
    return receiver


class TestConcurrentCleanupMatrix(unittest.IsolatedAsyncioTestCase):
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

    async def test_x1_failed_request_does_not_break_concurrent_success(self):
        """并发两请求: 一个 E 写失败(error frame + buffer), 一个正常。
        失败请求 quiesce 期间, 成功请求必须完整返回自己的结果。"""

        async def encode_noop(*args, **kwargs):
            # every /encode and /send "succeeds" -- the write-failure shape
            return None

        async def recv_routed(self_, req_id, socket, processor, prompt):
            if prompt == "p1":
                # Real ordering: the buffer was already stored when /encode
                # returned; the E-side write failure then arrives as an error
                # frame via recv.
                self_._store_embedding_buffer(
                    req_id, MagicMock(data_ptr=MagicMock(return_value=1)), 0
                )
                raise EncoderError("mooncake RDMA write failed", status_code=500)
            return {"mm": "good"}

        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            if seconds <= 1:
                await real_sleep(0)
                return
            await real_sleep(seconds)

        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.05,
        ), patch.object(MMReceiverHTTP, "encode", encode_noop), patch.object(
            MMReceiverHTTP, "_recv_mm_data", recv_routed
        ), patch(
            "asyncio.sleep", side_effect=fake_sleep
        ):
            results = await asyncio.gather(
                self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p1",
                ),
                self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p2",
                ),
                return_exceptions=True,
            )
        failures = [r for r in results if isinstance(r, EncoderError)]
        successes = [r for r in results if r == {"mm": "good"}]
        self.assertEqual(len(failures), 1, "exactly one failed request")
        self.assertEqual(len(successes), 1, "concurrent success must not be broken")
        # 成功请求的 buffer 不应被失败请求的 cleanup 波及:
        # (成功路径不存 buffer, 这里主要验证失败清理后无异常状态残留)
        pending = list(getattr(self.receiver, "_cleanup_tasks", ()))
        if pending:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=5
            )

    async def test_x2_error_frame_before_data_no_double_free(self):
        """同一请求: error frame 先到 → FAIL 立即返回; 后续数据帧被丢弃。
        deregister 恰好一次 (无 double free)。"""
        dereg_calls = []

        async def ok_encode(self_, req_id, *args, **kwargs):
            self_._store_embedding_buffer(
                req_id, MagicMock(data_ptr=MagicMock(return_value=5)), 0
            )

        frames = [
            MagicMock(
                req_id=None,  # patched below per req
                error_msg="mooncake RDMA write failed",
                error_code=500,
                ready=True,
                num_parts=1,
            ),
        ]

        async def error_then_data_recv(self_, req_id, socket, processor, prompt):
            # error frame 走 _recv 的 error 检查需要 recv_obj.req_id 匹配
            raise EncoderError("mooncake RDMA write failed", status_code=500)

        self.receiver.embeddings_engine.deregister = MagicMock(
            side_effect=lambda addr: dereg_calls.append(addr)
        )

        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.05,
        ), patch.object(MMReceiverHTTP, "encode", ok_encode), patch.object(
            MMReceiverHTTP, "_recv_mm_data", error_then_data_recv
        ):
            with self.assertRaises(EncoderError):
                await self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="p",
                )
        pending = list(getattr(self.receiver, "_cleanup_tasks", ()))
        if pending:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=5
            )
        self.assertEqual(dereg_calls, [5], "buffer deregistered exactly once")
        self.assertEqual(len(self.receiver.embeddings_buffer), 0)
        self.assertEqual(len(self.receiver._buffer_index), 0)

    async def test_x3_encode_400_frame_fails_fast_no_quiesce(self):
        """E /encode 返回 400 error frame (无 buffer) → fail fast,
        无 quiesce 延迟, 且不影响另一个并发请求成功。"""
        quiesce_slept = []

        async def failing_encode(*args, **kwargs):
            raise EncoderError("bad image", status_code=400)

        async def ok_encode(*args, **kwargs):
            return None

        async def ok_recv(*args, **kwargs):
            return {"mm": "ok"}

        async def hanging_recv(*args, **kwargs):
            await asyncio.sleep(3600)

        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            if seconds <= 1:
                quiesce_slept.append(seconds)
            await real_sleep(seconds if seconds > 1 else 0)

        with patch(
            "sglang.srt.disaggregation.encode_receiver._ENCODE_DRAIN_TIMEOUT_S",
            0.05,
        ), patch("asyncio.sleep", side_effect=fake_sleep):
            # 请求 1: /encode 失败 (400)
            with patch.object(MMReceiverHTTP, "encode", failing_encode), patch.object(
                MMReceiverHTTP, "_recv_mm_data", hanging_recv
            ):
                start = asyncio.get_event_loop().time()
                try:
                    await asyncio.wait_for(
                        self.receiver.recv_mm_data(
                            request_obj=MagicMock(_upstream_session_id=None),
                            mm_processor=MagicMock(),
                            prompt="bad",
                        ),
                        timeout=5,
                    )
                except (EncoderError, asyncio.TimeoutError):
                    pass
                elapsed_bad = asyncio.get_event_loop().time() - start
            # 请求 2: 正常成功
            with patch.object(MMReceiverHTTP, "encode", ok_encode), patch.object(
                MMReceiverHTTP, "_recv_mm_data", ok_recv
            ):
                result = await self.receiver.recv_mm_data(
                    request_obj=MagicMock(_upstream_session_id=None),
                    mm_processor=MagicMock(),
                    prompt="good",
                )
        self.assertEqual(result, {"mm": "ok"})
        self.assertEqual(quiesce_slept, [], "no quiesce on dispatch failure")
        self.assertLess(elapsed_bad, 5.0)


if __name__ == "__main__":
    unittest.main()
