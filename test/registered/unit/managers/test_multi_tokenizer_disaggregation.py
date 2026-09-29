import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import AbortReq  # noqa: E402
from sglang.srt.managers.multi_tokenizer_mixin import (  # noqa: E402
    MultiTokenizerRouter,
    TokenizerWorker,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class _FakeDispatcher:
    def __iadd__(self, other):
        return self


class TestMultiTokenizerDisaggregation(CustomTestCase):
    def test_worker_preserves_decode_role_without_starting_service(self):
        captured = {}

        def fake_parent_init(
            worker,
            server_args,
            port_args,
            *,
            start_disaggregation_service=True,
        ):
            captured["mode"] = server_args.disaggregation_mode
            captured["start_disaggregation_service"] = (
                start_disaggregation_service
            )
            worker.server_args = server_args
            worker.send_to_scheduler = Mock()
            worker._result_dispatcher = _FakeDispatcher()

        server_args = SimpleNamespace(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
        )
        port_args = SimpleNamespace(tokenizer_ipc_name="ipc:///tmp/test-tokenizer")

        with (
            patch.object(TokenizerManager, "__init__", new=fake_parent_init),
            patch("sglang.srt.managers.multi_tokenizer_mixin.FanOutCommunicator"),
            patch("setproctitle.setproctitle"),
        ):
            worker = TokenizerWorker(server_args, port_args)

        self.assertEqual(captured["mode"], "decode")
        self.assertFalse(captured["start_disaggregation_service"])
        self.assertEqual(server_args.disaggregation_mode, "decode")
        self.assertEqual(worker.disaggregation_mode.value, "decode")

    def test_disabled_service_still_creates_decode_mm_receiver(self):
        manager = object.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(
            disaggregation_mode="decode",
            language_only=True,
        )
        manager.model_config = SimpleNamespace(dtype="test-dtype")
        manager._start_disaggregation_service = False

        with (
            patch(
                "sglang.srt.managers.tokenizer_manager.start_disagg_service"
            ) as start_service,
            patch(
                "sglang.srt.managers.tokenizer_manager.create_mm_receiver"
            ) as create_receiver,
        ):
            manager.init_disaggregation()

        start_service.assert_not_called()
        self.assertIsNone(manager.bootstrap_server)
        create_receiver.assert_called_once_with(
            manager.server_args,
            dtype="test-dtype",
            is_decode_role=True,
        )

    def test_enabled_service_keeps_parent_behavior(self):
        manager = object.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(
            disaggregation_mode="prefill",
            language_only=False,
        )
        manager._start_disaggregation_service = True
        service = object()

        with patch(
            "sglang.srt.managers.tokenizer_manager.start_disagg_service",
            return_value=service,
        ) as start_service:
            manager.init_disaggregation()

        start_service.assert_called_once_with(manager.server_args)
        self.assertIs(manager.bootstrap_server, service)


class TestMultiTokenizerAbortRouting(CustomTestCase):
    @staticmethod
    def _make_router(*worker_ipcs: str) -> MultiTokenizerRouter:
        router = object.__new__(MultiTokenizerRouter)
        router.all_worker_ipcs = set(worker_ipcs)
        router.socket_mapping = Mock()
        return router

    def test_unrouted_abort_is_broadcast_to_all_registered_workers(self):
        router = self._make_router("ipc://worker-0", "ipc://worker-1")
        abort_req = AbortReq(rid="abort-rid")

        asyncio.run(router._distribute_result_to_workers(abort_req))

        self.assertEqual(router.socket_mapping.send_output.call_count, 2)
        router.socket_mapping.send_output.assert_has_calls(
            [
                call(
                    "ipc://worker-0",
                    abort_req,
                    is_tokenizer=True,
                ),
                call(
                    "ipc://worker-1",
                    abort_req,
                    is_tokenizer=True,
                ),
            ],
            any_order=True,
        )

    def test_routed_abort_keeps_single_worker_path(self):
        router = self._make_router("ipc://worker-0", "ipc://worker-1")
        abort_req = AbortReq(
            rid="abort-rid",
            http_worker_ipc="ipc://worker-1",
        )

        asyncio.run(router._distribute_result_to_workers(abort_req))

        router.socket_mapping.send_output.assert_called_once_with(
            "ipc://worker-1",
            abort_req,
        )

    def test_unrouted_abort_without_registered_workers_is_nonfatal(self):
        router = self._make_router()
        abort_req = AbortReq(rid="abort-rid")

        with self.assertLogs(
            "sglang.srt.managers.multi_tokenizer_mixin",
            level="WARNING",
        ):
            asyncio.run(router._distribute_result_to_workers(abort_req))

        router.socket_mapping.send_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
