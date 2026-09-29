# ruff: noqa: E402

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import zmq  # noqa: E402

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeCancel(unittest.TestCase):
    def test_send_cancel_to_prefill_uses_connect_context_manager(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        sock = MagicMock()
        entered = []
        exited = []

        @contextmanager
        def fake_connect(endpoint, is_ipv6=False):
            entered.append((endpoint, is_ipv6))
            try:
                yield sock
            finally:
                exited.append((endpoint, is_ipv6))

        manager._connect = fake_connect

        manager.send_cancel_to_prefill(
            [{"rank_ip": "127.0.0.1", "rank_port": 12345}], 17
        )

        self.assertEqual(entered, [("tcp://127.0.0.1:12345", False)])
        self.assertEqual(exited, entered)
        sock.send_multipart.assert_called_once_with(
            [MooncakeKVManager.CANCEL_TRANSFER_HEADER, b"17"],
            flags=zmq.NOBLOCK,
        )

    @patch("sglang.srt.disaggregation.mooncake.conn.time.sleep")
    def test_send_cancel_to_prefill_retries_on_backpressure(self, mock_sleep):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        attempts = []
        sockets = [MagicMock(), MagicMock()]
        sockets[0].send_multipart.side_effect = zmq.Again()

        @contextmanager
        def fake_connect(endpoint, is_ipv6=False):
            attempts.append((endpoint, is_ipv6))
            yield sockets[len(attempts) - 1]

        manager._connect = fake_connect

        manager.send_cancel_to_prefill(
            [{"rank_ip": "127.0.0.1", "rank_port": 12345}], 23
        )

        self.assertEqual(
            attempts,
            [
                ("tcp://127.0.0.1:12345", False),
                ("tcp://127.0.0.1:12345", False),
            ],
        )
        mock_sleep.assert_called_once_with(0.1)
        sockets[1].send_multipart.assert_called_once_with(
            [MooncakeKVManager.CANCEL_TRANSFER_HEADER, b"23"],
            flags=zmq.NOBLOCK,
        )


if __name__ == "__main__":
    unittest.main()
