import os
import threading
import unittest
from collections import namedtuple
from unittest.mock import MagicMock, patch

from sglang.srt.observability.shm_monitor import (
    ShmUsage,
    read_shm_usage,
    start_shm_monitor_thread,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu", nightly=True)


class TestShmMonitor(unittest.TestCase):
    @patch("sglang.srt.observability.shm_monitor.os.statvfs")
    def test_read_shm_usage(self, mock_statvfs):
        StatVfs = namedtuple(
            "StatVfs", ["f_frsize", "f_bsize", "f_blocks", "f_bfree", "f_bavail"]
        )
        mock_statvfs.return_value = StatVfs(4096, 4096, 100, 25, 20)

        with patch.object(os, "name", "posix"):
            usage = read_shm_usage()

        self.assertEqual(usage.total_bytes, 409600)
        self.assertEqual(usage.used_bytes, 307200)
        self.assertEqual(usage.available_bytes, 81920)
        self.assertEqual(usage.usage_ratio, 0.75)

    @patch("sglang.srt.observability.shm_monitor.time.sleep")
    @patch("sglang.srt.observability.shm_monitor.read_shm_usage")
    def test_monitor_updates_metrics(self, mock_read, mock_sleep):
        usage = ShmUsage(100, 40, 60)
        mock_read.return_value = usage
        mock_sleep.side_effect = SystemExit
        update_metrics = MagicMock()
        original_hook = threading.excepthook
        threading.excepthook = lambda args: None

        thread = start_shm_monitor_thread(update_metrics, interval=3.0)
        thread.join(timeout=1.0)
        threading.excepthook = original_hook

        self.assertTrue(thread.daemon)
        update_metrics.assert_called_once_with(usage)
        mock_sleep.assert_called_once_with(3.0)

    @patch("sglang.srt.observability.shm_monitor.logger.warning")
    @patch("sglang.srt.observability.shm_monitor.time.sleep")
    @patch("sglang.srt.observability.shm_monitor.read_shm_usage")
    def test_monitor_failure_does_not_escape(
        self, mock_read, mock_sleep, mock_warning
    ):
        mock_read.side_effect = [OSError("statvfs failed"), SystemExit]
        mock_sleep.return_value = None
        original_hook = threading.excepthook
        threading.excepthook = lambda args: None

        thread = start_shm_monitor_thread(MagicMock(), interval=3.0)
        thread.join(timeout=1.0)
        threading.excepthook = original_hook

        mock_warning.assert_called_once()
        mock_sleep.assert_called_once_with(3.0)

    @patch("sglang.srt.observability.shm_monitor.logger.warning")
    @patch(
        "sglang.srt.observability.shm_monitor.threading.Thread.start",
        side_effect=RuntimeError("cannot start thread"),
    )
    def test_thread_start_failure_is_non_fatal(self, mock_start, mock_warning):
        thread = start_shm_monitor_thread(MagicMock())

        self.assertIsNone(thread)
        mock_warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
