# ruff: noqa: E402

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPrefillAbortCleanup(CustomTestCase):
    @patch("sglang.srt.disaggregation.prefill.prepare_abort")
    @patch("sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group")
    def test_waiting_queue_peer_failure_aborts_and_releases_metadata(
        self, mock_poll, mock_prepare_abort
    ):
        sender = MagicMock()
        req = SimpleNamespace(
            rid="waiting-peer-failed",
            bootstrap_room=11,
            disagg_kv_sender=sender,
            metadata_buffer_index=4,
            return_logprob=False,
            time_stats=SimpleNamespace(trace_ctx=MagicMock()),
            finished=MagicMock(return_value=False),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting_queue = [req]
        scheduler.attn_cp_cpu_group = MagicMock()
        scheduler.attn_tp_cpu_group = MagicMock()
        scheduler.tp_rank = 0
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        scheduler.stream_output = MagicMock()
        scheduler.enable_metrics = False
        scheduler.enable_hicache_storage = False

        mock_poll.return_value = [KVPoll.Failed]

        Scheduler.resolve_waiting_queue_bootstrap(scheduler)

        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(req.metadata_buffer_index, -1)
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(4)
        scheduler.stream_output.assert_called_once_with([req], req.return_logprob)
        mock_prepare_abort.assert_called_once()

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    @patch("sglang.srt.managers.scheduler.prepare_abort")
    def test_pending_chunked_abort_releases_resources_at_scheduler_boundary(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        sender = MagicMock()
        req = SimpleNamespace(
            rid="chunked-abort",
            req_pool_idx=5,
            kv_committed_freed=False,
            metadata_buffer_index=3,
            disagg_kv_sender=sender,
            time_stats=SimpleNamespace(trace_ctx=MagicMock()),
            to_finish=object(),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._pending_chunked_abort_req = req
        scheduler.chunked_req = req
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        scheduler.enable_hicache_storage = False
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.supports_mamba.return_value = False
        scheduler.send_to_tokenizer = MagicMock()

        Scheduler.process_pending_chunked_abort(scheduler)

        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        self.assertIsNone(req.to_finish)
        self.assertEqual(req.metadata_buffer_index, -1)
        sender.abort.assert_called_once()
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.send_to_tokenizer.send_output.assert_called_once()
        mock_prepare_abort.assert_called_once_with(req, "Aborted")
        mock_release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )


if __name__ == "__main__":
    unittest.main()
