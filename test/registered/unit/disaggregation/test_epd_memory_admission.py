import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestEPDMemoryAdmission(unittest.TestCase):
    def test_release_req_mm_inputs_after_final_prefill(self):
        mm_inputs = Mock()
        req = SimpleNamespace(session=None, multimodal_inputs=mm_inputs)

        self.assertTrue(Scheduler._release_req_mm_inputs(req))
        mm_inputs.release_features.assert_called_once_with()
        self.assertIsNone(req.multimodal_inputs)

    def test_release_req_mm_inputs_preserves_session_metadata_only(self):
        mm_inputs = Mock()
        req = SimpleNamespace(session=Mock(), multimodal_inputs=mm_inputs)

        self.assertTrue(Scheduler._release_req_mm_inputs(req))
        mm_inputs.release_features.assert_called_once_with()
        self.assertIs(req.multimodal_inputs, mm_inputs)

    def test_disagg_prefill_limit_counts_all_memory_retaining_queues(self):
        scheduler = object.__new__(Scheduler)
        scheduler.max_queued_requests = 3
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[Mock()])
        scheduler.waiting_queue = [Mock()]
        scheduler.disagg_prefill_inflight_queue = [Mock()]
        scheduler.send_to_tokenizer = Mock()

        mm_inputs = Mock()
        req = SimpleNamespace(
            rid="new-request",
            session=None,
            multimodal_inputs=mm_inputs,
            time_stats=SimpleNamespace(
                trace_ctx=SimpleNamespace(abort=Mock()),
            ),
        )

        self.assertTrue(scheduler._abort_on_disagg_prefill_queued_limit(req))

        abort_req, aborted_req = scheduler.send_to_tokenizer.send_output.call_args.args
        self.assertIs(aborted_req, req)
        self.assertEqual(abort_req.rid, req.rid)
        self.assertEqual(
            abort_req.finished_reason["status_code"],
            HTTPStatus.SERVICE_UNAVAILABLE,
        )
        req.time_stats.trace_ctx.abort.assert_called_once_with(
            abort_info=abort_req.finished_reason
        )
        mm_inputs.release_features.assert_called_once_with()
        self.assertIsNone(req.multimodal_inputs)

    def test_disagg_prefill_limit_allows_request_below_capacity(self):
        scheduler = object.__new__(Scheduler)
        scheduler.max_queued_requests = 4
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[Mock()])
        scheduler.waiting_queue = [Mock()]
        scheduler.disagg_prefill_inflight_queue = [Mock()]
        scheduler.send_to_tokenizer = Mock()

        self.assertFalse(scheduler._abort_on_disagg_prefill_queued_limit(Mock()))
        scheduler.send_to_tokenizer.send_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
