"""Unit tests for entry-side request-stage and outbound latency metrics."""

import pickle
import time

import sglang.srt.observability.req_time_stats as rts
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    RequestStage,
    SchedulerReqTimeStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class MockCollector:
    def __init__(self):
        self.observed = []
        self.labels = {}

    def observe_per_stage_req_latency(self, stage: str, latency: float):
        self.observed.append((stage, latency))


def _pickle_across(obj, sender_diff, receiver_diff):
    old = rts.global_diff_realtime_monotonic
    try:
        rts.global_diff_realtime_monotonic = sender_diff
        blob = pickle.dumps(obj)
        rts.global_diff_realtime_monotonic = receiver_diff
        return pickle.loads(blob)
    finally:
        rts.global_diff_realtime_monotonic = old


class TestEntryStageMetrics(CustomTestCase):
    def test_entry_stages_are_observed(self):
        self.assertTrue(RequestStage.TOKENIZE.metrics_is_observed)
        self.assertTrue(RequestStage.TOKENIZE_QUEUE.metrics_is_observed)
        self.assertTrue(RequestStage.TOKENIZE_EXEC.metrics_is_observed)
        self.assertTrue(RequestStage.API_SERVER_DISPATCH.metrics_is_observed)
        self.assertTrue(RequestStage.DPC_DISPATCH.metrics_is_observed)
        self.assertEqual(RequestStage.TOKENIZE_QUEUE.stage_name, "tokenize_queue")
        self.assertEqual(RequestStage.TOKENIZE_EXEC.stage_name, "tokenize_exec")

    def test_cross_process_propagation_with_dp_controller(self):
        api = APIServerReqTimeStats(disagg_mode=DisaggregationMode.DECODE)
        api.set_created_time(10.00)
        api.set_tokenize_queue_entry_time(10.05)
        api.set_tokenize_exec_start_time(10.07)
        api.set_tokenize_exec_finish_time(10.25)
        api.set_tokenize_finish_time(10.30)
        api.set_api_server_dispatch_time(10.31)

        sender_diff, dp_diff, scheduler_diff = 1000.0, 999.5, 999.0
        dp_in = _pickle_across(api, sender_diff, dp_diff)
        shift_to_dp = sender_diff - dp_diff
        self.assertAlmostEqual(dp_in.created_time, 10.00 + shift_to_dp, places=9)
        self.assertAlmostEqual(
            dp_in.tokenize_finish_time, 10.30 + shift_to_dp, places=9
        )

        dp = DPControllerReqTimeStats.new_from_obj(dp_in)
        self.assertAlmostEqual(
            dp.tokenize_queue_entry_time, 10.05 + shift_to_dp, places=9
        )
        self.assertAlmostEqual(
            dp.tokenize_exec_start_time, 10.07 + shift_to_dp, places=9
        )
        self.assertAlmostEqual(
            dp.tokenize_exec_finish_time, 10.25 + shift_to_dp, places=9
        )
        dp.set_dp_dispatch_time(10.35 + shift_to_dp)
        dp.set_dp_dispatch_finish_time(10.36 + shift_to_dp)

        sched_in = _pickle_across(dp, dp_diff, scheduler_diff)
        shift_to_scheduler = sender_diff - scheduler_diff
        self.assertAlmostEqual(
            sched_in.dpc_dispatch_time, 10.35 + shift_to_scheduler, places=9
        )

        sched = SchedulerReqTimeStats.new_from_obj(sched_in)
        collector = MockCollector()
        sched.set_metrics_collector(collector)
        sched.set_scheduler_recv_time(10.50 + shift_to_scheduler)

        observed = dict(collector.observed)
        self.assertAlmostEqual(observed["tokenize"], 0.30, places=9)
        self.assertAlmostEqual(observed["tokenize_queue"], 0.02, places=9)
        self.assertAlmostEqual(observed["tokenize_exec"], 0.18, places=9)
        self.assertAlmostEqual(observed["api_server_dispatch"], 0.05, places=9)
        self.assertAlmostEqual(observed["dpc_dispatch"], 0.15, places=9)

    def test_cross_process_propagation_without_dp_controller(self):
        api = APIServerReqTimeStats(disagg_mode=DisaggregationMode.PREFILL)
        api.set_created_time(20.00)
        api.set_tokenize_finish_time(20.40)
        api.set_api_server_dispatch_time(20.41)

        sched_in = _pickle_across(api, 500.0, 499.75)
        sched = SchedulerReqTimeStats.new_from_obj(sched_in)
        collector = MockCollector()
        sched.set_metrics_collector(collector)
        sched.set_scheduler_recv_time(20.55 + (500.0 - 499.75))

        observed = dict(collector.observed)
        self.assertAlmostEqual(observed["tokenize"], 0.40, places=9)
        self.assertAlmostEqual(observed["api_server_dispatch"], 0.15, places=9)
        self.assertNotIn("dpc_dispatch", observed)
        self.assertNotIn("tokenize_queue", observed)
        self.assertNotIn("tokenize_exec", observed)

    def test_no_observation_without_propagated_stamps(self):
        sched = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.DECODE)
        collector = MockCollector()
        sched.set_metrics_collector(collector)
        sched.set_scheduler_recv_time(1.0)
        self.assertEqual(collector.observed, [])

    def test_output_emit_time_stamped_and_passed_through(self):
        sched = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.DECODE)
        start = time.perf_counter()
        detok_copy = pickle.loads(pickle.dumps(sched))
        finish = time.perf_counter()
        self.assertLessEqual(start, detok_copy.output_emit_time)
        self.assertLessEqual(detok_copy.output_emit_time, finish)

        time.sleep(0.02)
        worker_copy = pickle.loads(pickle.dumps(detok_copy))
        self.assertAlmostEqual(
            worker_copy.output_emit_time, detok_copy.output_emit_time, places=9
        )
        self.assertEqual(sched.output_emit_time, 0.0)

    def test_scheduler_getstate_keeps_metrics_payload_when_enabled(self):
        sched = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.DECODE)
        sched.set_metrics_collector(MockCollector())
        sched.wait_queue_entry_time = 42.0
        copy = pickle.loads(pickle.dumps(sched))
        self.assertAlmostEqual(copy.wait_queue_entry_time, 42.0, places=9)
        self.assertGreater(copy.output_emit_time, 0.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
