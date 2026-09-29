import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestEPDMemoryRelease(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
