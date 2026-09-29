import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class TestTokenizerManagerAbort(unittest.TestCase):
    def setUp(self):
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.server_args = SimpleNamespace(
            skip_tokenizer_init=False,
            weight_version="test",
        )
        self.manager.shm_owner_table = MagicMock()

    def test_abort_releases_request_state_before_waking_waiter(self):
        state = MagicMock()
        state.obj = SimpleNamespace(stream=False, return_logprob=False)
        state.output_ids = [1, 2]
        state.out_list = []
        state.shm_owner_handle = 17
        state.get_text.return_value = "partial"
        state.time_stats.get_e2e_latency.return_value = 1.5
        self.manager.rid_to_state = {"rid": state}

        self.manager._handle_abort_req(AbortReq(rid="rid"))

        self.assertNotIn("rid", self.manager.rid_to_state)
        self.manager.shm_owner_table.release.assert_called_once_with(17)
        self.assertIsNone(state.shm_owner_handle)
        state.event.set.assert_called_once_with()
        self.assertEqual(state.out_list[0]["meta_info"]["finish_reason"]["type"], "abort")

    def test_late_abort_for_removed_request_is_ignored(self):
        self.manager.rid_to_state = {}

        self.manager._handle_abort_req(AbortReq(rid="finished"))

        self.manager.shm_owner_table.release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
