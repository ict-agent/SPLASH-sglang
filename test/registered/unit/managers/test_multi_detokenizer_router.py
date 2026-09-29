import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.multi_tokenizer_mixin import MultiDetokenizerRouter


class TestMultiDetokenizerRouter(unittest.TestCase):
    def setUp(self):
        self.workers = [f"worker-{i}" for i in range(4)]
        self.router = MultiDetokenizerRouter.__new__(MultiDetokenizerRouter)
        self.router.ipc_name_list = self.workers
        self.router.num_workers = len(self.workers)
        self.router.worker_loads = {worker: 0 for worker in self.workers}
        self.router.rid_to_worker = {}
        self.router.next_worker_index = 0

    def test_new_requests_are_evenly_distributed(self):
        assignments = [self.router._get_or_assign_worker(f"rid-{i}") for i in range(8)]

        self.assertEqual(assignments, self.workers * 2)
        self.assertEqual(
            self.router.worker_loads,
            {worker: 2 for worker in self.workers},
        )

    def test_request_stays_pinned_until_completion(self):
        worker = self.router._get_or_assign_worker("rid")

        self.assertEqual(self.router._get_or_assign_worker("rid"), worker)
        self.assertEqual(self.router.worker_loads[worker], 1)

        self.router._complete_requests(worker, ["rid"])
        self.assertNotIn("rid", self.router.rid_to_worker)
        self.assertEqual(self.router.worker_loads[worker], 0)

    def test_worker_without_completions_is_avoided(self):
        assignments = {
            f"rid-{i}": self.router._get_or_assign_worker(f"rid-{i}") for i in range(4)
        }
        stuck_worker = assignments["rid-0"]

        for rid in ("rid-1", "rid-2", "rid-3"):
            self.router._complete_requests(assignments[rid], [rid])

        new_assignments = [
            self.router._get_or_assign_worker(f"new-rid-{i}") for i in range(3)
        ]
        self.assertNotIn(stuck_worker, new_assignments)

    def test_completion_from_wrong_worker_is_ignored(self):
        assigned_worker = self.router._get_or_assign_worker("rid")
        wrong_worker = next(
            worker for worker in self.workers if worker != assigned_worker
        )

        self.router._complete_requests(wrong_worker, ["rid"])

        self.assertEqual(self.router.rid_to_worker["rid"], assigned_worker)
        self.assertEqual(self.router.worker_loads[assigned_worker], 1)
        self.assertEqual(self.router.worker_loads[wrong_worker], 0)


class TestDetokenizerCompletionAck(unittest.TestCase):
    def test_only_finished_requests_are_acknowledged(self):
        manager = DetokenizerManager.__new__(DetokenizerManager)
        manager.detokenizer_worker_ipc_name = "worker-0"
        manager.send_to_detokenizer_router = Mock()
        recv_obj = SimpleNamespace(
            rids=["running", "finished", "aborted"],
            finished_reasons=[None, {"type": "length"}, {"type": "abort"}],
        )

        manager.acknowledge_finished_requests(recv_obj)

        manager.send_to_detokenizer_router.send_pyobj.assert_called_once_with(
            ("worker-0", ["finished", "aborted"])
        )


if __name__ == "__main__":
    unittest.main()
