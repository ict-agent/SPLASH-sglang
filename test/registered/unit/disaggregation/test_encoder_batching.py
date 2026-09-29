import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.encode_receiver import (  # noqa: E402
    _split_mooncake_encode_requests,
)
from sglang.srt.disaggregation.encode_server import (  # noqa: E402
    EncoderScheduler,
    MMEncoder,
    PendingRequest,
)
from sglang.srt.managers.schedule_batch import Modality  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _request(part_idx, num_parts, items):
    return {
        "req_id": f"upstream_local_part_{part_idx}",
        "mm_items": items,
        "modality": "IMAGE",
        "num_parts": num_parts,
        "part_idx": part_idx,
    }


class _FakeEncoder:
    metrics = None

    def __init__(self):
        self.calls = []

    async def batch_encode(self, requests, modality):
        self.calls.append(([request["req_id"] for request in requests], modality))
        return [(1, 1, 1, None, None)] * len(requests)


class TestEncoderBatchScheduler(unittest.IsolatedAsyncioTestCase):
    async def test_multi_image_upstream_parts_are_bounded_by_batch_size(self):
        encoder = _FakeEncoder()
        scheduler = EncoderScheduler(encoder, [], max_batch_size=2)
        loop = asyncio.get_running_loop()
        for part_idx in range(5):
            await scheduler.pending_queue.put(
                PendingRequest(_request(part_idx, 5, [f"image-{part_idx}"]), loop)
            )

        batches = []
        while not scheduler.pending_queue.empty():
            batch = await scheduler._collect_batch()
            batches.append(batch)

        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(
            [[p.request["part_idx"] for p in batch] for batch in batches],
            [[0, 1], [2, 3], [4]],
        )

    async def test_dispatch_preserves_part_identity_and_order(self):
        encoder = _FakeEncoder()
        scheduler = EncoderScheduler(encoder, [], max_batch_size=8)
        loop = asyncio.get_running_loop()
        pending = [PendingRequest(_request(i, 3, [f"image-{i}"]), loop) for i in range(3)]

        await scheduler._dispatch_group(pending, Modality.IMAGE)

        self.assertEqual(
            encoder.calls,
            [
                (
                    [
                        "upstream_local_part_0",
                        "upstream_local_part_1",
                        "upstream_local_part_2",
                    ],
                    Modality.IMAGE,
                )
            ],
        )
        self.assertTrue(all(item.future.done() for item in pending))


class TestMultiImageLoadBalancerSplit(unittest.TestCase):
    def test_mooncake_splits_one_upstream_request_into_single_image_parts(self):
        grouped = [
            {
                "encoder_idx": 0,
                "mm_items": ["image-a", "image-b", "image-c"],
                "part_idx": 0,
                "req_id": "upstream_local_part_0",
                "modality": "IMAGE",
            }
        ]

        requests, _ = _split_mooncake_encode_requests("upstream", grouped, None)

        self.assertEqual(len(requests), 3)
        self.assertEqual([request["mm_items"] for request in requests], [["image-a"], ["image-b"], ["image-c"]])
        self.assertEqual([request["part_idx"] for request in requests], [0, 1, 2])
        self.assertEqual([request["num_parts"] for request in requests], [3, 3, 3])
        self.assertEqual(
            [request["req_id"] for request in requests],
            [
                "upstream_local_part_0",
                "upstream_local_part_1",
                "upstream_local_part_2",
            ],
        )


class TestMMEncoderBatchEncode(unittest.IsolatedAsyncioTestCase):
    def _make_encoder(self, slices):
        encoder = object.__new__(MMEncoder)
        encoder.metrics = None
        encoder.model_type = "glm4v"
        encoder.image_processor = SimpleNamespace(merge_size=2)
        encoder.server_args = SimpleNamespace(enable_prefix_mm_cache=False)
        encoder.rank = 0
        encoder.embedding_to_send = {}
        encoder.profiler = None
        encoder._process_mm_items = AsyncMock(
            return_value=(
                {
                    "pixel_values": torch.zeros((4 * len(slices), 3)),
                    "image_grid_thw": torch.tensor([[1, 2, 2]] * len(slices)),
                },
                object(),
                None,
            )
        )
        encoder._encode_missing = AsyncMock(return_value=slices)
        return encoder

    async def test_split_parts_from_multi_image_upstream_are_reassociated(self):
        slices = [torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0, 2.0]]), torch.tensor([[3.0, 3.0]])]
        encoder = self._make_encoder(slices)
        requests = [_request(i, 3, [f"image-{i}"]) for i in range(3)]

        results = await encoder.batch_encode(requests, Modality.IMAGE)

        self.assertEqual(len(results), 3)
        for part_idx, expected in enumerate(slices):
            stored = encoder.embedding_to_send[f"upstream_local_part_{part_idx}"]
            self.assertEqual(stored.part_idx, part_idx)
            self.assertEqual(stored.num_parts, 3)
            torch.testing.assert_close(stored.embedding, expected)

    async def test_grouped_multi_image_part_is_sliced_and_concatenated(self):
        slices = [torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0, 2.0]]), torch.tensor([[3.0, 3.0]])]
        encoder = self._make_encoder(slices)
        requests = [
            _request(0, 2, ["image-a", "image-b"]),
            _request(1, 2, ["image-c"]),
        ]

        results = await encoder.batch_encode(requests, Modality.IMAGE)

        self.assertEqual([result[1] for result in results], [2, 1])
        first = encoder.embedding_to_send["upstream_local_part_0"]
        second = encoder.embedding_to_send["upstream_local_part_1"]
        torch.testing.assert_close(first.embedding, torch.cat(slices[:2], dim=0))
        torch.testing.assert_close(second.embedding, slices[2])
        self.assertEqual(len(first.grid_dim), 2)
        self.assertEqual(len(second.grid_dim), 1)


if __name__ == "__main__":
    unittest.main()
