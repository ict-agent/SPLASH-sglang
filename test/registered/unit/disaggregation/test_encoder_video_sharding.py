import asyncio
import concurrent.futures
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.encode_receiver import (  # noqa: E402
    EmbeddingData,
    MMReceiverHTTP,
    MultiModalEmbeddingData,
)
from sglang.srt.disaggregation.encode_server import MMEncoder  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.schedule_batch import Modality  # noqa: E402
from sglang.srt.multimodal.processors.glm4v import (  # noqa: E402
    preprocess_video_frames_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _FakeResponse:
    status = 200

    async def json(self):
        return None


class _FakeClientSession:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers):
        self.calls.append((url, json, headers))
        return _FakeResponse()


class _FakeVideo:
    avg_fps = 2.0
    frame_shape = (4, 5)

    def __len__(self):
        return 12


class TestEncoderVideoSharding(CustomTestCase):
    def test_video_sampling_config_survives_receiver_extraction(self):
        receiver = object.__new__(MMReceiverHTTP)
        request = SimpleNamespace(
            image_data=None,
            video_data=[
                {
                    "url": "video.mp4",
                    "preprocess_kwargs": {
                        "fps": 1,
                        "max_frames": 240,
                        "max_image_tokens": 1200,
                    },
                }
            ],
            audio_data=None,
        )

        extracted = receiver._extract_url_data(request)

        self.assertEqual(
            extracted,
            [
                {
                    "url": {
                        "url": "video.mp4",
                        "fps": 1,
                        "max_frames": 240,
                        "max_image_tokens": 1200,
                    },
                    "modality": Modality.VIDEO,
                }
            ],
        )

    def test_receiver_preserves_video_frame_sequence_as_one_video(self):
        receiver = object.__new__(MMReceiverHTTP)
        frames = [
            {"url": "data:image/png;base64,AA==", "timestamp": "0.0"},
            {"url": "data:image/png;base64,AQ==", "timestamp": "0.5"},
        ]
        request = SimpleNamespace(
            image_data=None,
            video_data=[frames],
            audio_data=None,
        )

        extracted = receiver._extract_url_data(request)

        self.assertEqual(len(extracted), 1)
        self.assertIs(extracted[0]["url"], frames)
        self.assertEqual(extracted[0]["modality"], Modality.VIDEO)

    def test_video_shard_thresholds(self):
        receiver = object.__new__(MMReceiverHTTP)
        remote = {"url": "https://example.com/video.mp4", "max_frames": 239}

        with (
            envs.SGLANG_ENCODER_VIDEO_SHARD_MIN_MB.override(128),
            envs.SGLANG_ENCODER_VIDEO_SHARD_MIN_FRAMES_PER_ENCODER.override(80),
        ):
            self.assertFalse(
                receiver._should_shard_video(
                    {"url": remote}, num_encoders=3
                )
            )
            remote["max_frames"] = 240
            self.assertTrue(
                receiver._should_shard_video(
                    {"url": remote}, num_encoders=3
                )
            )
            self.assertFalse(
                receiver._should_shard_video(
                    {"url": [{"frame": 1}]}, num_encoders=3
                )
            )

        with tempfile.NamedTemporaryFile() as video:
            video.write(b"small-video")
            video.flush()
            item = {"url": video.name}
            with envs.SGLANG_ENCODER_VIDEO_SHARD_MIN_MB.override(1):
                self.assertFalse(receiver._should_shard_video(item, 2))
            with envs.SGLANG_ENCODER_VIDEO_SHARD_MIN_MB.override(0):
                self.assertTrue(receiver._should_shard_video(item, 2))

    def test_http_receiver_builds_one_request_per_encoder(self):
        receiver = object.__new__(MMReceiverHTTP)
        receiver.encoder_transfer_backend = "zmq_to_tokenizer"
        receiver.encode_urls = [
            "http://encoder-0",
            "http://encoder-1",
            "http://encoder-2",
        ]
        receiver.host = "prefill-host"
        receiver.meta_only = False
        fake_session = _FakeClientSession()

        with (
            envs.SGLANG_ENCODER_VIDEO_SHARD_MIN_MB.override(0),
            patch(
                "sglang.srt.disaggregation.encode_receiver.aiohttp.ClientSession",
                return_value=fake_session,
            ),
        ):
            asyncio.run(
                receiver.encode(
                    req_id="request",
                    mm_data=[
                        {
                            "url": "https://example.com/large.mp4",
                            "modality": Modality.VIDEO,
                        }
                    ],
                    embedding_port=1234,
                    endpoint_encode="encode",
                    endpoint_send="send",
                    num_items_assigned={Modality.VIDEO: [1, 0, 0]},
                )
            )

        payloads = [call[1] for call in fake_session.calls]
        self.assertEqual(len(payloads), 3)
        self.assertEqual([p["encoder_idx"] for p in payloads], [0, 1, 2])
        self.assertEqual([p["part_idx"] for p in payloads], [0, 1, 2])
        self.assertEqual([p["video_shard_idx"] for p in payloads], [0, 1, 2])
        self.assertTrue(all(p["video_num_shards"] == 3 for p in payloads))
        self.assertTrue(all(p["num_parts"] == 3 for p in payloads))

    def test_empty_embedding_shards_are_ignored(self):
        aggregate = MultiModalEmbeddingData(
            part_idx=0,
            num_parts=2,
            req_id="request",
            grid_dim=None,
            modality=Modality.VIDEO,
            embedding=torch.empty((0, 4)),
            embedding_shape=[0, 4],
        )
        aggregate.add(
            EmbeddingData(
                req_id="request",
                num_parts=2,
                part_idx=1,
                grid_dim=torch.tensor([[2, 1, 1]]),
                modality=Modality.VIDEO,
                embedding=torch.arange(8).reshape(2, 4),
            )
        )

        result = aggregate.get_embedding(is_concat=True)
        self.assertEqual(result[Modality.VIDEO].shape, (2, 4))
        torch.testing.assert_close(
            result[Modality.VIDEO], torch.arange(8).reshape(2, 4)
        )

    def test_contiguous_buffer_builds_one_view_per_modality(self):
        aggregate = MultiModalEmbeddingData(
            part_idx=0,
            num_parts=3,
            req_id="request",
            grid_dim=None,
            modality=Modality.IMAGE,
            embedding=None,
            embedding_shape=[1, 2],
        )
        aggregate.add(
            EmbeddingData(
                req_id="request",
                num_parts=3,
                part_idx=1,
                grid_dim=None,
                modality=Modality.IMAGE,
                embedding=None,
                embedding_shape=[2, 2],
            )
        )
        aggregate.add(
            EmbeddingData(
                req_id="request",
                num_parts=3,
                part_idx=2,
                grid_dim=None,
                modality=Modality.VIDEO,
                embedding=None,
                embedding_shape=[1, 2],
            )
        )
        values = torch.arange(8, dtype=torch.float32)
        raw_buffer = values.view(torch.uint8).flatten()

        result = aggregate.get_embedding_from_contiguous_buffer(
            raw_buffer, torch.float32
        )

        torch.testing.assert_close(
            result[Modality.IMAGE], values[:6].reshape(3, 2)
        )
        torch.testing.assert_close(
            result[Modality.VIDEO], values[6:].reshape(1, 2)
        )
        self.assertEqual(
            result[Modality.IMAGE].untyped_storage().data_ptr(),
            raw_buffer.untyped_storage().data_ptr(),
        )

    def test_predecoded_tensor_frames_remain_nhwc_tensor(self):
        frames = [
            {"frame_image": torch.zeros((3, 2, 4)), "timestamp": "0.0"},
            {"frame_image": torch.ones((3, 2, 4)), "timestamp": "0.5"},
        ]

        images, metadata = preprocess_video_frames_sync(frames)

        self.assertIsInstance(images, torch.Tensor)
        self.assertEqual(tuple(images.shape), (2, 2, 4, 3))
        self.assertTrue(images.is_contiguous())
        self.assertEqual(metadata["frames_indices"], [0, 1])

    def test_encoder_does_not_resample_predecoded_frames(self):
        encoder = object.__new__(MMEncoder)
        encoder.model_type = "glm4v"
        encoder.video_processor = SimpleNamespace(max_image_tokens=1200)
        encoder.vision_config = {}
        encoder.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        frames = [
            {"frame_image": torch.zeros((3, 2, 4)), "timestamp": "0.0"},
            {"frame_image": torch.ones((3, 2, 4)), "timestamp": "0.5"},
        ]
        loaded = concurrent.futures.Future()
        loaded.set_result(frames)
        encoder.submit_data_loading_tasks = Mock(return_value=([loaded], None))

        try:
            videos, kwargs = asyncio.run(
                encoder._flatten_and_load_videos([frames])
            )
        finally:
            encoder.io_executor.shutdown(wait=True)

        self.assertIsInstance(videos[0], torch.Tensor)
        self.assertFalse(kwargs["do_sample_frames"])

    def test_shard_decode_uses_contiguous_temporal_units(self):
        encoder = object.__new__(MMEncoder)
        encoder.video_processor = SimpleNamespace(max_image_tokens=1200)
        encoder.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        encoder.await_gpu_bytes = AsyncMock(return_value=(None, 0))
        encoder._release_admit_bytes = Mock()
        video = _FakeVideo()
        decoded = np.zeros((4, 4, 5, 3), dtype=np.uint8)

        try:
            with patch(
                "sglang.srt.disaggregation.encode_server.glm_decode_frames_at",
                return_value=decoded,
            ) as decode:
                videos, kwargs = asyncio.run(
                    encoder._shard_decode_single_video(
                        video,
                        {},
                        1,
                        shard_idx=1,
                        num_shards=4,
                        video_processor_kwargs={},
                        precomputed_indices=list(range(12)),
                    )
                )
        finally:
            encoder.io_executor.shutdown(wait=True)

        self.assertIs(videos[0], decoded)
        self.assertEqual(decode.call_args.args[1], [4, 5, 6, 7])
        self.assertEqual(kwargs["_shard_meta"]["start_unit"], 2)
        self.assertEqual(kwargs["_shard_meta"]["count"], 2)
        self.assertEqual(kwargs["max_image_tokens"], 400)


if __name__ == "__main__":
    unittest.main()
