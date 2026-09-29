import asyncio
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import encode_server  # noqa: E402
from sglang.srt.disaggregation.encode_server import MMEncoder  # noqa: E402
from sglang.srt.managers.schedule_batch import Modality  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestEncoderDecodeMetadata(CustomTestCase):
    def test_metadata_only_requires_decode_role_and_mooncake(self):
        mooncake = SimpleNamespace(encoder_transfer_backend="mooncake")
        zmq = SimpleNamespace(encoder_transfer_backend="zmq_to_scheduler")

        self.assertTrue(
            encode_server._is_mooncake_metadata_only({"role": "decode"}, mooncake)
        )
        self.assertFalse(
            encode_server._is_mooncake_metadata_only({"role": "prefill"}, mooncake)
        )
        self.assertFalse(
            encode_server._is_mooncake_metadata_only({"role": "decode"}, zmq)
        )
        self.assertFalse(encode_server._is_mooncake_metadata_only({}, mooncake))

    def test_metadata_only_encode_skips_vit(self):
        encoder = object.__new__(MMEncoder)
        encoder.rank = 0
        encoder.model_type = "glm4v"
        encoder.metrics = None
        encoder.embedding_to_send = {}
        grid = torch.tensor([[1, 8, 8]])
        get_feature_fn = Mock(side_effect=AssertionError("ViT must not run"))
        encoder._process_mm_items = AsyncMock(
            return_value=(
                {
                    "image_grid_thw": grid,
                    "pixel_values": torch.zeros((64, 3)),
                },
                get_feature_fn,
                None,
            )
        )

        result = asyncio.run(
            encoder.encode_metadata(
                mm_items=["image"],
                modality=Modality.IMAGE,
                req_id="request",
                num_parts=1,
                part_idx=0,
            )
        )

        self.assertEqual(result, (0, 0, 0, None, None))
        get_feature_fn.assert_not_called()
        mm_data = encoder.embedding_to_send["request"]
        self.assertIsNone(mm_data.embedding)
        self.assertIsNone(mm_data.shape)
        torch.testing.assert_close(mm_data.grid_dim, grid)

    def test_metadata_only_releases_vit_semaphore(self):
        encoder = object.__new__(MMEncoder)
        encoder.rank = 0
        encoder.model_type = "glm4v"
        encoder.metrics = None
        encoder.embedding_to_send = {}
        grid = torch.tensor([[1, 8, 8]])
        release_fn = Mock()
        encoder._process_mm_items = AsyncMock(
            return_value=(
                {
                    "image_grid_thw": grid,
                    "pixel_values": torch.zeros((64, 3)),
                },
                Mock(),
                release_fn,
            )
        )

        result = asyncio.run(
            encoder.encode_metadata(
                mm_items=["image"],
                modality=Modality.IMAGE,
                req_id="request",
                num_parts=1,
                part_idx=0,
            )
        )

        self.assertEqual(result, (0, 0, 0, None, None))
        release_fn.assert_called_once()

    def test_metadata_only_encode_preserves_preprocess_error_status(self):
        cases = (
            (TimeoutError("timed out"), HTTPStatus.SERVICE_UNAVAILABLE),
            (ValueError("bad image"), HTTPStatus.BAD_REQUEST),
        )
        for error, expected_status in cases:
            with self.subTest(error=type(error).__name__):
                encoder = object.__new__(MMEncoder)
                encoder.rank = 0
                encoder.model_type = "glm4v"
                encoder.metrics = None
                encoder.embedding_to_send = {}
                encoder._process_mm_items = AsyncMock(side_effect=error)

                result = asyncio.run(
                    encoder.encode_metadata(
                        mm_items=["image"],
                        modality=Modality.IMAGE,
                        req_id="request",
                        num_parts=1,
                        part_idx=0,
                    )
                )

                self.assertEqual(result[-1], expected_status)
                self.assertEqual(
                    encoder.embedding_to_send["request"].error_code,
                    expected_status,
                )

    def test_decode_request_is_not_broadcast_or_fully_encoded(self):
        fake_encoder = SimpleNamespace(
            server_args=SimpleNamespace(encoder_transfer_backend="mooncake"),
            mm_global_cache=None,
            metrics=None,
            encode_metadata=AsyncMock(return_value=(0, 0, 0, None, None)),
            encode=AsyncMock(),
            send=AsyncMock(),
            embedding_to_send={"request": object()},
        )
        worker_socket = Mock()
        request = {
            "req_id": "request",
            "modality": "image",
            "role": "decode",
            "mm_items": ["image"],
            "num_parts": 1,
            "part_idx": 0,
            "prefill_host": "127.0.0.1",
            "embedding_port": 1234,
        }

        with (
            patch.object(encode_server, "encoder", fake_encoder),
            patch.object(encode_server, "send_sockets", [worker_socket]),
        ):
            asyncio.run(encode_server.handle_encode_request(request))

        fake_encoder.encode_metadata.assert_awaited_once()
        fake_encoder.encode.assert_not_awaited()
        worker_socket.send_pyobj.assert_not_called()
        fake_encoder.send.assert_awaited_once_with(
            req_id="request",
            prefill_host="127.0.0.1",
            embedding_port=1234,
            meta_only=True,
        )
        self.assertNotIn("request", fake_encoder.embedding_to_send)

    def test_decode_send_failure_releases_metadata(self):
        fake_encoder = SimpleNamespace(
            server_args=SimpleNamespace(encoder_transfer_backend="mooncake"),
            mm_global_cache=None,
            metrics=None,
            encode_metadata=AsyncMock(return_value=(0, 0, 0, None, None)),
            send=AsyncMock(side_effect=RuntimeError("send failed")),
            embedding_to_send={"request": object()},
        )
        request = {
            "req_id": "request",
            "modality": "image",
            "role": "decode",
            "mm_items": ["image"],
            "num_parts": 1,
            "part_idx": 0,
            "prefill_host": "127.0.0.1",
            "embedding_port": 1234,
        }

        with patch.object(encode_server, "encoder", fake_encoder):
            response = asyncio.run(encode_server.handle_encode_request(request))

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("request", fake_encoder.embedding_to_send)

    def test_decode_encode_failure_releases_metadata(self):
        fake_encoder = SimpleNamespace(
            server_args=SimpleNamespace(encoder_transfer_backend="mooncake"),
            mm_global_cache=None,
            metrics=None,
            encode_metadata=AsyncMock(
                return_value=(0, 0, 0, "metadata failed", 500)
            ),
            embedding_to_send={"request": object()},
        )
        request = {
            "req_id": "request",
            "modality": "image",
            "role": "decode",
            "mm_items": ["image"],
            "num_parts": 1,
            "part_idx": 0,
        }

        with patch.object(encode_server, "encoder", fake_encoder):
            response = asyncio.run(encode_server.handle_encode_request(request))

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("request", fake_encoder.embedding_to_send)

    def test_prefill_request_keeps_full_encode_path(self):
        fake_encoder = SimpleNamespace(
            server_args=SimpleNamespace(encoder_transfer_backend="mooncake"),
            mm_global_cache=None,
            metrics=None,
            encode=AsyncMock(return_value=(16, 2, 8, None, None)),
            encode_metadata=AsyncMock(),
        )
        worker_socket = Mock()
        request = {
            "req_id": "prefill",
            "modality": "image",
            "role": "prefill",
            "mm_items": ["prefill-image"],
            "num_parts": 1,
            "part_idx": 0,
        }

        with (
            patch.object(encode_server, "encoder", fake_encoder),
            patch.object(encode_server, "send_sockets", [worker_socket]),
        ):
            response = asyncio.run(encode_server.handle_encode_request(request))

        self.assertEqual(response.status_code, 200)
        worker_socket.send_pyobj.assert_called_once()
        fake_encoder.encode.assert_awaited_once_with(
            mm_items=["prefill-image"],
            modality=Modality.IMAGE,
            req_id="prefill",
            num_parts=1,
            part_idx=0,
        )
        fake_encoder.encode_metadata.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
