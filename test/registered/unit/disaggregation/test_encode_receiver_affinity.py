import hashlib
import unittest
from unittest.mock import AsyncMock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.encode_receiver import (  # noqa: E402
    MMReceiverHTTP,
    _encoder_request_headers,
    _split_mooncake_encode_requests,
    create_encoder_session_id,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.schedule_batch import Modality  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class _FakeClientSession:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if url.endswith("/encode"):
            payload = {key: value for key, value in json.items() if key != "mm_items"}
            payload.update(
                {"embedding_size": 16, "embedding_len": 1, "embedding_dim": 4}
            )
            return _FakeResponse(payload)
        return _FakeResponse(None)


class TestEncoderAffinity(unittest.IsolatedAsyncioTestCase):
    def test_session_id_header_is_controlled_by_env(self):
        with envs.GLM_ENABLE_ENCODER_SESSION_ID_HEADER.override(True):
            self.assertEqual(
                _encoder_request_headers("request", "encoder-session"),
                {"Request-Id": "request", "Session-Id": "encoder-session"},
            )

        with envs.GLM_ENABLE_ENCODER_SESSION_ID_HEADER.override(False):
            self.assertEqual(
                _encoder_request_headers("request", "encoder-session"),
                {"Request-Id": "request"},
            )

    def test_disabled_session_id_header_skips_hashing(self):
        grouped_requests = [
            {
                "encoder_idx": 0,
                "mm_items": ["image-a"],
                "num_parts": 1,
                "part_idx": 0,
                "req_id": "request_local_part_0",
                "modality": "IMAGE",
            }
        ]

        with envs.GLM_ENABLE_ENCODER_SESSION_ID_HEADER.override(False):
            with patch(
                "sglang.srt.disaggregation.encode_receiver.hashlib.sha256"
            ) as sha256:
                requests, session_ids = _split_mooncake_encode_requests(
                    req_id="request",
                    grouped_encode_requests=grouped_requests,
                    upstream_session_id="upstream",
                )

        self.assertEqual(len(requests), 1)
        self.assertEqual(session_ids, {})
        sha256.assert_not_called()

    def test_session_id_uses_url_or_base64_hash(self):
        media = "data:image/png;base64,abc"
        media_hash = hashlib.sha256(media.encode("utf-8")).hexdigest()[:16]

        self.assertEqual(create_encoder_session_id(None, media), media_hash)
        self.assertEqual(create_encoder_session_id("", media), media_hash)
        self.assertEqual(
            create_encoder_session_id("upstream-session", media),
            f"upstream-session_{media_hash}",
        )

    def test_builds_one_mooncake_request_per_media_item(self):
        mm_data = [
            {"url": "image-a", "modality": Modality.IMAGE},
            {"url": "image-b", "modality": Modality.IMAGE},
            {"url": "video-a", "modality": Modality.VIDEO},
        ]

        grouped_requests = [
            {
                "encoder_idx": 0,
                "mm_items": ["image-a", "image-b"],
                "num_parts": 2,
                "part_idx": 0,
                "req_id": "request_local_part_0",
                "modality": "IMAGE",
            },
            {
                "encoder_idx": 0,
                "mm_items": ["video-a"],
                "num_parts": 2,
                "part_idx": 1,
                "req_id": "request_local_part_1",
                "modality": "VIDEO",
            },
        ]
        with envs.GLM_ENABLE_ENCODER_SESSION_ID_HEADER.override(True):
            requests, session_ids = _split_mooncake_encode_requests(
                req_id="request",
                grouped_encode_requests=grouped_requests,
                upstream_session_id="upstream",
            )

        self.assertEqual(len(requests), 3)
        self.assertEqual([request["part_idx"] for request in requests], [0, 1, 2])
        self.assertEqual(
            [request["mm_items"] for request in requests],
            [["image-a"], ["image-b"], ["video-a"]],
        )
        self.assertTrue(all(request["num_parts"] == 3 for request in requests))
        self.assertEqual(
            session_ids,
            {
                index: create_encoder_session_id("upstream", item["url"])
                for index, item in enumerate(mm_data)
            },
        )

    async def test_encode_and_send_reuse_each_media_session_id(self):
        receiver = object.__new__(MMReceiverHTTP)
        receiver.encoder_transfer_backend = "mooncake"
        receiver.encode_urls = ["http://encoder-lb"]
        receiver.host = "prefill-host"
        receiver.meta_only = False
        receiver.embeddings_engine = type(
            "EmbeddingEngine", (), {"session_id": "rdma-session"}
        )()
        receiver.allocate_embedding_buffer = AsyncMock(return_value=1000)

        fake_session = _FakeClientSession()
        mm_data = [
            {"url": "image-a", "modality": Modality.IMAGE},
            {"url": "image-b", "modality": Modality.IMAGE},
        ]

        with envs.GLM_ENABLE_ENCODER_SESSION_ID_HEADER.override(True):
            with self.assertLogs(
                "sglang.srt.disaggregation.encode_receiver", level="INFO"
            ) as captured_logs:
                with patch(
                    "sglang.srt.disaggregation.encode_receiver.aiohttp.ClientSession",
                    return_value=fake_session,
                ):
                    await receiver.encode(
                        req_id="request",
                        mm_data=mm_data,
                        embedding_port=1234,
                        endpoint_encode="encode",
                        endpoint_send="send",
                        num_items_assigned={Modality.IMAGE: [2]},
                        upstream_session_id="upstream",
                    )

        encode_calls = [
            call for call in fake_session.calls if call["url"].endswith("/encode")
        ]
        send_calls = [
            call for call in fake_session.calls if call["url"].endswith("/send")
        ]
        self.assertEqual(len(encode_calls), 2)
        self.assertEqual(len(send_calls), 2)
        self.assertTrue(
            all(len(call["json"]["mm_items"]) == 1 for call in encode_calls)
        )

        encode_sessions_by_req = {
            call["headers"]["Request-Id"]: call["headers"]["Session-Id"]
            for call in encode_calls
        }
        send_sessions_by_req = {
            call["headers"]["Request-Id"]: call["headers"]["Session-Id"]
            for call in send_calls
        }
        self.assertEqual(send_sessions_by_req, encode_sessions_by_req)
        self.assertEqual(len(set(encode_sessions_by_req.values())), 2)
        log_output = "\n".join(captured_logs.output)
        self.assertIn("phase=encode", log_output)
        self.assertIn("phase=send", log_output)
        for session_id in encode_sessions_by_req.values():
            self.assertIn(f"session_id={session_id}", log_output)
        self.assertTrue(
            all(call["json"]["session_id"] == "rdma-session" for call in send_calls)
        )


if __name__ == "__main__":
    unittest.main()
