import os
import unittest
import openai
from openai import APIStatusError

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    DEFAULT_MODEL_NAME_FOR_GLM_MOE_TEST,
)

content_above_128 = "hello" * 1000
content_below_128 = "hello"

class TestChatResponse(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_GLM_MOE_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.context_length = 128
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--tp=8",
                "--cuda-graph-max-bs=32", # TODO: set to 1 will lead to IMA, should be fixed in the future
                "--glm-check-chat-prompt-length",
                "--max-prefill-tokens=128",
                "--context-length=128",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def setUp(self):
        self.client = openai.Client(
            base_url=self.base_url + "/v1",
            api_key=self.api_key,
        )
        self.model_name = self.client.models.list().data[0].id

    def test_streaming_chat_response_413(self):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content_above_128,
                    }
                ],
                stream=True,
            )
            chunks = list(response)
            self.fail(f"Expected APIStatusError with 413, but got response with {len(chunks)} chunks")
        except APIStatusError as e:
            self.assertEqual(e.status_code, 413, f"Expected 413, got {e.status_code}: {e.message}")

    def test_nonstreaming_chat_response_413(self):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content_above_128,
                    }
                ],
                stream=False,
            )
            self.fail(f"Expected APIStatusError with 413, but got response: {response}")
        except APIStatusError as e:
            self.assertEqual(e.status_code, 413, f"Expected 413, got {e.status_code}: {e.message}")

    def test_streaming_chat_response_200(self):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content_below_128,
                }
            ],
            stream=True,
        )
        chunks = []
        for chunk in response:
            if chunk.choices:
                chunks.append(chunk)
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")

    def test_nonstreaming_chat_response_200(self):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content_below_128,
                }
            ],
            stream=False,
        )
        # Verify response is successful and has expected structure
        self.assertIsNotNone(response.id)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)


if __name__ == "__main__":
    unittest.main()
