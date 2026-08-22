import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from PIL import Image

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constrained.glm.escape import (  # noqa: E402
    EscapedSpecialTokens,
    reset_global_escaped_special_tokens,
    set_global_escaped_special_tokens,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentVideoFrameURL,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageContentVideoURL,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.models.glm4v import swiglu_clamped  # noqa: E402
from sglang.srt.multimodal.processors.base_processor import (  # noqa: E402
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.glm4v import (  # noqa: E402
    Glm4vImageProcessor,
    glm_budget_kwargs,
    glm_sample_frame_indices,
)
from sglang.srt.utils.common import load_image, load_video, smart_to_rgb  # noqa: E402
from sglang.srt.utils.hf_transformers.processor import (  # noqa: E402
    _escape_processor_special_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class TestGlmMultimodalCompatibility(CustomTestCase):
    def tearDown(self):
        reset_global_escaped_special_tokens()

    def test_clamped_swiglu_limits_both_halves(self):
        values = torch.tensor([[20.0, -20.0, 30.0, -30.0]])
        eager_fn = getattr(
            swiglu_clamped, "_torchdynamo_orig_callable", swiglu_clamped
        )

        actual = eager_fn(values, 10.0)
        expected = F.silu(torch.tensor([[10.0, -20.0]])) * torch.tensor(
            [[10.0, -10.0]]
        )

        torch.testing.assert_close(actual, expected)

    def test_smart_rgba_uses_contrasting_background(self):
        image = Image.new("RGBA", (3, 3), (255, 255, 255, 255))
        image.putpixel((1, 1), (255, 0, 0, 0))

        converted = smart_to_rgb(image)

        self.assertEqual(converted.mode, "RGB")
        self.assertEqual(converted.getpixel((1, 1)), (32, 32, 32))

    def test_load_image_smart_rgb_flag(self):
        image = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
        with envs.SGLANG_ENABLE_SMART_IMAGE_RGB.override(True):
            converted, size = load_image(image, gpu_image_decode=False)

        self.assertEqual(converted.mode, "RGB")
        self.assertEqual(converted.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(size, (2, 2))

    def test_glm_processor_uses_escaped_multimodal_tokens(self):
        mapping = {
            token: f"{token}<escaped>"
            for token in (
                "<|image|>",
                "<|video|>",
                "<|begin_of_image|>",
                "<|end_of_image|>",
                "<|begin_of_video|>",
                "<|end_of_video|>",
            )
        }
        set_global_escaped_special_tokens(
            EscapedSpecialTokens(mapping=mapping, seed=1, enabled=True)
        )
        config = SimpleNamespace(
            image_token_id=1,
            video_token_id=2,
            image_start_token_id=3,
            image_end_token_id=4,
            video_start_token_id=5,
            video_end_token_id=6,
        )

        with (
            patch.object(BaseMultimodalProcessor, "__init__", return_value=None),
            patch.object(
                MultimodalSpecialTokens, "build", return_value=Mock()
            ),
        ):
            processor = Glm4vImageProcessor(config, Mock(), Mock())

        self.assertEqual(processor.IMAGE_TOKEN, mapping["<|image|>"])
        self.assertEqual(processor.VIDEO_TOKEN, mapping["<|video|>"])
        self.assertEqual(
            processor.IMAGE_START_TOKEN, mapping["<|begin_of_image|>"]
        )

    def test_cached_processor_tokens_and_placeholder_are_escaped(self):
        mapping = {
            "<|image|>": "<|image|><escaped>",
            "<|video|>": "<|video|><escaped>",
            "<|begin_of_image|>": "<|begin_of_image|><escaped>",
        }
        set_global_escaped_special_tokens(
            EscapedSpecialTokens(mapping=mapping, seed=7, enabled=True)
        )
        processor = SimpleNamespace(
            chat_template="{{ '<|image|>' }}",
            image_token="<|image|>",
            video_token="<|video|>",
            glm_image_start_token="<|begin_of_image|>",
            glm_image_placeholder_token="<|placeholder|>",
        )

        _escape_processor_special_tokens(processor)

        self.assertIn(mapping["<|image|>"], processor.chat_template)
        self.assertEqual(processor.image_token, mapping["<|image|>"])
        self.assertEqual(processor.video_token, mapping["<|video|>"])
        self.assertTrue(
            processor.glm_image_placeholder_token.startswith("<|placeholder|><")
        )

    def test_mixed_image_video_offsets_are_disambiguated(self):
        processor = SimpleNamespace(
            IM_TOKEN_ID=1,
            VIDEO_START_TOKEN_ID=2,
            VIDEO_END_TOKEN_ID=3,
        )
        image = MultimodalDataItem(modality=Modality.IMAGE)
        video = MultimodalDataItem(modality=Modality.VIDEO)

        Glm4vImageProcessor.assign_mm_offsets(
            processor,
            [video, image],
            torch.tensor([9, 1, 1, 8, 2, 1, 1, 7, 1, 3, 6]),
            Mock(),
        )

        self.assertEqual(image.offsets, [(1, 2)])
        self.assertEqual(video.offsets, [(5, 6), (8, 8)])

    def test_epd_timestamps_include_fraction_and_unit(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, **_: [
            {"0.5 seconds": 50, "1.0 seconds": 100}[text]
        ]
        processor = SimpleNamespace(
            _tokenizer=tokenizer,
            IM_TOKEN_ID=1,
            VIDEO_TOKEN_ID=2,
            IMAGE_START_TOKEN_ID=3,
            IMAGE_END_TOKEN_ID=4,
            spatial_merge_size=1,
        )

        input_ids, offsets, modalities = (
            Glm4vImageProcessor.build_input_ids_with_timestamps(
                processor,
                [9, 2, 8],
                embeddings=None,
                img_grid_thw=None,
                video_grid_thw=torch.tensor([[2, 1, 1]]),
                video_timestamps=[[0.5, 1.0]],
            )
        )

        self.assertEqual(tokenizer.encode.call_args_list[0].args[0], "0.5 seconds")
        self.assertEqual(tokenizer.encode.call_args_list[1].args[0], "1.0 seconds")
        self.assertEqual(offsets, [(2, 2), (6, 6)])
        self.assertEqual(modalities, [Modality.VIDEO])
        self.assertIn(50, input_ids)
        self.assertIn(100, input_ids)

    def test_sampling_defaults_and_multi_video_budget(self):
        with (
            envs.SGLANG_GLM_VIDEO_FPS.override(2.0),
            envs.SGLANG_GLM_VIDEO_MAX_FRAMES.override(2048),
            envs.SGLANG_GLM_VIDEO_MAX_DURATION.override(0),
        ):
            indices = glm_sample_frame_indices(300, fps=30, duration=10)

        self.assertEqual(len(indices), 20)
        processor = SimpleNamespace(max_image_tokens=1200)
        self.assertEqual(
            glm_budget_kwargs(processor, count=3, split=True),
            {"max_image_tokens": 400},
        )
        self.assertEqual(
            glm_budget_kwargs(
                processor,
                user_max_image_tokens=600,
                count=3,
                split=True,
            ),
            {"max_image_tokens": 200},
        )

    def test_protocol_accepts_image_and_video_token_budgets(self):
        image = ChatCompletionMessageContentImageURL(
            url="image.png", max_image_tokens=512
        )
        video = ChatCompletionMessageContentVideoURL(
            url="video.mp4", fps=2, max_frames=64, max_image_tokens=1536
        )

        self.assertEqual(image.max_image_tokens, 512)
        self.assertEqual(video.max_image_tokens, 1536)
        self.assertEqual(video.fps, 2)
        self.assertEqual(video.max_frames, 64)

        framed = ChatCompletionMessageContentVideoPart(
            type="video_url",
            video_frame_url=[
                ChatCompletionMessageContentVideoFrameURL(
                    url="frame.png", timestamp="0.0"
                )
            ],
        )
        self.assertIsNone(framed.video_url)
        self.assertEqual(framed.video_frame_url[0].timestamp, "0.0")

    def test_video_frame_data_url_is_loaded(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (2, 2), "red").save(image_buffer, format="PNG")
        encoded = base64.b64encode(image_buffer.getvalue()).decode()
        frames = [
            {
                "url": f"data:image/png;base64,{encoded}",
                "timestamp": "0.0",
            }
        ]

        loaded = load_video(frames, use_gpu=False)

        self.assertEqual(loaded[0]["url"], "")
        self.assertIsInstance(loaded[0]["frame_image"], Image.Image)

    def test_release_features_clears_precomputed_embeddings(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.ones(1),
            precomputed_embeddings=torch.ones(1),
        )
        inputs = MultimodalInputs(mm_items=[item])

        inputs.release_features()

        self.assertIsNone(item.feature)
        self.assertIsNone(item.precomputed_embeddings)


if __name__ == "__main__":
    unittest.main()
