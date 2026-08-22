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
from sglang.srt.models.glm4v import swiglu_clamped  # noqa: E402
from sglang.srt.multimodal.processors.base_processor import (  # noqa: E402
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.glm4v import (  # noqa: E402
    Glm4vImageProcessor,
)
from sglang.srt.utils.common import load_image, smart_to_rgb  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
