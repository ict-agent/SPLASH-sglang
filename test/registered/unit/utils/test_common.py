import base64
import tempfile
import unittest
from array import array
from io import BytesIO
from unittest.mock import Mock, patch

import torch
from PIL import Image, UnidentifiedImageError

from sglang.srt.utils.common import (
    ImageData,
    flatten_arrays_to_int64_tensor,
    get_image_bytes,
    load_image,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestLoadImage(CustomTestCase):
    @staticmethod
    def _image_bytes(image_format: str) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (2, 3), color="red").save(buffer, format=image_format)
        return buffer.getvalue()

    def test_raw_base64_png_jpeg_gif_webp(self):
        for image_format in ("PNG", "JPEG", "GIF", "WEBP"):
            with self.subTest(image_format=image_format):
                encoded = base64.b64encode(self._image_bytes(image_format)).decode(
                    "ascii"
                )
                image, _ = load_image(encoded, gpu_image_decode=False)
                self.assertEqual(image.size, (2, 3))

    def test_openai_image_url_internal_image_data(self):
        encoded = base64.b64encode(self._image_bytes("PNG")).decode("ascii")
        image, _ = load_image(ImageData(url=encoded), gpu_image_decode=False)
        self.assertEqual(image.size, (2, 3))

    def test_standard_data_uri(self):
        raw = self._image_bytes("PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        self.assertEqual(get_image_bytes(data_uri), raw)

    @patch("sglang.srt.utils.common.requests.get")
    def test_http_and_https_urls(self, mock_get):
        raw = self._image_bytes("JPEG")
        response = Mock(content=raw)
        mock_get.return_value = response

        for url in (
            "http://example.com/image.jpg",
            "https://example.com/image.jpg",
        ):
            with self.subTest(url=url):
                self.assertEqual(get_image_bytes(url), raw)
                mock_get.assert_called_with(url, timeout=3)
        self.assertEqual(response.raise_for_status.call_count, 2)
        self.assertEqual(response.close.call_count, 2)

    def test_local_path(self):
        raw = self._image_bytes("PNG")
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            image_file.write(raw)
            image_file.flush()
            self.assertEqual(get_image_bytes(image_file.name), raw)

    def test_invalid_base64_is_safely_rejected(self):
        invalid = "not an image"
        with self.assertRaisesRegex(ValueError, "Invalid image data") as ctx:
            get_image_bytes(invalid)
        self.assertNotIn(invalid, str(ctx.exception))

    def test_oversized_base64_skips_path_check_and_is_rejected_as_non_image(self):
        encoded = "A" * (1024 * 1024)
        with patch("sglang.srt.utils.common.os.path.isfile") as isfile:
            with self.assertRaises(UnidentifiedImageError) as ctx:
                load_image(encoded, gpu_image_decode=False)
            isfile.assert_not_called()
        self.assertNotIn(encoded, str(ctx.exception))

    def test_valid_base64_non_image_is_rejected_without_echo(self):
        encoded = base64.b64encode(b"plain text").decode("ascii")
        with self.assertRaises(UnidentifiedImageError) as ctx:
            load_image(encoded, gpu_image_decode=False)
        self.assertNotIn(encoded, str(ctx.exception))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req fill_ids slices. Tests the full
    matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


if __name__ == "__main__":
    unittest.main()
