import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sglang.srt.utils import common as common_utils
from sglang.srt.utils import sample_video_frames
from sglang.srt.utils import video_decoder as video_decoder_utils
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="stage-a-test-cpu")


class DummyVideo:
    def __init__(self, total_frames: int, avg_fps: float):
        self._frames = total_frames
        self._fps = avg_fps

    def __len__(self):
        return self._frames

    @property
    def avg_fps(self):
        return self._fps


@dataclass(kw_only=True)
class Case:
    frames: int
    avg_fps: float
    desired_fps: int
    max_frames: int
    expected_frames: list[int]
    description: str


# fmt: off
@pytest.mark.parametrize("case", [
    Case(
        frames=100, avg_fps=25.0, desired_fps=5, max_frames=200,
        expected_frames=[0, 5, 10, 15, 20, 26, 31, 36, 41, 46, 52, 57, 62, 67, 72, 78, 83, 88, 93, 99],
        description="capped by desired_fps"
    ),
    Case(
        frames=10, avg_fps=10.0, desired_fps=100, max_frames=5,
        expected_frames=[0, 2, 4, 6, 9],
        description="capped by max_frames"
    ),
    Case(
        frames=50, avg_fps=25.0, desired_fps=50, max_frames=200,
        expected_frames=list(range(50)),
        description="capped by total_frames"
    ),
    Case(
        frames=1, avg_fps=30.0, desired_fps=0, max_frames=0,
        expected_frames=[0],
        description="always sample at least 1 frame"
    )
],     ids=lambda c: c.description)
def test_sample_video_frames_lengths(case: Case):
    video = DummyVideo(case.frames, case.avg_fps)
    result = sample_video_frames(video, desired_fps=case.desired_fps, max_frames=case.max_frames)
    assert result == case.expected_frames


def test_load_video_forces_cpu_without_real_cuda():
    sentinel = object()
    with (
        patch.object(common_utils, "_normalize_video_input", return_value="video.mp4"),
        patch.object(common_utils, "is_cuda", return_value=False),
        patch.object(
            common_utils, "VideoDecoderWrapper", return_value=sentinel
        ) as decoder,
    ):
        assert common_utils.load_video("video.mp4", use_gpu=True) is sentinel

    decoder.assert_called_once_with("video.mp4", device="cpu")


def test_decord_reports_cpu_device_and_frame_shape():
    class FakeVideoReader:
        def __init__(self, source, ctx):
            self.source = source
            self.ctx = ctx

        def __getitem__(self, index):
            return np.zeros((7, 11, 3), dtype=np.uint8)

    decord = ModuleType("decord")
    decord.VideoReader = FakeVideoReader
    decord.cpu = lambda index: ("cpu", index)

    with (
        patch.object(video_decoder_utils, "_BACKEND", "decord"),
        patch.dict(sys.modules, {"decord": decord}),
    ):
        decoder = VideoDecoderWrapper("video.mp4", device="cuda")
        try:
            assert decoder.device == "cpu"
            assert decoder.frame_shape == (7, 11)
            assert decoder.frame_shape == (7, 11)
        finally:
            decoder.close()


def test_torchcodec_cuda_fallback_reports_cpu_device():
    calls = []

    class FakeVideoDecoder:
        def __init__(self, source, **kwargs):
            calls.append(kwargs.copy())
            if kwargs.get("device") == "cuda":
                raise RuntimeError("CUDA decoder unavailable")
            self.metadata = SimpleNamespace(height=13, width=17)

    class FakeCudaBackendContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    decoders = ModuleType("torchcodec.decoders")
    decoders.set_cuda_backend = lambda backend: FakeCudaBackendContext()
    torchcodec = ModuleType("torchcodec")
    torchcodec.decoders = decoders

    with (
        patch.object(video_decoder_utils, "_BACKEND", "torchcodec"),
        patch.object(video_decoder_utils, "_try_cuda_backend", return_value=True),
        patch.object(
            video_decoder_utils, "VideoDecoder", FakeVideoDecoder, create=True
        ),
        patch.dict(
            sys.modules,
            {"torchcodec": torchcodec, "torchcodec.decoders": decoders},
        ),
    ):
        decoder = VideoDecoderWrapper("video.mp4", device="cuda")
        try:
            assert decoder.device == "cpu"
            assert decoder.frame_shape == (13, 17)
        finally:
            decoder.close()

    assert calls == [
        {"dimension_order": "NHWC", "device": "cuda"},
        {"dimension_order": "NHWC"},
    ]

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
