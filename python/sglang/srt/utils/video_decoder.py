"""Unified video decoder: torchcodec preferred, decord as fallback."""

import logging

import numpy as np

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

try:
    from torchcodec.decoders import VideoDecoder

    _BACKEND = "torchcodec"
except (ImportError, RuntimeError):
    _BACKEND = "decord"


_cuda_backend_enabled: bool | None = None


def _try_cuda_backend() -> bool:
    """Try to enable torchcodec CUDA backend. Caches result after first call."""
    global _cuda_backend_enabled
    if _cuda_backend_enabled is not None:
        return _cuda_backend_enabled
    try:
        from torchcodec.decoders import set_cuda_backend

        set_cuda_backend("beta")
        _cuda_backend_enabled = True
    except Exception:
        _cuda_backend_enabled = False
    return _cuda_backend_enabled


class VideoDecoderWrapper:
    """Unified video decoder that uses torchcodec when available, decord as fallback.

    All frames are returned in NHWC uint8 numpy format for consistency.
    """

    def __init__(self, source, device: str = "cpu"):
        """source: file path (str) or video bytes.
        device: "cpu" or "cuda". GPU decoding only supported with torchcodec.
        """
        self._source_bytes = source if isinstance(source, bytes) else None
        self._source_path = source if isinstance(source, str) else None
        self._tmp_path = None
        self._source = source
        self._device = "cpu"
        self._frame_shape = None
        if _BACKEND == "torchcodec":
            kwargs = {"dimension_order": "NHWC"}
            if device == "cuda" and _try_cuda_backend():
                kwargs["device"] = "cuda"
            from torchcodec.decoders import set_cuda_backend

            try:
                # set_cuda_backend is a no-op unless kwargs sets device="cuda".
                with set_cuda_backend(envs.SGLANG_VIDEO_CUDA_BACKEND.get()):
                    self._decoder = VideoDecoder(source, **kwargs)
                if "device" in kwargs:
                    self._device = "cuda"
            except RuntimeError:
                if "device" in kwargs:
                    logger.warning("CUDA video decoding failed, falling back to CPU.")
                    kwargs.pop("device")
                    self._decoder = VideoDecoder(source, **kwargs)
                    self._device = "cpu"
                else:
                    raise
        else:
            from decord import VideoReader, cpu

            if isinstance(source, bytes):
                import os
                import tempfile

                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                try:
                    os.write(fd, source)
                finally:
                    os.close(fd)
                self._tmp_path = tmp_path
                self._decoder = VideoReader(tmp_path, ctx=cpu(0))
            else:
                self._decoder = VideoReader(source, ctx=cpu(0))

    def __len__(self):
        return len(self._decoder)

    @property
    def device(self) -> str:
        """The device actually used by the decoder after any fallback."""
        return self._device

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Return decoded frame height and width without retaining a frame."""
        if self._frame_shape is not None:
            return self._frame_shape

        height = width = None
        if _BACKEND == "torchcodec":
            metadata = self._decoder.metadata
            height = getattr(metadata, "height", None)
            width = getattr(metadata, "width", None)

        if height is None or width is None:
            frame = self._decoder[0]
            data = (
                frame.data
                if _BACKEND == "torchcodec" and hasattr(frame, "data")
                else frame
            )
            if len(data.shape) < 2:
                raise ValueError(f"Invalid decoded video frame shape: {data.shape}")
            height, width = data.shape[0], data.shape[1]

        self._frame_shape = (int(height), int(width))
        return self._frame_shape

    def __getitem__(self, idx):
        """Return single frame as NHWC uint8. numpy on CPU, torch tensor on CUDA."""
        if _BACKEND == "torchcodec":
            frame = self._decoder[idx]
            data = frame.data if hasattr(frame, "data") else frame
            return data if data.is_cuda else data.numpy()
        else:
            frame = self._decoder[idx]
            return frame.asnumpy() if hasattr(frame, "asnumpy") else np.array(frame)

    @property
    def avg_fps(self) -> float:
        if _BACKEND == "torchcodec":
            return self._decoder.metadata.average_fps
        else:
            return self._decoder.get_avg_fps()

    def get_frames_at(self, indices: list) -> np.ndarray:
        """Return frames at given indices as numpy array with shape (N, H, W, C)."""
        if _BACKEND == "torchcodec":
            data = self._decoder.get_frames_at(indices).data
            return data if data.is_cuda else data.numpy()
        else:
            return self._decoder.get_batch(indices).asnumpy()

    def get_frames_as_tensor(self, indices: list):
        """Return frames at given indices as a torch tensor (NHWC, uint8).

        On CPU the tensor is pinned for faster H2D; on CUDA it is returned
        as-is (already on device; pin_memory() is invalid for CUDA tensors)."""
        import torch

        if _BACKEND == "torchcodec":
            data = self._decoder.get_frames_at(indices).data
            return data if data.is_cuda else data.pin_memory()
        else:
            arr = self._decoder.get_batch(indices).asnumpy()
            return torch.from_numpy(arr).pin_memory()

    @property
    def source_bytes(self) -> bytes | None:
        """Return raw video bytes if available (needed for audio extraction)."""
        if self._source_bytes is not None:
            return self._source_bytes
        path = self._tmp_path or self._source_path
        if path is not None:
            import os

            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read()
        return None

    def close(self):
        """Explicitly release the decoder and clean up temporary files."""
        self._decoder = None
        self._source = None
        if self._tmp_path is not None:
            import os

            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            self._tmp_path = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
