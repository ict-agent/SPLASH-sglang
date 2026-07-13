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

        _cuda_backend_enabled = set_cuda_backend is not None
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
        self._tmp_path = None
        self._source = source
        self._device = device
        if _BACKEND == "torchcodec":
            kwargs = {"dimension_order": "NHWC"}
            if device == "cuda" and _try_cuda_backend():
                kwargs["device"] = "cuda"
            from torchcodec.decoders import set_cuda_backend
            try:
                # set_cuda_backend is a no-op unless kwargs sets device="cuda".
                with set_cuda_backend(envs.SGLANG_VIDEO_CUDA_BACKEND.get()):
                    self._decoder = VideoDecoder(source, **kwargs)
            except RuntimeError:
                if "device" in kwargs:
                    logger.warning("CUDA video decoding failed, falling back to CPU.")
                    kwargs.pop("device")
                    self._decoder = VideoDecoder(source, **kwargs)
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

    @property
    def frame_shape(self) -> tuple:
        """(height, width) of decoded frames, from container metadata (no
        decode); falls back to decoding frame 0 if metadata lacks dimensions."""
        if _BACKEND == "torchcodec":
            md = self._decoder.metadata
            h = getattr(md, "height", None)
            w = getattr(md, "width", None)
            if h and w:
                return int(h), int(w)
        shape = self[0].shape  # HWC
        return int(shape[-3]), int(shape[-2])

    def get_frames_at(self, indices: list):
        """Return frames at given indices, shape (N, H, W, C), uint8.

        Returns a numpy array when decoding on CPU (preserves the existing
        contract for CPU callers), or a torch CUDA tensor when decoding on GPU
        (so the whole video pipeline can stay on-device, zero-copy to ViT)."""
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

    def close(self):
        """Explicitly release the decoder and clean up temporary files."""
        self._decoder = None
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
