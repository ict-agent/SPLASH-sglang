import asyncio
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from enum import IntEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import torch
import zmq
import zmq.asyncio
from transformers import PretrainedConfig

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_mooncake_transfer_engine,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import Modality, Req
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ImageData
from sglang.srt.utils.hf_transformers_utils import get_processor
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)

# GLM Note: A 64-bit media digest keeps Session-Id headers compact while
# retaining sufficient collision resistance for encoder load-balancer affinity.
_ENCODER_MEDIA_HASH_HEX_LENGTH = 16


def _rdma_pool_max_bytes() -> int:
    return envs.SGLANG_MC_RDMA_POOL_MAX_MB.get() * 1024 * 1024


def _rdma_pool_max_buffers() -> int:
    return envs.SGLANG_MC_RDMA_POOL_MAX_BUFFERS.get()


def rdma_pool_enabled() -> bool:
    """Return whether both Mooncake RDMA pool limits enable pooling."""
    return _rdma_pool_max_bytes() > 0 and _rdma_pool_max_buffers() > 0


if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _grpc_target(url: str) -> str:
    if url.startswith("grpc://"):
        return url[len("grpc://") :]
    if url.startswith("grpcs://"):
        raise ValueError("grpcs:// is not supported; use grpc://")
    return url


def _normalize_embedding_ports(embedding_port):
    if embedding_port is None:
        return []
    if isinstance(embedding_port, list):
        return embedding_port
    return [embedding_port]


def _grpc_scheduler_receive_url(target, req_id, receive_url, receive_count):
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        stub.SchedulerReceiveUrl(
            sglang_encoder_pb2.SchedulerReceiveUrlRequest(
                req_id=req_id,
                receive_url=receive_url,
                receive_count=receive_count,
            ),
            timeout=timeout_secs,
        )
    finally:
        channel.close()


def _grpc_encode_request(target, encode_request):
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        response = stub.Encode(
            sglang_encoder_pb2.EncodeRequest(
                mm_items=encode_request["mm_items"],
                req_id=encode_request["req_id"],
                num_parts=encode_request["num_parts"],
                part_idx=encode_request["part_idx"],
                prefill_host=encode_request["prefill_host"],
                embedding_port=_normalize_embedding_ports(
                    encode_request["embedding_port"]
                ),
            ),
            timeout=timeout_secs,
        )
        return response
    finally:
        channel.close()


def _grpc_send_request(target, request_json):
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        stub.Send(
            sglang_encoder_pb2.SendRequest(
                req_id=request_json["req_id"],
                prefill_host=request_json["prefill_host"],
                embedding_port=request_json["embedding_port"],
                session_id=request_json["session_id"],
                buffer_address=request_json["buffer_address"],
            ),
            timeout=timeout_secs,
        )
    finally:
        channel.close()


class EncoderError(Exception):
    """Encoder dispatch/transfer failed; embeddings can never arrive.

    Raised by the encode task (connection failure, non-200 encoder response,
    RDMA buffer registration failure) and by the receive path (encoder error
    frame). Carries the HTTP status code to propagate to the client instead
    of letting the request wait out the full recv timeout and return 504.
    """

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def _log_task_exception(task):
    """Done-callback: retrieve and log a task's exception so asyncio never
    warns "Task exception was never retrieved"."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("encode task failed after embeddings were received: %s", exc)


class RdmaBufferPool:
    """Pool of long-lived, RDMA-registered CPU receive buffers."""

    def __init__(self, engine):
        self._engine = engine
        self._lock = threading.Lock()
        self._free = {}
        self._floor = 1 * 1024 * 1024
        self._max_total_bytes = _rdma_pool_max_bytes()
        self._max_buffers = _rdma_pool_max_buffers()
        self._total_bytes = 0
        self._total_count = 0
        self._warned_over = False

    def _size_class(self, nbytes: int) -> int:
        nbytes = max(int(nbytes), self._floor)
        return 1 << (nbytes - 1).bit_length()

    def acquire(self, nbytes: int) -> torch.Tensor:
        class_bytes = self._size_class(nbytes)
        with self._lock:
            free = self._free.get(class_bytes)
            if free:
                return free.pop()

            best_key = None
            for size, buffers in self._free.items():
                if (
                    size > class_bytes
                    and buffers
                    and (best_key is None or size < best_key)
                ):
                    best_key = size
            if best_key is not None:
                return self._free[best_key].pop()

        buffer = torch.empty(class_bytes, dtype=torch.uint8)
        ret = self._engine.register(buffer.data_ptr(), buffer.nbytes)
        if ret != 0:
            raise RuntimeError(
                f"mooncake register_memory failed (ret={ret}, bytes={class_bytes})"
            )

        with self._lock:
            self._total_bytes += class_bytes
            self._total_count += 1
            if (
                self._total_bytes > self._max_total_bytes
                or self._total_count > self._max_buffers
            ) and not self._warned_over:
                self._warned_over = True
                logger.warning(
                    "mooncake RDMA buffer pool over budget "
                    "(bytes=%d/%d, count=%d/%d); released buffers will be "
                    "deregistered to shrink. Increase "
                    "SGLANG_MC_RDMA_POOL_MAX_MB or "
                    "SGLANG_MC_RDMA_POOL_MAX_BUFFERS.",
                    self._total_bytes,
                    self._max_total_bytes,
                    self._total_count,
                    self._max_buffers,
                )
        return buffer

    def release(self, buffer: torch.Tensor) -> None:
        if buffer is None:
            return
        with self._lock:
            over_budget = (
                self._total_bytes > self._max_total_bytes
                or self._total_count > self._max_buffers
            )
            if not over_budget:
                self._free.setdefault(buffer.numel(), []).append(buffer)
                return
            self._total_bytes -= buffer.numel()
            self._total_count -= 1
        self._engine.deregister(buffer.data_ptr())

    def discard(self, buffer: torch.Tensor) -> None:
        """Deregister an aborted request's buffer instead of reusing it."""
        if buffer is None:
            return
        with self._lock:
            self._total_bytes -= buffer.numel()
            self._total_count -= 1
            remaining_bytes = self._total_bytes
            remaining_count = self._total_count
        logger.warning(
            "mooncake RDMA pool: discarding buffer (bytes=%d) on abort/timeout; "
            "pool now bytes=%d count=%d",
            buffer.numel(),
            remaining_bytes,
            remaining_count,
        )
        self._engine.deregister(buffer.data_ptr())


class RdmaRegRefcount:
    """Reference-count registrations of a shared Mooncake source tensor."""

    def __init__(self, engine):
        self._engine = engine
        self._lock = threading.Lock()
        self._refcounts = {}
        self._pinned_tensors = {}

    def acquire(self, tensor: torch.Tensor) -> int:
        addr = tensor.data_ptr()
        with self._lock:
            refcount = self._refcounts.get(addr, 0)
            if refcount == 0:
                ret = self._engine.register(addr, tensor.nbytes)
                if ret != 0:
                    raise RuntimeError(
                        f"mooncake register_memory failed (ret={ret}, "
                        f"bytes={tensor.nbytes})"
                    )
                self._pinned_tensors[addr] = tensor
            self._refcounts[addr] = refcount + 1
        return addr

    def release(self, addr: int) -> None:
        with self._lock:
            refcount = self._refcounts.get(addr, 0)
            if refcount <= 1:
                self._refcounts.pop(addr, None)
                self._pinned_tensors.pop(addr, None)
                self._engine.deregister(addr)
            else:
                self._refcounts[addr] = refcount - 1


class EmbeddingData:
    def __init__(
        self,
        req_id,
        num_parts,
        part_idx,
        grid_dim,
        modality,
        embedding=None,
        embedding_shape=None,
        error_msg=None,
        error_code=None,
        **kwargs,
    ):
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.grid_dim = grid_dim
        self.modality = modality
        self.embedding = embedding
        self.send_time = None
        self.created_at = time.perf_counter()
        self.dtype = embedding.dtype if embedding is not None else None
        if embedding_shape is not None:
            self.shape = embedding_shape
        else:
            self.shape = list(embedding.shape) if embedding is not None else None
        self.error_msg = error_msg
        self.error_code = error_code
        # Store additional metadata (e.g., video_timestamps for qwen3_vl)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_grid(self):
        """Get the grid dimension of the embedding, used for image/video/audio."""
        return self.grid_dim

    def get_embedding(self):
        return self.embedding

    def __repr__(self):
        return f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}) error_msg={self.error_msg}"

    def copy_without_embedding(self):
        new_data = EmbeddingData(
            req_id=self.req_id,
            num_parts=self.num_parts,
            part_idx=self.part_idx,
            grid_dim=self.grid_dim,
            modality=self.modality,
            embedding=None,
            embedding_shape=self.shape,
            error_msg=self.error_msg,
            error_code=self.error_code,
        )
        for key, value in self.__dict__.items():
            if key.startswith("_") or key == "embedding":
                continue
            setattr(new_data, key, value)
        return new_data


# Modality -> (list attr name, whether to flatten grid for that list)
_MODALITY_GRID_ATTRS = {
    Modality.IMAGE: ("img_grid_thw", False),
    Modality.VIDEO: ("video_grid_thw", False),
    Modality.AUDIO: ("audio_feature_lens", True),
}
_VIDEO_META_ATTRS = ("video_timestamps", "second_per_grid_ts")


def _cat_grid(dims, flatten_items=False):
    """Concatenate non-None grid entries; supports tensor/ndarray/list inputs."""

    def _to_tensor(g):
        if isinstance(g, torch.Tensor):
            return g.cpu() if g.is_cuda else g
        if isinstance(g, np.ndarray):
            return torch.from_numpy(g)
        return torch.as_tensor(g)

    valid = []
    for g in dims:
        if g is None:
            continue
        t = _to_tensor(g)
        if flatten_items:
            t = t.flatten()
        elif t.ndim == 0:
            # Keep cat semantics stable for scalar-like metadata.
            t = t.unsqueeze(0)
        valid.append(t)

    return torch.cat(valid, dim=0) if valid else None


class MultiModalEmbeddingData(EmbeddingData):
    def __init__(
        self,
        part_idx,
        num_parts,
        req_id,
        grid_dim,
        modality,
        embedding,
        embedding_shape,
        **kwargs,
    ):
        super().__init__(
            req_id,
            num_parts,
            part_idx,
            grid_dim,
            modality,
            embedding,
            embedding_shape,
            **kwargs,
        )
        self.img_grid_thw = [None] * num_parts
        self.video_grid_thw = [None] * num_parts
        self.audio_feature_lens = [None] * num_parts
        self.modality_list = [
            modality if part_idx == i else None for i in range(num_parts)
        ]
        self.ready_list = [i == part_idx for i in range(num_parts)]
        self.embedding_list = [
            embedding if i == part_idx else None for i in range(num_parts)
        ]
        self.embedding_shape_list = [
            embedding_shape if i == part_idx else None for i in range(num_parts)
        ]
        self.video_timestamps = [None] * num_parts
        self.second_per_grid_ts = [None] * num_parts

        self._set_part_grid(part_idx, modality, self.get_grid())
        if modality == Modality.VIDEO:
            self._set_video_meta_for_part(part_idx, kwargs)

    def _set_part_grid(self, part_idx, modality, grid):
        """Set the grid for one part according to modality (IMAGE/VIDEO/AUDIO)."""
        spec = _MODALITY_GRID_ATTRS.get(modality)
        if spec is None:
            raise ValueError(f"Invalid modality: {modality}")
        attr_name, flatten = spec
        value = grid.flatten() if flatten else grid
        getattr(self, attr_name)[part_idx] = value

    def _set_video_meta_for_part(self, part_idx, source):
        """Copy video_timestamps and second_per_grid_ts from source (dict or object)."""
        for attr_name in _VIDEO_META_ATTRS:
            val = (
                source.get(attr_name)
                if isinstance(source, dict)
                else getattr(source, attr_name, None)
            )
            if val is not None:
                getattr(self, attr_name)[part_idx] = val

    @classmethod
    def from_embedding_data(cls, embedding_data: EmbeddingData):
        """Create MultiModalEmbeddingData from an EmbeddingData instance."""
        # Only forward known optional attrs (e.g. video metadata) so they land on the instance
        extra = {}
        for attr in _VIDEO_META_ATTRS:
            val = getattr(embedding_data, attr, None)
            if val is not None:
                extra[attr] = val
        mm_data = cls(
            part_idx=embedding_data.part_idx,
            num_parts=embedding_data.num_parts,
            req_id=embedding_data.req_id,
            grid_dim=embedding_data.grid_dim,
            modality=embedding_data.modality,
            embedding=embedding_data.embedding,
            embedding_shape=embedding_data.shape,
            **extra,
        )
        mm_data.send_time = embedding_data.send_time
        return mm_data

    def __repr__(self):
        return f"MultiModalEmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}, modality={self.modality})"

    def get_embedding(self, is_concat=False):
        if is_concat:
            groups = defaultdict(list)
            for i, e in enumerate(self.embedding_list):
                # A cross-encoder video shard can be empty when the video has
                # fewer temporal units than encoders. It contributes no tokens.
                if e is not None and e.shape[0] > 0:
                    groups[self.modality_list[i]].append(e)
            return {
                modality: tensors[0] if len(tensors) == 1 else torch.cat(tensors)
                for modality, tensors in groups.items()
            }
        return self.embedding_list

    def get_embedding_from_contiguous_buffer(
        self,
        raw_buffer: torch.Tensor,
        dtype: torch.dtype,
        clone: bool = False,
    ) -> Dict[Modality, torch.Tensor]:
        """Build one view per modality over a contiguous Mooncake buffer."""
        if raw_buffer.device.type != "cpu" or raw_buffer.dtype != torch.uint8:
            raise ValueError(
                "Mooncake embedding buffer must be a CPU uint8 tensor, got "
                f"device={raw_buffer.device}, dtype={raw_buffer.dtype}"
            )

        element_size = torch.empty((), dtype=dtype).element_size()
        ranges = OrderedDict()
        byte_offset = 0
        current_modality = None
        closed_modalities = set()

        for index in range(self.num_parts):
            shape = self.embedding_shape_list[index]
            modality = self.modality_list[index]
            if shape is None:
                continue
            if modality is None:
                raise ValueError(f"Missing modality for embedding part {index}")
            if len(shape) == 0:
                raise ValueError(f"Invalid scalar embedding shape for part {index}")

            if modality != current_modality:
                if current_modality is not None:
                    closed_modalities.add(current_modality)
                if modality in closed_modalities:
                    raise ValueError(
                        "Mooncake embedding parts for a modality must be "
                        f"contiguous; {modality} appears in multiple ranges"
                    )
                current_modality = modality

            part_numel = math.prod(shape)
            part_bytes = part_numel * element_size
            next_byte_offset = byte_offset + part_bytes
            if next_byte_offset > raw_buffer.numel():
                raise ValueError(
                    "Mooncake embedding metadata exceeds the received buffer: "
                    f"part={index}, required_bytes={next_byte_offset}, "
                    f"buffer_bytes={raw_buffer.numel()}"
                )

            if shape[0] > 0:
                tail_shape = tuple(shape[1:])
                if modality not in ranges:
                    ranges[modality] = {
                        "start": byte_offset,
                        "end": next_byte_offset,
                        "rows": shape[0],
                        "tail_shape": tail_shape,
                    }
                else:
                    group = ranges[modality]
                    if tail_shape != group["tail_shape"]:
                        raise ValueError(
                            "Embedding parts for the same modality must have "
                            "matching trailing shapes, got "
                            f"{group['tail_shape']} and {tail_shape}"
                        )
                    if byte_offset != group["end"]:
                        raise ValueError(
                            "Mooncake embedding parts for a modality are not "
                            f"byte-contiguous: modality={modality}"
                        )
                    group["end"] = next_byte_offset
                    group["rows"] += shape[0]

            byte_offset = next_byte_offset

        result = {}
        for modality, group in ranges.items():
            embedding = (
                raw_buffer[group["start"] : group["end"]]
                .view(dtype)
                .reshape(group["rows"], *group["tail_shape"])
            )
            result[modality] = embedding.clone() if clone else embedding
        return result

    @property
    def ready(self):
        return sum(self.ready_list) == self.num_parts

    def get_mm_extra_meta(self):
        """Build kwargs for mm_processor.get_mm_data() from grid and optional video meta."""
        kwargs = {
            "img_grid_thw": _cat_grid(self.img_grid_thw),
            "video_grid_thw": _cat_grid(self.video_grid_thw),
            "audio_feature_lens": _cat_grid(
                self.audio_feature_lens, flatten_items=True
            ),
        }
        for attr in _VIDEO_META_ATTRS:
            lst = getattr(self, attr, None)
            if not lst:
                continue
            valid = [a for a in lst if a is not None]
            if valid:
                kwargs[attr] = list(itertools.chain(*valid))
        return kwargs

    def add(self, embedding_data: EmbeddingData):
        if self.req_id != embedding_data.req_id:
            logger.warning(
                f"Dropping embedding data with mismatched req_id: "
                f"expected {self.req_id}, got {embedding_data.req_id}"
            )
            return False
        assert not self.ready_list[embedding_data.part_idx]
        pid = embedding_data.part_idx
        self.ready_list[pid] = True
        self.modality_list[pid] = embedding_data.modality
        self.embedding_list[pid] = embedding_data.get_embedding()
        self.embedding_shape_list[pid] = embedding_data.shape
        self._set_part_grid(pid, embedding_data.modality, embedding_data.get_grid())
        if embedding_data.modality == Modality.VIDEO:
            self._set_video_meta_for_part(pid, embedding_data)


class WaitingImageRequestStatus(IntEnum):
    FAIL = -1
    PENDING = 0
    SUCCESS = 1
    TIMEOUT = -2


def create_part_req_id(original_req_id: str, part_idx: int) -> str:
    """Create a unique part request ID by appending part index suffix."""
    return f"{original_req_id}_local_part_{part_idx}"


def extract_original_req_id(part_req_id: str) -> str:
    """Extract the original request ID from a part request ID."""
    if "_local_part_" in part_req_id:
        return part_req_id.rsplit("_local_part_", 1)[0]
    return part_req_id


# GLM Note: Route duplicate URL/base64 media to the same encoder before the
# encoder-side, feature-based multimodal hash is available.
def create_encoder_session_id(
    upstream_session_id: Optional[str], media_identifier
) -> str:
    if isinstance(media_identifier, bytes):
        media_bytes = media_identifier
    elif isinstance(media_identifier, str):
        media_bytes = media_identifier.encode("utf-8")
    else:
        media_bytes = json.dumps(
            media_identifier,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    media_hash = hashlib.sha256(media_bytes).hexdigest()[
        :_ENCODER_MEDIA_HASH_HEX_LENGTH
    ]
    return f"{upstream_session_id}_{media_hash}" if upstream_session_id else media_hash


def _encoder_request_headers(req_id: str, encoder_session_id: Optional[str] = None):
    headers = {"Request-Id": req_id}
    if encoder_session_id and envs.GLM_ENABLE_ENCODER_SESSION_ID_HEADER.get():
        headers["Session-Id"] = encoder_session_id
    return headers


def _split_mooncake_encode_requests(
    req_id: str,
    grouped_encode_requests: List[Dict],
    upstream_session_id: Optional[str],
) -> Tuple[List[Dict], Dict[int, str]]:
    """Split grouped Mooncake requests into one request per media item."""
    encode_requests: List[Dict] = []
    encoder_session_ids = {}
    session_id_header_enabled = envs.GLM_ENABLE_ENCODER_SESSION_ID_HEADER.get()

    for grouped_request in grouped_encode_requests:
        for media_identifier in grouped_request["mm_items"]:
            part_idx = len(encode_requests)
            encode_request = grouped_request.copy()
            encode_request.update(
                {
                    "mm_items": [media_identifier],
                    "part_idx": part_idx,
                    "req_id": create_part_req_id(req_id, part_idx),
                }
            )
            encode_requests.append(encode_request)
            if session_id_header_enabled:
                encoder_session_ids[part_idx] = create_encoder_session_id(
                    upstream_session_id, media_identifier
                )

    total_num_parts = len(encode_requests)
    for encode_request in encode_requests:
        encode_request["num_parts"] = total_num_parts

    return encode_requests, encoder_session_ids


def calculate_modality_num_parts(modalities, num_items_assigned):
    """
    Calculate total number of parts and number of parts per modality.

    Args:
        modalities: List of modalities in order
        num_items_assigned: Dictionary mapping modality to list of assignment counts per encoder

    Returns:
        Tuple of (total_num_parts, modality_num_parts_dict)
        - total_num_parts: Total number of parts across all modalities
        - modality_num_parts: Dictionary mapping modality to number of parts for that modality
    """
    total_num_parts = 0
    modality_num_parts = {}
    for modality in modalities:
        num_items_assigned_modality = num_items_assigned.get(modality)
        num_parts = sum(1 for x in num_items_assigned_modality if x != 0)
        modality_num_parts[modality] = num_parts
        total_num_parts += num_parts
    return total_num_parts, modality_num_parts


# For zmq_to_scheduler
class WaitingImageRequest:
    def __init__(
        self,
        rid: str,
        recv_req: TokenizedGenerateReqInput,
        mm_processor,
        encoder_urls,
        host_name,
        receive_count,
    ):
        self.rid = rid
        self.recv_req = recv_req
        self.mm_inputs = None
        self.error = None
        self.thread = None
        self.mm_processor = mm_processor
        self.encoder_urls = encoder_urls
        self.host_name = host_name
        self.receive_count = receive_count
        self.num_items_assigned = recv_req.num_items_assigned
        self.embedding_port, self.recv_socket = get_zmq_socket_on_host(
            zmq.Context(), zmq.PULL, host=host_name
        )
        logger.info(f"Waiting for input {self.embedding_port = }")
        self.recv_embedding_data = None
        # ok=1 pending=0 fail=-1
        self.status = WaitingImageRequestStatus.PENDING
        self.error_msg = None
        self.error_code = None
        self.start_time = time.time()

    def send_encode_request(self):
        async def _send_single_request(session, url, payload):
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        msg = await response.text()
                        raise RuntimeError(
                            f"encoder {url} returned {response.status}: {msg}"
                        )
                    return await response.text()
            except Exception as e:
                logger.error(f"Failed to send request to {url}: {e}")
                raise

        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=1800)
            ) as session:
                tasks = []
                logger.info(f"{self.num_items_assigned = } ")

                # Calculate part_idx_offset similar to encode() method
                modalities = list(self.num_items_assigned.keys())
                _, modality_num_parts = calculate_modality_num_parts(
                    modalities, self.num_items_assigned
                )

                part_idx_offset = 0
                for modality in modalities:
                    assigned_nums = self.num_items_assigned[modality]
                    num_parts = modality_num_parts[modality]
                    cum_idx = 0
                    for idx, assigned_num in enumerate(assigned_nums):
                        if assigned_num == 0:
                            continue
                        part_idx = part_idx_offset + cum_idx
                        part_req_id = create_part_req_id(req_id, part_idx)
                        encoder_url = self.encoder_urls[idx]
                        target_url = f"{encoder_url}/scheduler_receive_url"
                        payload = {
                            "req_id": part_req_id,  # use part_req_id to match encode request
                            "receive_count": receive_count,
                            "receive_url": NetworkAddress(
                                host_name, embedding_port
                            ).to_host_port_str(),
                            "modality": modality.name,
                        }
                        logger.info(
                            f"Preparing to send to {target_url} with part_req_id={part_req_id}"
                        )
                        task = _send_single_request(session, target_url, payload)
                        tasks.append(task)
                        cum_idx += 1
                    part_idx_offset += num_parts

                if not tasks:
                    logger.info("No tasks to send.")
                    return
                logger.info(f"Concurrently sending {len(tasks)} requests...")
                results = await asyncio.gather(*tasks, return_exceptions=True)

                failed_results = []
                timed_out = False
                timeout_val = 1800
                for i, result in enumerate(results):
                    if isinstance(result, asyncio.TimeoutError):
                        timed_out = True
                        msg = (
                            f"Request {i} to encoder /scheduler_receive_url timed out "
                            f"({timeout_val}s) for req_id={req_id}"
                        )
                        logger.error(msg)
                        failed_results.append(msg)
                    elif isinstance(result, Exception):
                        msg = (
                            f"Request {i} to encoder /scheduler_receive_url failed for "
                            f"req_id={req_id}: {result}"
                        )
                        logger.error(msg, exc_info=result)
                        failed_results.append(msg)
                    else:
                        logger.debug(f"Request {i} succeeded.")

                if failed_results:
                    self.error_msg = (
                        f"Failed to notify encoder backends for request {req_id}: "
                        + "; ".join(failed_results)
                    )
                    self.error_code = (
                        HTTPStatus.REQUEST_TIMEOUT
                        if timed_out
                        else HTTPStatus.SERVICE_UNAVAILABLE
                    )
                    self.status = (
                        WaitingImageRequestStatus.TIMEOUT
                        if timed_out
                        else WaitingImageRequestStatus.FAIL
                    )
                    self.recv_socket.close()
                    logger.error(self.error_msg)
                    return

        asyncio.run(
            send_embedding_port(
                self.recv_req.rid,
                self.receive_count,
                self.host_name,
                self.embedding_port,
            )
        )

    def _try_recv_mm_data(self):
        if self.status != WaitingImageRequestStatus.PENDING:
            return
        while self.recv_embedding_data is None or not self.recv_embedding_data.ready:
            try:
                parts = self.recv_socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                # No data available yet, wait a bit and retry
                return
            recv_obj: EmbeddingData = pickle.loads(parts[0])
            if getattr(recv_obj, "error_msg", None) is not None:
                logger.warning(
                    f"Received error signal from encoder for {self.rid}: {recv_obj.error_msg} {recv_obj.error_code = }"
                )
                self.error_msg = recv_obj.error_msg
                self.error_code = recv_obj.error_code
                self.status = WaitingImageRequestStatus.FAIL
                self.recv_socket.close()
                return

            # Extract original req_id from part_req_id and drop stale payloads
            # that may arrive on a reused ZMQ port after a prior request aborted.
            original_req_id = extract_original_req_id(recv_obj.req_id)
            if original_req_id != self.recv_req.rid:
                logger.warning(
                    f"Dropping stale embedding data: expected rid={self.recv_req.rid}, "
                    f"got rid={recv_obj.req_id} (likely from ZMQ port reuse)"
                )
                continue
            recv_obj.req_id = original_req_id

            buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
            recv_obj.embedding = (
                torch.frombuffer(buffer, dtype=recv_obj.dtype)
                .reshape(recv_obj.shape)
                .clone()
            )

            if self.recv_embedding_data is None:
                self.recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                    recv_obj
                )
            else:
                self.recv_embedding_data.add(recv_obj)

        recv_embedding = self.recv_embedding_data.get_embedding(is_concat=True)
        mm_inputs = self.mm_processor.get_mm_data(
            self.recv_req.input_text,
            recv_embedding,
            **self.recv_embedding_data.get_mm_extra_meta(),
        )
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = mm_inputs.input_ids
        self.status = WaitingImageRequestStatus.SUCCESS
        self.recv_socket.close()


class WaitingImageRequestGrpc(WaitingImageRequest):
    def send_encode_request(self):
        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            tasks = []
            # gRPC image-only: flatten modality dict to flat list
            assigned = list(self.num_items_assigned.values())[0]
            logger.info(f"num_items_assigned={assigned}")

            for idx, assigned_num in enumerate(assigned):
                if assigned_num == 0:
                    continue
                encoder_url = self.encoder_urls[idx]
                receive_url = f"{host_name}:{embedding_port}"
                target_url = f"{encoder_url}/SchedulerReceiveUrl"
                logger.info(f"Preparing to send to {target_url}")
                tasks.append(
                    asyncio.to_thread(
                        _grpc_scheduler_receive_url,
                        _grpc_target(encoder_url),
                        req_id,
                        receive_url,
                        receive_count,
                    )
                )

            if not tasks:
                logger.info("No tasks to send.")
                return
            logger.info(f"Concurrently sending {len(tasks)} requests...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Request {i} failed: {result}")
                else:
                    logger.debug(f"Request {i} succeeded.")

        asyncio.run(
            send_embedding_port(
                self.recv_req.rid,
                self.receive_count,
                self.host_name,
                self.embedding_port,
            )
        )


def _determine_tensor_transport_mode(server_args):
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"


class MMReceiverBase(ABC):
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
        is_decode_role: bool = False,
    ):
        self.context = zmq.asyncio.Context(20)
        self.encoder_transfer_backend = server_args.encoder_transfer_backend
        self.encode_urls = server_args.encoder_urls
        self.host = get_local_ip_auto(server_args.host)
        self.is_decode_role = is_decode_role
        self.meta_only = is_decode_role and self.encoder_transfer_backend == "mooncake"
        if self.encoder_transfer_backend == "mooncake":
            self.dtype = dtype
            self.embeddings_engine = get_mooncake_transfer_engine()
            if self.embeddings_engine is None:
                from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                    init_mooncake_transfer_engine,
                )

                self.embeddings_engine = init_mooncake_transfer_engine(
                    hostname=self.host,
                    ib_device=(
                        server_args.disaggregation_ib_device
                        or server_args.mooncake_ib_device
                    ),
                )
            self.embeddings_buffer = dict()
            self._use_rdma_pool = rdma_pool_enabled()
            self._rdma_pool = (
                RdmaBufferPool(self.embeddings_engine)
                if self._use_rdma_pool
                else None
            )
        elif self.encoder_transfer_backend == "zmq_to_scheduler":
            self.pp_rank = pp_rank
            self.tp_rank = tp_rank
            self.tp_size = server_args.tp_size
            self.tp_group = tp_group
            self.nnodes = server_args.nnodes
            self.hostname = get_local_ip_auto()
            self.waiting_list: List[WaitingImageRequest] = []
            self.scheduler = scheduler
            self.wait_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
            if hf_config is not None:
                transport_mode = _determine_tensor_transport_mode(server_args)
                import_processors("sglang.srt.multimodal.processors")
                _processor = None
                try:
                    _processor = get_processor(
                        server_args.tokenizer_path,
                        tokenizer_mode=server_args.tokenizer_mode,
                        trust_remote_code=server_args.trust_remote_code,
                        revision=server_args.revision,
                        use_fast=not server_args.disable_fast_image_processor,
                        tokenizer_backend=server_args.tokenizer_backend,
                    )
                except ValueError as e:
                    error_message = str(e)
                    if "does not have a slow version" in error_message:
                        logger.info(
                            f"Processor {server_args.tokenizer_path} does not have a slow version. Automatically use fast version"
                        )
                        _processor = get_processor(
                            server_args.tokenizer_path,
                            tokenizer_mode=server_args.tokenizer_mode,
                            trust_remote_code=server_args.trust_remote_code,
                            revision=server_args.revision,
                            use_fast=True,
                            tokenizer_backend=server_args.tokenizer_backend,
                        )
                    else:
                        raise e

                # Skip mm_pool if not adaptive dispatch to encoder
                enable_adaptive_dispatch_to_encoder = (
                    server_args.enable_adaptive_dispatch_to_encoder
                )
                self.mm_processor = get_mm_processor(
                    hf_config,
                    server_args,
                    _processor,
                    transport_mode,
                    model_config=(
                        getattr(self.scheduler, "model_config", None)
                        if self.scheduler is not None
                        else None
                    ),
                    skip_mm_pool=not enable_adaptive_dispatch_to_encoder,
                )

    @abstractmethod
    def process_waiting_requests(self, recv_reqs):
        pass

    async def recv_mm_data(
        self, request_obj, mm_processor, prompt, need_wait_for_mm_inputs=True
    ):
        req_id = None
        encode_task = None
        recv_task = None
        try:
            if len(self.encode_urls) == 0 or not need_wait_for_mm_inputs:
                return None
            req_id = uuid.uuid4().hex
            embedding_port, recv_socket = get_zmq_socket_on_host(
                self.context, zmq.PULL, host=self.host
            )
            mm_data = self._extract_url_data(request_obj)
            encode_task = asyncio.create_task(
                self.encode(
                    req_id,
                    mm_data,
                    embedding_port,
                    "encode",
                    "send",
                    # GLM Note: Forward the upstream session only as far as the
                    # encoder HTTP client; scheduler request state does not need it.
                    upstream_session_id=getattr(
                        request_obj, "_upstream_session_id", None
                    ),
                )
            )
            recv_task = asyncio.create_task(
                self._recv_mm_data(req_id, recv_socket, mm_processor, prompt)
            )
            result = await asyncio.wait_for(
                self._race_encode_and_recv(encode_task, recv_task),
                timeout=envs.SGLANG_ENCODER_RECV_TIMEOUT.get(),
            )
            # Success: if the encode task is still finishing (e.g. draining
            # /send responses), make sure its eventual exception is retrieved.
            if not encode_task.done():
                encode_task.add_done_callback(_log_task_exception)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Embedding recv timeout for request {req_id}")
            await self._abort_encode_and_cleanup(encode_task, req_id, recv_task)
            return None
        except asyncio.CancelledError:
            # The awaiting request was cancelled (e.g. client disconnect during
            # the encode/E stage). Re-raise to preserve cancellation semantics.
            await self._abort_encode_and_cleanup(encode_task, req_id, recv_task)
            raise
        except BaseException:
            # EncoderError (fail-fast from the encode task or an encoder error
            # frame) and any unexpected error (pickle, mm_processor, zmq...):
            # stop both tasks and discard the buffer, then let the caller
            # surface the failure immediately instead of waiting out the
            # full recv timeout.
            await self._abort_encode_and_cleanup(encode_task, req_id, recv_task)
            raise

    @staticmethod
    async def _race_encode_and_recv(encode_task, recv_task):
        """Wait for recv while failing fast if the encode dispatch fails.

        The encode task normally finishes long before recv (it returns right
        after posting /send). Its success is a no-op here; its failure means
        the embeddings can never arrive, so surface the error immediately.
        """
        done, _ = await asyncio.wait(
            {encode_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if recv_task in done:
            # Prefer the recv outcome (data actually arrived or recv failed).
            return await recv_task
        # Only the encode task finished: raise its failure now, otherwise
        # keep waiting for recv.
        exc = encode_task.exception()
        if exc is not None:
            raise exc
        return await recv_task

    async def _abort_encode_and_cleanup(self, encode_task, req_id, recv_task=None):
        """Stop the encode/recv tasks, then discard req_id's RDMA buffer.

        The encode task is what allocates and registers the RDMA buffer, so it
        must be cancelled and drained BEFORE we discard -- otherwise it could
        allocate a fresh buffer after cleanup (re-leaking it), and a /send
        could still be in flight when we deregister the MR. Tasks that are
        already done are drained too, so their exceptions are always
        retrieved (no "Task exception was never retrieved" warnings).
        """
        for task in (encode_task, recv_task):
            if task is None:
                continue
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                # Expected: CancelledError from our cancel(), or the task's
                # own failure. We only need it to stop touching the buffer.
                pass
        if req_id is not None:
            self._cleanup_mooncake_buffer(req_id)

    def _cleanup_mooncake_buffer(self, req_id):
        if self.encoder_transfer_backend != "mooncake":
            return
        if not hasattr(self, "embeddings_buffer"):
            return
        embeddings = self.embeddings_buffer.pop(req_id, None)
        if embeddings is None:
            return
        try:
            if self._use_rdma_pool:
                self._rdma_pool.discard(embeddings)
            else:
                self.embeddings_engine.deregister(embeddings.data_ptr())
        except Exception:
            logger.exception(
                "mooncake: failed to discard/deregister buffer for req_id=%s",
                req_id,
            )

    async def _recv_mm_data(self, req_id, recv_socket, mm_processor, prompt):
        if req_id is None:
            return None

        recv_embedding = None

        recv_embedding_data: MultiModalEmbeddingData = None

        try:
            while recv_embedding_data is None or not recv_embedding_data.ready:
                parts = await recv_socket.recv_multipart(copy=False)
                if not parts:
                    continue
                recv_obj: EmbeddingData = pickle.loads(parts[0])
                if getattr(recv_obj, "error_msg", None) is not None:
                    error_code = getattr(recv_obj, "error_code", None)
                    logger.warning(
                        f"Encoder error for req_id={req_id}: {recv_obj.error_msg} "
                        f"error_code={error_code}"
                    )
                    self._cleanup_mooncake_buffer(req_id)
                    # Propagate the encoder's real error code (e.g. 500 for an
                    # RDMA write failure) instead of collapsing into a generic
                    # recv-timeout 504.
                    raise EncoderError(
                        f"Encoder error: {recv_obj.error_msg}",
                        status_code=int(error_code) if error_code else 500,
                    )
                logger.debug("recv_obj=%s", recv_obj)
                # Extract original req_id from part_req_id
                part_req_id = recv_obj.req_id
                original_req_id = extract_original_req_id(part_req_id)
                # Update recv_obj.req_id to original for aggregation
                recv_obj.req_id = original_req_id
                if self.encoder_transfer_backend == "zmq_to_tokenizer":
                    if len(parts) < 2:
                        logger.error(
                            "zmq_to_tokenizer expected 2-part message, got %d parts",
                            len(parts),
                        )
                        return None
                    buffer = (
                        parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
                    )
                    # Clone so we don't depend on ZMQ buffer after next recv.
                    recv_obj.embedding = (
                        torch.frombuffer(buffer, dtype=recv_obj.dtype)
                        .reshape(recv_obj.shape)
                        .clone()
                    )
                if recv_embedding_data is None:
                    recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                        recv_obj
                    )
                else:
                    recv_embedding_data.add(recv_obj)

            if self.encoder_transfer_backend == "mooncake" and not self.meta_only:
                if req_id not in self.embeddings_buffer:
                    logger.error(
                        "mooncake: embeddings_buffer missing req_id=%s", req_id
                    )
                    return None
                raw_buffer = self.embeddings_buffer.pop(req_id)
                if self._use_rdma_pool:
                    try:
                        recv_embedding = (
                            recv_embedding_data.get_embedding_from_contiguous_buffer(
                                raw_buffer, self.dtype, clone=True
                            )
                        )
                    finally:
                        self._rdma_pool.release(raw_buffer)
                else:
                    self.embeddings_engine.deregister(raw_buffer.data_ptr())
                    recv_embedding = (
                        recv_embedding_data.get_embedding_from_contiguous_buffer(
                            raw_buffer, self.dtype
                        )
                    )
            else:
                recv_embedding = recv_embedding_data.get_embedding(is_concat=True)

            mm_inputs = mm_processor.get_mm_data(
                prompt,
                recv_embedding,
                **recv_embedding_data.get_mm_extra_meta(),
            )
            return mm_inputs
        finally:
            recv_socket.close()

    def send_encode_request(self, obj):
        self._send_encode_request(obj)

    def _send_encode_request(self, obj):
        mm_data = self._extract_url_data(obj)
        if obj.rid is None:
            obj.rid = uuid.uuid4().hex
        if mm_data and self.encode_urls:
            logger.info(f"Processing {len(mm_data)} mm items for request {obj.rid}")
            obj.need_wait_for_mm_inputs = True

            num_items_assigned = self._assign_items_by_modality(
                mm_data, len(self.encode_urls)
            )
            obj.num_items_assigned = num_items_assigned
            encode_thread = threading.Thread(
                target=self._run_encode_in_thread,
                args=(
                    obj.rid,
                    mm_data,
                    "encode",
                    num_items_assigned,
                    None,
                ),
                daemon=True,
            )
            encode_thread.start()

    # For zmq_to_scheduler
    def _process_waiting_requests(self, recv_reqs, waiting_cls):
        new_recv_reqs = []
        for recv_req in recv_reqs:
            if (
                isinstance(recv_req, TokenizedGenerateReqInput)
                and recv_req.need_wait_for_mm_inputs is True
            ):
                waiting_req = waiting_cls(
                    rid=recv_req.rid,
                    recv_req=recv_req,
                    mm_processor=self.mm_processor,
                    encoder_urls=self.encode_urls,
                    host_name=self.hostname,
                    receive_count=self.tp_size,
                )
                waiting_req.send_encode_request()
                self.waiting_list.append(waiting_req)
            else:
                new_recv_reqs.append(recv_req)

        if len(self.waiting_list) == 0:
            return new_recv_reqs, []

        current_time = time.time()
        local_status = []
        for waiting_req in self.waiting_list:
            waiting_req._try_recv_mm_data()
            if current_time - waiting_req.start_time > self.wait_timeout:
                waiting_req.status = WaitingImageRequestStatus.TIMEOUT
            local_status.append(waiting_req.status)

        local_status = torch.tensor(local_status, device="cpu", dtype=torch.int32)

        torch.distributed.all_reduce(
            local_status,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group.cpu_group,
        )

        new_waiting = []
        abort_reqs = []
        for i, waiting_req in enumerate(self.waiting_list):
            status_value = local_status[i].item()
            if status_value == WaitingImageRequestStatus.SUCCESS:
                new_recv_reqs.append(waiting_req.recv_req)
            elif status_value == WaitingImageRequestStatus.FAIL:
                logger.error(
                    f"Waiting request {waiting_req.rid} failed: {waiting_req.error_msg} {waiting_req.error_code = }"
                )
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        waiting_req.error_msg,
                        waiting_req.error_code,
                    )
                )
            elif status_value == WaitingImageRequestStatus.TIMEOUT:
                logger.error(
                    f"Timed out waiting for image embeddings for request {waiting_req.rid}"
                )
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        f"Timeout waiting for image embedding after {self.wait_timeout}s",
                        HTTPStatus.REQUEST_TIMEOUT,
                    )
                )
            else:  # status_value == WaitingImageRequestStatus.PENDING
                new_waiting.append(waiting_req)

        self.waiting_list = new_waiting
        return new_recv_reqs, abort_reqs

    def _run_encode_in_thread(
        self, req_id, mm_data, endpoint_encode, num_items_assigned, embedding_port
    ):
        try:
            asyncio.run(
                self.encode(
                    req_id=req_id,
                    mm_data=mm_data,
                    embedding_port=embedding_port,
                    endpoint_encode=endpoint_encode,
                    endpoint_send=None,
                    num_items_assigned=num_items_assigned,
                )
            )
        except Exception as e:
            logger.error(f"Encode failed for request {req_id}: {e}", exc_info=True)

    def create_req(self, recv_req: TokenizedGenerateReqInput):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            return_logprob=recv_req.return_logprob,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            stream=recv_req.stream,
            lora_id=recv_req.lora_id,
            input_embeds=recv_req.input_embeds,
            custom_logit_processor=recv_req.custom_logit_processor,
            require_reasoning=recv_req.require_reasoning,
            return_hidden_states=recv_req.return_hidden_states,
            return_routed_experts=recv_req.return_routed_experts,
            routed_experts_start_len=recv_req.routed_experts_start_len,
            eos_token_ids=self.scheduler.model_config.hf_eos_token_id,
            bootstrap_host=recv_req.bootstrap_host,
            bootstrap_port=recv_req.bootstrap_port,
            bootstrap_room=recv_req.bootstrap_room,
            disagg_mode=self.scheduler.disaggregation_mode,
            routed_dp_rank=recv_req.routed_dp_rank,
            disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
            vocab_size=self.scheduler.model_config.vocab_size,
            priority=recv_req.priority,
            metrics_collector=(
                self.scheduler.metrics_collector
                if self.scheduler.enable_metrics
                else None
            ),
            http_worker_ipc=recv_req.http_worker_ipc,
            dllm_config=self.scheduler.dllm_config,
        )
        req.tokenizer = self.scheduler.tokenizer
        return req

    async def allocate_embedding_buffer(self, req_id, total_bytes):
        if self._use_rdma_pool:
            embeddings = self._rdma_pool.acquire(total_bytes)
        else:
            embeddings = torch.empty(total_bytes, dtype=torch.uint8)
            ret = self.embeddings_engine.register(
                embeddings.data_ptr(),
                embeddings.nbytes,
            )
            if ret != 0:
                # Do NOT store the buffer or hand the unregistered address to the
                # encoder -- its RDMA write would fail anyway and the request
                # would silently hang for the full recv timeout. Fail fast.
                raise EncoderError(
                    f"mooncake: receiver register failed for req_id={req_id} "
                    f"(ret={ret}, bytes={total_bytes})",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
        self.embeddings_buffer[req_id] = embeddings
        return embeddings.data_ptr()

    def _assign_items_by_modality(
        self, mm_data, encoder_num, random_shuffle=True
    ) -> Dict:
        """
        Assign multimodal items across encoders by modality with cross-modality load balancing.

        Args:
            mm_data: List of multimodal data items, each with a "modality" key
            encoder_num: Number of encoders
            random_shuffle: Whether to shuffle the encoder indices

        Returns:
            Dictionary mapping modality to list of assignment counts per encoder
            Format: {modality: [count_for_encoder_0, count_for_encoder_1, ...]}
        """
        encode_idx = list(range(encoder_num))
        if random_shuffle:
            random.shuffle(encode_idx)
        # Get unique modalities with order preserved
        modalities = list(dict.fromkeys(mm_item.get("modality") for mm_item in mm_data))
        # Use OrderedDict to explicitly maintain modality order
        num_items_assigned = OrderedDict()
        current_offset = 0

        for modality in modalities:
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.get("modality") == modality
            ]
            num_items = len(mm_data_modality)
            if num_items == 0:
                continue

            base = num_items // len(encode_idx)
            remainder = num_items % len(encode_idx)
            # Rotate assignments based on current_offset to balance load across modalities
            assignments = [0] * len(encode_idx)
            for i in range(len(encode_idx)):
                # keep shuffle order when assigning items to encoders
                pos_in_shuffled = (current_offset + i) % len(encode_idx)
                actual_encoder_idx = encode_idx[pos_in_shuffled]
                assignments[actual_encoder_idx] = base + (1 if i < remainder else 0)
            num_items_assigned[modality] = assignments
            current_offset = (current_offset + remainder) % len(encode_idx)

        return num_items_assigned

    @staticmethod
    def _video_max_frames(video_item) -> Optional[int]:
        """Return a request-provided sampled-frame cap, if present."""
        url = video_item.get("url")
        if isinstance(url, dict):
            max_frames = url.get("max_frames")
            if max_frames is not None:
                try:
                    return int(max_frames)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _video_is_framed(url) -> bool:
        """Whether the source is already a list of decoded video frames."""
        if isinstance(url, dict):
            url = url.get("url")
        return isinstance(url, (list, tuple))

    @staticmethod
    def _video_size_bytes(url) -> Optional[int]:
        """Return a local/in-memory video's size without doing network IO."""
        if isinstance(url, dict):
            url = url.get("url")
        try:
            if isinstance(url, (bytes, bytearray)):
                return len(url)
            if isinstance(url, str):
                if url.startswith("data:"):
                    return len(url)
                path = url[len("file://") :] if url.startswith("file://") else url
                if os.path.isfile(path):
                    return os.path.getsize(path)
        except Exception:
            pass
        return None

    def _should_shard_video(self, video_item, num_encoders) -> bool:
        """Whether one video is large enough to split across all encoders."""
        url = video_item.get("url")
        if self._video_is_framed(url):
            return False

        min_frames_per_encoder = (
            envs.SGLANG_ENCODER_VIDEO_SHARD_MIN_FRAMES_PER_ENCODER.get()
        )
        max_frames = self._video_max_frames(video_item)
        if (
            max_frames is not None
            and max_frames < num_encoders * min_frames_per_encoder
        ):
            return False

        min_bytes = envs.SGLANG_ENCODER_VIDEO_SHARD_MIN_MB.get() * 1024 * 1024
        size_bytes = self._video_size_bytes(url)
        if size_bytes is not None and size_bytes < min_bytes:
            return False
        return True

    def _extract_url_data(self, request_obj) -> List[Dict]:
        def is_video_frame_sequence(items):
            return (
                isinstance(items, (list, tuple))
                and bool(items)
                and all(
                    isinstance(item, dict)
                    and "url" in item
                    and "timestamp" in item
                    for item in items
                )
            )

        def flatten_mm_items(items, preserve_video_frames=False):
            if not isinstance(items, list):
                return [items]
            if preserve_video_frames and is_video_frame_sequence(items):
                return [items]

            flat = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    if preserve_video_frames and is_video_frame_sequence(item):
                        flat.append(item)
                    else:
                        flat.extend(
                            flatten_mm_items(
                                list(item),
                                preserve_video_frames=preserve_video_frames,
                            )
                        )
                else:
                    flat.append(item)
            return flat

        def to_raw_url(mm_item, modality):
            if isinstance(mm_item, ImageData):
                return mm_item.url
            if modality == Modality.VIDEO and hasattr(mm_item, "url"):
                payload = {"url": mm_item.url}
                preprocess_kwargs = getattr(mm_item, "preprocess_kwargs", None)
                if isinstance(preprocess_kwargs, dict):
                    payload.update(preprocess_kwargs)
                return payload
            if isinstance(mm_item, dict):
                if modality in (Modality.IMAGE, Modality.VIDEO):
                    payload = dict(mm_item)
                    preprocess_kwargs = payload.pop("preprocess_kwargs", None)
                    if isinstance(preprocess_kwargs, dict):
                        payload.update(preprocess_kwargs)
                    return payload
                # tolerate {"url": ...} shaped image/audio payloads
                return mm_item.get("url", mm_item)
            return mm_item

        mm_data = []
        for attr, modality in [
            ("image_data", Modality.IMAGE),
            ("video_data", Modality.VIDEO),
            ("audio_data", Modality.AUDIO),
        ]:
            mm_items = getattr(request_obj, attr, None)
            if mm_items:
                mm_items = flatten_mm_items(
                    mm_items, preserve_video_frames=modality == Modality.VIDEO
                )
                for mm_item in mm_items:
                    mm_data.append(
                        {
                            "url": to_raw_url(mm_item, modality),
                            "modality": modality,
                        }
                    )
        return mm_data


class MMReceiverHTTP(MMReceiverBase):
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
        is_decode_role: bool = False,
    ):
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
            is_decode_role=is_decode_role,
        )

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        return self._process_waiting_requests(recv_reqs, WaitingImageRequest)

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
        upstream_session_id=None,
    ):
        if len(mm_data) == 0:
            return

        # get unique modalities with order preserved
        modalities = [mm_item.get("modality") for mm_item in mm_data]
        modalities = list(dict.fromkeys(modalities))
        encode_requests = []

        if num_items_assigned is None:
            num_items_assigned = self._assign_items_by_modality(
                mm_data, len(self.encode_urls)
            )

        # A single sufficiently large video is split into contiguous temporal
        # shards, one per Encoder service. Other modalities retain item-based
        # distribution.
        num_encoders = len(self.encode_urls)
        video_items = [m for m in mm_data if m.get("modality") == Modality.VIDEO]
        shard_video = (
            len(video_items) == 1
            and num_encoders > 1
            and self._should_shard_video(video_items[0], num_encoders)
        )

        def _base_payload(part_idx, encoder_idx, mm_items, modality):
            return {
                "encoder_idx": encoder_idx,
                "mm_items": mm_items,
                "part_idx": part_idx,
                "req_id": create_part_req_id(req_id, part_idx),
                "modality": modality.name,
                "prefill_host": self.host,
                "embedding_port": embedding_port,
                "role": "decode" if self.meta_only else "prefill",
            }

        part_idx = 0
        for modality in modalities:
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.get("modality") == modality
            ]
            if modality == Modality.VIDEO and shard_video:
                url = mm_data_modality[0].get("url")
                for shard_idx in range(num_encoders):
                    payload = _base_payload(part_idx, shard_idx, [url], modality)
                    payload["video_num_shards"] = num_encoders
                    payload["video_shard_idx"] = shard_idx
                    encode_requests.append(payload)
                    part_idx += 1
            else:
                assigned = num_items_assigned.get(modality)
                cum_num_items = 0
                for encoder_idx, assigned_num in enumerate(assigned):
                    if assigned_num == 0:
                        continue
                    items = [
                        mm_item.get("url")
                        for mm_item in mm_data_modality[
                            cum_num_items : cum_num_items + assigned_num
                        ]
                    ]
                    encode_requests.append(
                        _base_payload(part_idx, encoder_idx, items, modality)
                    )
                    part_idx += 1
                    cum_num_items += assigned_num

        total_num_parts = len(encode_requests)
        for encode_request in encode_requests:
            encode_request["num_parts"] = total_num_parts

        encoder_session_ids = {}
        if self.encoder_transfer_backend == "mooncake":
            # GLM Note: Give every RDMA media item its own HTTP request and LB
            # session so independent images/videos can execute on different encoders.
            encode_requests, encoder_session_ids = _split_mooncake_encode_requests(
                req_id, encode_requests, upstream_session_id
            )
            total_num_parts = len(encode_requests)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=1800
            )  # Add timeout for request reliability
        ) as session:
            async def post_encoder_request(payload, endpoint, phase):
                encoder_idx = payload["encoder_idx"]
                encoder_session_id = encoder_session_ids.get(payload["part_idx"])
                url = f"{self.encode_urls[encoder_idx]}/{endpoint}"
                headers = _encoder_request_headers(payload["req_id"], encoder_session_id)
                # GLM Note: Log encoder routing metadata so operators can verify
                # per-media parallel dispatch and Session-Id affinity in production.
                logger.info(
                    "Sending request to encoder: phase=%s url=%s req_id=%s "
                    "part_idx=%s num_parts=%s modality=%s encoder_idx=%s "
                    "session_id=%s",
                    phase,
                    url,
                    payload["req_id"],
                    payload["part_idx"],
                    payload["num_parts"],
                    payload["modality"],
                    encoder_idx,
                    encoder_session_id,
                )
                return await session.post(url, json=payload, headers=headers)

            # Send encode requests. Session-Id provides media affinity while
            # Request-Id retains the unique request/part identity.
            tasks = [
                post_encoder_request(encode_request, endpoint_encode, "encode")
                for encode_request in encode_requests
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for response in responses:
                if isinstance(response, Exception):
                    # Fail fast: surface the dispatch failure to recv_mm_data
                    # instead of silently returning and letting the request
                    # wait out the full recv timeout (180s pile-up).
                    raise EncoderError(
                        f"Encoder request failed for {req_id}: {response}",
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    ) from response
                if response.status != 200:
                    try:
                        err_data = await response.json()
                        msg = err_data.get("message", "Unknown encoder error")
                    except Exception:
                        msg = await response.text()

                    raise EncoderError(
                        f"Encoder returned error {response.status}: {msg}",
                        status_code=response.status,
                    )
            response_json_list_unsort = [
                await response.json() for response in responses
            ]

            if self.meta_only:
                return

            # zmq backend: return is None
            if None in response_json_list_unsort:
                return

            # mooncake backend: send bootstrap info

            embedding_size_list_sort = [None for _ in range(total_num_parts)]
            response_json_list_sort = [None for _ in range(total_num_parts)]
            for response_json in response_json_list_unsort:
                idx = response_json["part_idx"]
                embedding_size_list_sort[idx] = response_json["embedding_size"]
                response_json_list_sort[idx] = response_json

            total_embedding_bytes = sum(
                s for s in embedding_size_list_sort if s is not None
            )
            offset = 0
            metadata_tasks = []
            buffer_address = await self.allocate_embedding_buffer(
                req_id,
                total_embedding_bytes,
            )
            for idx in range(len(tasks)):
                response_json = response_json_list_sort[idx]
                buffer_address_adjust = offset + buffer_address
                response_json.update(
                    {
                        "session_id": self.embeddings_engine.session_id,
                        "buffer_address": buffer_address_adjust,
                    }
                )
                metadata_tasks.append(
                    # GLM Note: Reuse the media affinity key so /send reaches
                    # the encoder replica that retained the /encode result.
                    post_encoder_request(response_json, endpoint_send, "send")
                )
                offset += embedding_size_list_sort[idx]
            await asyncio.gather(*metadata_tasks)


class MMReceiverGrpc(MMReceiverBase):
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
        is_decode_role: bool = False,
    ):
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
            is_decode_role=is_decode_role,
        )

    def build_and_send_encode_request(self, image_urls, rid):
        encode_req = GenerateReqInput(
            image_data=[ImageData(url=url) for url in image_urls],
            rid=rid,
        )
        self.send_encode_request(encode_req)
        return encode_req

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        return self._process_waiting_requests(recv_reqs, WaitingImageRequestGrpc)

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
        upstream_session_id=None,
    ):
        if not mm_data:
            return

        # gRPC currently only supports image; flatten new dict formats to simple lists
        if mm_data and isinstance(mm_data[0], dict):
            non_image = [
                item.get("modality")
                for item in mm_data
                if item.get("modality") != Modality.IMAGE
            ]
            if non_image:
                raise NotImplementedError(
                    f"gRPC encode only supports IMAGE modality, got: {non_image}"
                )
            img_data = [item.get("url") for item in mm_data]
        else:
            img_data = mm_data
        if isinstance(num_items_assigned, dict):
            num_items_assigned = list(num_items_assigned.values())[0]

        encode_requests = []
        if num_items_assigned is None:
            encode_idx = list(range(len(self.encode_urls)))
            random.shuffle(encode_idx)
            num_items_assigned = [
                (idx + len(img_data)) // len(self.encode_urls) for idx in encode_idx
            ]
        num_parts = sum(1 for x in num_items_assigned if x != 0)
        cum_num_items = 0
        cum_idx = 0
        for idx, assigned_num in enumerate(num_items_assigned):
            if assigned_num == 0:
                continue
            start = cum_num_items
            end = cum_num_items + assigned_num
            encode_requests.append(
                {
                    "encoder_idx": idx,
                    "mm_items": img_data[start:end],
                    "num_parts": num_parts,
                    "part_idx": cum_idx,
                    "req_id": req_id,
                    "prefill_host": self.host,
                    "embedding_port": embedding_port,
                }
            )
            cum_idx += 1
            cum_num_items += assigned_num

        grpc_tasks = [
            asyncio.to_thread(
                _grpc_encode_request,
                _grpc_target(self.encode_urls[encode_request["encoder_idx"]]),
                encode_request,
            )
            for encode_request in encode_requests
        ]
        grpc_responses = await asyncio.gather(*grpc_tasks)
        response_json_unsorted = []
        for encode_request, response in zip(encode_requests, grpc_responses):
            if self.encoder_transfer_backend == "zmq_to_scheduler":
                response_json_unsorted.append(None)
                continue
            response_json_unsorted.append(
                {
                    "req_id": encode_request["req_id"],
                    "prefill_host": encode_request["prefill_host"],
                    "embedding_port": encode_request["embedding_port"],
                    "encoder_idx": encode_request["encoder_idx"],
                    "part_idx": encode_request["part_idx"],
                    "embedding_size": response.embedding_size,
                    "embedding_len": response.embedding_len,
                    "embedding_dim": response.embedding_dim,
                }
            )

        if None in response_json_unsorted:
            return

        embedding_size_by_part = [None for _ in range(num_parts)]
        response_json_sorted = [None for _ in range(num_parts)]
        for response_json in response_json_unsorted:
            idx = response_json["part_idx"]
            embedding_size_by_part[idx] = response_json["embedding_size"]
            response_json_sorted[idx] = response_json

        total_embedding_bytes = sum(s for s in embedding_size_by_part if s is not None)
        offset = 0
        buffer_address = await self.allocate_embedding_buffer(
            req_id,
            total_embedding_bytes,
        )
        grpc_metadata_tasks = []
        for response_json in response_json_sorted:
            response_json.update(
                {
                    "session_id": self.embeddings_engine.session_id,
                    "buffer_address": offset + buffer_address,
                }
            )
            grpc_metadata_tasks.append(
                asyncio.to_thread(
                    _grpc_send_request,
                    _grpc_target(self.encode_urls[response_json["encoder_idx"]]),
                    response_json,
                )
            )
            offset += embedding_size_by_part[response_json["part_idx"]]

        if grpc_metadata_tasks:
            await asyncio.gather(*grpc_metadata_tasks)


def _validate_transport_mode(transport_mode: str, encoder_urls):
    if transport_mode == "grpc":
        invalid_prefix = "http://"
        error_msg = (
            "EPD MMReceiver: grpc mode requires grpc:// encoder URLs. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=http for http:// URLs."
        )
    elif transport_mode == "http":
        invalid_prefix = "grpc://"
        error_msg = (
            "EPD MMReceiver: http mode requires http:// encoder URLs. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=grpc for grpc:// URLs."
        )
    else:
        return

    if any(url.startswith(invalid_prefix) for url in encoder_urls):
        raise ValueError(error_msg)


_MM_RECEIVER_BY_MODE = {
    "grpc": MMReceiverGrpc,
    "http": MMReceiverHTTP,
}


def create_mm_receiver(
    server_args: ServerArgs,
    dtype: Optional[torch.dtype] = None,
    hf_config: Optional[PretrainedConfig] = None,
    pp_rank: Optional[int] = None,
    tp_rank: Optional[int] = None,
    tp_group: Optional[GroupCoordinator] = None,
    scheduler: Optional["Scheduler"] = None,
    transport_mode: Optional[str] = None,
    is_decode_role: bool = False,
):
    if transport_mode is None:
        transport_mode = envs.SGLANG_ENCODER_MM_RECEIVER_MODE.get()
        logger.debug(f"MMReceiver transport_mode from env: {transport_mode}")

    _validate_transport_mode(transport_mode, server_args.encoder_urls)
    logger.info(f"EPD MMReceiver: using transport_mode={transport_mode}")

    receiver_cls = _MM_RECEIVER_BY_MODE.get(transport_mode)
    if receiver_cls is None:
        raise ValueError(f"Unsupported transport_mode: {transport_mode}")
    return receiver_cls(
        server_args,
        dtype=dtype,
        hf_config=hf_config,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_group=tp_group,
        scheduler=scheduler,
        is_decode_role=is_decode_role,
    )
