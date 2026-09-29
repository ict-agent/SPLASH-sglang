import asyncio
import concurrent.futures
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
from array import array
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


def _check_recv_embedding_nan(req_id, recv_embedding) -> None:
    """Env-gated (SGLANG_ENCODER_CHECK_NAN) read-only NaN/Inf check on the
    embedding P read from the encoder, before it is scattered into input_embeds.
    Catches whole-segment NaN from an unwritten/stale buffer -- which the size
    checks and the encoder-side check do not (size is right, values are stale).
    ``recv_embedding`` is a {modality: tensor} dict (mooncake) or a tensor."""
    items = (
        recv_embedding.items()
        if isinstance(recv_embedding, dict)
        else [(None, recv_embedding)]
    )
    for modality, emb in items:
        if not (isinstance(emb, torch.Tensor) and emb.numel()):
            continue
        f = emb.detach().float()
        nan, inf = int(torch.isnan(f).sum()), int(torch.isinf(f).sum())
        if nan or inf:
            logger.error(
                "[recv-nan] req_id=%s modality=%s shape=%s nan=%d inf=%d all_nan=%s",
                req_id,
                modality,
                tuple(emb.shape),
                nan,
                inf,
                nan == emb.numel(),
            )


# GLM Note: A 64-bit media digest keeps Session-Id headers compact while
# retaining sufficient collision resistance for encoder load-balancer affinity.
_ENCODER_MEDIA_HASH_HEX_LENGTH = 16
_encoder_affinity_invalid_shards_warned = False


def _rdma_pool_max_bytes() -> int:
    return envs.SGLANG_MC_RDMA_POOL_MAX_MB.get() * 1024 * 1024


def _rdma_pool_acquire_timeout_secs() -> float:
    """Max seconds ``RdmaBufferPool.acquire`` blocks waiting for budget before
    it allocates over the limit anyway. This is the admission control that
    turns SGLANG_MC_RDMA_POOL_MAX_MB into a real cap on peak
    live buffers (previously it only bounded the idle free list, so a burst
    of concurrent large multimodal receives could allocate unbounded and OOM
    the prefill pod). 0 disables the wait (legacy unbounded behavior)."""
    return envs.SGLANG_MC_RDMA_POOL_ACQUIRE_TIMEOUT_SECS.get()


def rdma_pool_enabled() -> bool:
    """Whether to use the Mooncake RDMA registered-buffer pools.

    Controlled by the byte cap that sizes the pool:
      * SGLANG_MC_RDMA_POOL_MAX_MB (default 0)
    > 0 ENABLES the pool: the receiver and the sender both use
    RdmaBufferPool. 0 (the default) DISABLES it -- the receiver and sender
    fall back to the original per-request register + deregister logic.
    """
    return _rdma_pool_max_bytes() > 0


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
    except grpc.RpcError as e:
        # Map RpcError to EncoderError so encoder-side encode failures carry a
        # status code (INTERNAL -> 500 with the encoder's details, anything
        # else -> 503) instead of leaking a raw grpc error to the tokenizer.
        status_code = (
            HTTPStatus.INTERNAL_SERVER_ERROR
            if e.code() == grpc.StatusCode.INTERNAL
            else HTTPStatus.SERVICE_UNAVAILABLE
        )
        raise EncoderError(
            f"Encoder /encode request failed for {encode_request['req_id']}: "
            f"{e.code()} {e.details()}",
            status_code=status_code,
        ) from e
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
    except grpc.RpcError as e:
        # The server signals a reclaimed embedding with NOT_FOUND; map it to the
        # same 410 GONE the HTTP path returns, otherwise surface a generic
        # failure instead of leaking a raw grpc error.
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise EncoderError(
                f"Encoder returned NOT_FOUND on /send for {request_json['req_id']}: "
                f"{e.details()}",
                status_code=HTTPStatus.GONE,
            ) from e
        raise EncoderError(
            f"Encoder /send request failed for {request_json['req_id']}: {e}",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        ) from e
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


# Bound on how long _abort_encode_and_cleanup waits for the encode task to
# drain before force-cancelling it. A part's /send completes (or times out) at
# mooncake's MC_TRANSFER_TIMEOUT (seconds, default 30, min 5); the extra
# margin covers HTTP and ZMQ ack latency. This is only ever reached by a stuck
# /encode, never on the happy path.
#
# NOTE: mooncake reads MC_TRANSFER_TIMEOUT per process, and the timeout that
# actually bounds an in-flight RDMA write lives on the ENCODER (it calls
# transfer_sync). Deployments must set MC_TRANSFER_TIMEOUT to the same value
# on the prefill/decode and encoder processes; otherwise this receiver-side
# bound can expire while E's write is legitimately still in flight and
# reintroduce the deregister-while-writing race. On the gRPC receiver path,
# MC_TRANSFER_TIMEOUT plus this margin must also stay under
# SGLANG_ENCODER_GRPC_TIMEOUT_SECS (default 60s), or the client-side RPC
# deadline fires before the write bound does.
def _encode_drain_timeout_s() -> float:
    try:
        # float() accepts "30" and "30.5" alike; int() would silently reject
        # float strings and fall back, mis-aligning the bound with the
        # encoder's actual write timeout.
        mc_timeout = int(float(os.environ.get("MC_TRANSFER_TIMEOUT", "30")))
    except (TypeError, ValueError):
        logger.warning(
            "invalid MC_TRANSFER_TIMEOUT=%r (not a number); falling back to 30s "
            "for the encode drain bound. The bound can expire while an "
            "encoder's RDMA write is still in flight and reintroduce the "
            "deregister-while-writing race.",
            os.environ.get("MC_TRANSFER_TIMEOUT"),
        )
        mc_timeout = 30
    return max(5, mc_timeout) + 5.0


def _validate_encode_drain_bound() -> None:
    """Validate the drain bound against cross-process constraints we can read.

    The bound that actually ends an in-flight RDMA write lives on the ENCODER
    process (it calls transfer_sync), so this process can only validate what
    it can see: the gRPC receiver's own RPC deadline. Anything that depends on
    the encoder's environment is logged as a deployment requirement.

    This module is imported by every SGLang process (tokenizer_manager,
    scheduler), including non-mooncake / non-EPD deployments, so the unset-env
    note is informational; only a genuinely misconfigured gRPC deadline (a
    real, actionable error) stays at WARNING.
    """
    grpc_timeout = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    if _ENCODE_DRAIN_TIMEOUT_S >= grpc_timeout:
        logger.warning(
            "mm receiver: encode drain bound %.0fs >= "
            "SGLANG_ENCODER_GRPC_TIMEOUT_SECS %.0fs -- the client-side RPC "
            "deadline fires before the drain completes, so the drain is "
            "ineffective on the gRPC path. Lower MC_TRANSFER_TIMEOUT or raise "
            "SGLANG_ENCODER_GRPC_TIMEOUT_SECS.",
            _ENCODE_DRAIN_TIMEOUT_S,
            grpc_timeout,
        )
    mc_timeout = os.environ.get("MC_TRANSFER_TIMEOUT")
    if mc_timeout is None:
        logger.info(
            "mm receiver: MC_TRANSFER_TIMEOUT is unset on this process; the "
            "encode drain bound assumes the default 30s. Encoder processes "
            "in a mooncake EPD deployment MUST set the same value -- the "
            "transfer_sync timeout that actually bounds an in-flight RDMA "
            "write is read from the ENCODER's environment, and a larger value "
            "there makes this bound expire mid-write "
            "(deregister-while-writing race)."
        )


_ENCODE_DRAIN_TIMEOUT_S = _encode_drain_timeout_s()
_validate_encode_drain_bound()


class RdmaBufferPool:
    """Pool of long-lived, RDMA-registered CPU buffers.

    The receiver uses these buffers as RDMA-write targets. The encoder sender
    copies embeddings into the same kind of registered-once buffers and uses
    them as RDMA-write sources, avoiding per-transfer registration churn.

    The byte cap (SGLANG_MC_RDMA_POOL_MAX_MB) acts on two sides.
    release() bounds only the idle reuse cache -- in-flight and quiesced
    buffers stay registered until their transfers are safe to release, so a
    traffic or failure spike does not trigger deregister/registration churn.
    acquire() additionally admission-controls the total live working set
    against the same cap: it first evicts idle buffers of smaller size
    classes to make room, and only when none are left blocks until a
    release/discard frees budget instead of allocating unbounded (which
    would OOM the pod). That wait is bounded by
    SGLANG_MC_RDMA_POOL_ACQUIRE_TIMEOUT_SECS, and a request larger
    than the whole cap is always admitted, so the working set can still
    briefly exceed the cap.
    """

    def __init__(self, engine):
        self._engine = engine
        self._lock = threading.Lock()
        # Signalled whenever budget frees up (release/discard) so acquire()
        # waiters can retry instead of allocating over budget.
        self._cond = threading.Condition(self._lock)
        self._free = {}
        self._floor = 1 * 1024 * 1024
        self._max_total_bytes = _rdma_pool_max_bytes()
        self._total_bytes = 0
        self._total_count = 0
        self._free_bytes = 0
        self._free_count = 0
        self._warned_free_over = False
        self._acquire_timeout = _rdma_pool_acquire_timeout_secs()

    def _size_class(self, nbytes: int) -> int:
        nbytes = max(int(nbytes), self._floor)
        return 1 << (nbytes - 1).bit_length()

    def acquire(self, nbytes: int) -> torch.Tensor:
        class_bytes = self._size_class(nbytes)
        deadline = None
        evicted = []
        reused = None
        with self._lock:
            while True:
                free = self._free.get(class_bytes)
                if free:
                    reused = free.pop()
                    self._free_bytes -= reused.numel()
                    self._free_count -= 1
                    break

                best_key = None
                for size, buffers in self._free.items():
                    if (
                        size > class_bytes
                        and buffers
                        and (best_key is None or size < best_key)
                    ):
                        best_key = size
                if best_key is not None:
                    reused = self._free[best_key].pop()
                    self._free_bytes -= reused.numel()
                    self._free_count -= 1
                    break

                # No reusable buffer: we must allocate a new one. Admission
                # control -- block until releases free budget instead of
                # allocating unbounded (a burst of concurrent large multimodal
                # receives otherwise blows past the cap and OOMs the pod).
                would_exceed = self._total_bytes + class_bytes > self._max_total_bytes
                # Proceed anyway when within budget, when nothing is
                # outstanding to free up (a single request larger than the
                # whole cap must still run; also avoids deadlock), or when the
                # wait is disabled.
                if (
                    not would_exceed
                    or self._total_count == 0
                    or self._acquire_timeout <= 0
                ):
                    break
                # GLM NOTE: any idle buffer left here is smaller than
                # class_bytes (a larger one would have been reused above).
                # Evict idle buffers for budget instead of waiting -- a free
                # cache saturated with small size classes must not starve
                # larger allocations into the timeout path (that wedge held
                # prefill pods in permanent 60s-wait + register churn).
                if self._free_count > 0:
                    while (
                        self._free_count > 0
                        and self._total_bytes + class_bytes > self._max_total_bytes
                    ):
                        smallest = min(
                            size for size, buffers in self._free.items() if buffers
                        )
                        buffer = self._free[smallest].pop()
                        self._free_bytes -= buffer.numel()
                        self._free_count -= 1
                        self._total_bytes -= buffer.numel()
                        self._total_count -= 1
                        evicted.append(buffer)
                    continue
                if deadline is None:
                    deadline = time.monotonic() + self._acquire_timeout
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "mooncake RDMA pool: acquire waited %.0fs for %d bytes "
                        "but pool still at bytes=%d/%d count=%d; allocating "
                        "over budget",
                        self._acquire_timeout,
                        class_bytes,
                        self._total_bytes,
                        self._max_total_bytes,
                        self._total_count,
                    )
                    break
                self._cond.wait(timeout=remaining)

            if reused is None:
                # Reserve the budget under the lock so concurrent acquirers
                # observe it and cannot all race past the check together.
                self._total_bytes += class_bytes
                self._total_count += 1

        # Deregister evicted idle buffers outside the lock (engine calls are
        # slow). Safe: they sat in the free cache, so no in-flight write
        # references their MRs.
        for buffer in evicted:
            self._engine.deregister(buffer.data_ptr())
        if reused is not None:
            return reused

        try:
            buffer = torch.empty(class_bytes, dtype=torch.uint8)
            ret = self._engine.register(buffer.data_ptr(), buffer.nbytes)
            if ret != 0:
                raise RuntimeError(
                    f"mooncake register_memory failed (ret={ret}, bytes={class_bytes})"
                )
        except BaseException:
            # Roll back the reservation so a failed allocation does not leak
            # budget (which would wedge every future acquire).
            with self._lock:
                self._total_bytes -= class_bytes
                self._total_count -= 1
                self._cond.notify_all()
            raise
        return buffer

    def release(self, buffer: torch.Tensor) -> None:
        if buffer is None:
            return
        buffer_bytes = buffer.numel()
        with self._lock:
            over_budget = self._free_bytes + buffer_bytes > self._max_total_bytes
            if not over_budget:
                self._free.setdefault(buffer_bytes, []).append(buffer)
                self._free_bytes += buffer_bytes
                self._free_count += 1
                self._cond.notify_all()
                return
            if not self._warned_free_over:
                self._warned_free_over = True
                logger.warning(
                    "mooncake RDMA free buffer cache at budget "
                    "(cached bytes=%d/%d, count=%d; returned bytes=%d; "
                    "registered bytes=%d count=%d); deregistering returned "
                    "buffer. Increase SGLANG_MC_RDMA_POOL_MAX_MB to retain a "
                    "larger working set.",
                    self._free_bytes,
                    self._max_total_bytes,
                    self._free_count,
                    buffer_bytes,
                    self._total_bytes,
                    self._total_count,
                )
            self._total_bytes -= buffer_bytes
            self._total_count -= 1
            self._cond.notify_all()
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
            self._cond.notify_all()
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
        item_hashes: Optional[List[int]] = None,
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
        self.item_hashes = item_hashes
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


def combine_ordered_item_hashes(item_hashes: List[int]) -> Optional[int]:
    """Combine multiple item hashes without relying on Python's process hash."""
    if not item_hashes:
        return None
    if len(item_hashes) == 1:
        return item_hashes[0]

    hasher = hashlib.sha256(b"sglang-epd-item-hashes-v1\0")
    for item_hash in item_hashes:
        encoded = str(item_hash).encode("ascii")
        hasher.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        hasher.update(encoded)
    return int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)


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
        item_hashes=None,
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
            item_hashes=item_hashes,
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
        self.item_hashes_by_part = [
            item_hashes if i == part_idx else None for i in range(num_parts)
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
            item_hashes=getattr(embedding_data, "item_hashes", None),
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

        # raw_buffer is sliced to the bytes the encoder actually wrote, so the
        # reconstructed offset must consume it exactly. A mismatch means the
        # shapes/dtype disagree with the buffer layout (offset drift -> wrong
        # data); this also catches a wrong dtype, since element_size scales it.
        if byte_offset != raw_buffer.numel():
            raise ValueError(
                "Mooncake embedding reconstruction mismatch: consumed "
                f"{byte_offset} bytes but buffer holds {raw_buffer.numel()} "
                f"(dtype={dtype} shapes={self.embedding_shape_list})"
            )

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

    def get_item_hashes_by_modality(self) -> Dict[Modality, List[int]]:
        """Return ordered hashes for modalities whose non-empty parts are complete."""
        hashes_by_modality = defaultdict(list)
        invalid_modalities = set()

        for part_idx, modality in enumerate(self.modality_list):
            if modality is None:
                continue

            shape = self.embedding_shape_list[part_idx]
            is_empty_part = bool(shape is not None and len(shape) > 0 and shape[0] == 0)
            part_hashes = self.item_hashes_by_part[part_idx]
            if part_hashes is None:
                if not is_empty_part:
                    invalid_modalities.add(modality)
                continue
            if not isinstance(part_hashes, (list, tuple)) or any(
                not isinstance(item_hash, int) or isinstance(item_hash, bool)
                for item_hash in part_hashes
            ):
                invalid_modalities.add(modality)
                continue
            hashes_by_modality[modality].extend(part_hashes)

        return {
            modality: hashes
            for modality, hashes in hashes_by_modality.items()
            if modality not in invalid_modalities and hashes
        }

    def inject_item_hashes(self, processor_output) -> None:
        """Attach encoder hashes to rebuilt items, falling back per modality."""
        hashes_by_modality = self.get_item_hashes_by_modality()
        items_by_modality = defaultdict(list)
        for item in processor_output.mm_items:
            items_by_modality[item.modality].append(item)

        for modality, item_hashes in hashes_by_modality.items():
            items = items_by_modality.get(modality, [])
            if len(items) == len(item_hashes):
                assignments = list(zip(items, item_hashes))
            elif len(items) == 1 and len(item_hashes) > 1:
                assignments = [(items[0], combine_ordered_item_hashes(item_hashes))]
            else:
                logger.warning(
                    "Ignoring encoder item hashes for %s: received %d hashes "
                    "but rebuilt %d items",
                    modality.name,
                    len(item_hashes),
                    len(items),
                )
                continue

            for item, item_hash in assignments:
                item.hash = item_hash

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
        self.item_hashes_by_part[pid] = getattr(embedding_data, "item_hashes", None)
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
def _encoder_affinity_shard_id(req_id: Optional[str]) -> Tuple[int, int]:
    global _encoder_affinity_invalid_shards_warned

    affinity_shards = envs.GLM_ENCODER_AFFINITY_SHARDS.get()
    if affinity_shards < 1:
        if not _encoder_affinity_invalid_shards_warned:
            logger.warning(
                "Invalid GLM_ENCODER_AFFINITY_SHARDS=%s; falling back to 1",
                affinity_shards,
            )
            _encoder_affinity_invalid_shards_warned = True
        affinity_shards = 1
    if affinity_shards == 1:
        return 0, affinity_shards
    if not req_id:
        raise ValueError("req_id is required when encoder affinity sharding is enabled")

    request_hash = hashlib.sha256(req_id.encode("utf-8")).digest()
    shard_id = int.from_bytes(request_hash[:8], byteorder="big") % affinity_shards
    return shard_id, affinity_shards


def create_encoder_session_id(
    upstream_session_id: Optional[str],
    media_identifier,
    req_id: Optional[str] = None,
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
    session_id = (
        f"{upstream_session_id}_{media_hash}" if upstream_session_id else media_hash
    )

    # Keep K=1 byte-for-byte compatible with the original media affinity. For
    # K>1, distribute a hot media key over a bounded number of LB affinity
    # keys while keeping /encode and /send for the same part on one encoder.
    shard_id, affinity_shards = _encoder_affinity_shard_id(req_id)
    if affinity_shards == 1:
        return session_id

    # Re-hash the bounded affinity key so the LB always receives the original
    # fixed-width 16-character hexadecimal format. This remains compatible
    # with LBs that parse Session-Id as a uint64 or validate it as hex.
    sharded_key = f"{session_id}:{shard_id}".encode("utf-8")
    return hashlib.sha256(sharded_key).hexdigest()[:_ENCODER_MEDIA_HASH_HEX_LENGTH]


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
                    upstream_session_id,
                    media_identifier,
                    req_id=encode_request["req_id"],
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
            # TODO(P0-5, follow-up): this PULL socket binds the pod IP and the
            # host:port travels in plaintext inside every /encode payload, so
            # any pod in the cluster can push an arbitrary pickle here (RCE).
            # Plan: generate a random per-port token, hand it out with the
            # /encode request, and have the encoder send it as the first ZMQ
            # frame for comparison BEFORE pickle.loads. Requires an atomic
            # encoder+receiver upgrade (protocol change).
            recv_obj: EmbeddingData = pickle.loads(parts[0])
            # Drop frames belonging to a different request that landed here via
            # OS port reuse (a cancelled/timed-out request freed this random
            # port). Without this, a foreign frame is aggregated by part_idx and
            # its shape metadata is applied to our buffer -> NaN; a foreign error
            # frame would also wrongly FAIL this request. See
            # docs/epd-mm-frame-pollution.
            if extract_original_req_id(recv_obj.req_id) != self.rid:
                logger.warning(
                    "mm: dropping foreign frame on reused port req_id=%s "
                    "(expected %s)",
                    recv_obj.req_id,
                    self.rid,
                )
                continue
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
        self.recv_embedding_data.inject_item_hashes(mm_inputs)
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = array("q", mm_inputs.input_ids)
        self.status = WaitingImageRequestStatus.SUCCESS
        self.recv_socket.close()


class WaitingImageRequestGrpc(WaitingImageRequest):
    def send_encode_request(self):
        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            tasks = []
            # gRPC image-only: flatten modality dict to flat list
            assigned = list(self.num_items_assigned.values())[0]
            logger.info(f"num_items_assigned={assigned}")

            cum_idx = 0
            for idx, assigned_num in enumerate(assigned):
                if assigned_num == 0:
                    continue
                # Match MMReceiverGrpc.encode's part numbering: the encoder's
                # send_with_url looks the receive endpoint up by part req_id.
                part_req_id = create_part_req_id(req_id, cum_idx)
                encoder_url = self.encoder_urls[idx]
                receive_url = f"{host_name}:{embedding_port}"
                target_url = f"{encoder_url}/SchedulerReceiveUrl"
                logger.info(
                    f"Preparing to send to {target_url} with part_req_id={part_req_id}"
                )
                tasks.append(
                    asyncio.to_thread(
                        _grpc_scheduler_receive_url,
                        _grpc_target(encoder_url),
                        part_req_id,
                        receive_url,
                        receive_count,
                    )
                )
                cum_idx += 1

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
        # Detached drain+cleanup tasks spawned on the request-cancellation
        # path (see _schedule_drained_cleanup); kept referenced so they are
        # never garbage-collected mid-flight.
        self._cleanup_tasks = set()
        self.encoder_transfer_backend = server_args.encoder_transfer_backend
        self.encode_urls = server_args.encoder_urls
        self.host = get_local_ip_auto(server_args.host)
        self.is_decode_role = is_decode_role
        self.meta_only = is_decode_role and self.encoder_transfer_backend == "mooncake"
        # Element type used to validate incoming frames (mooncake) and to
        # reinterpret ZMQ buffers. Assigned unconditionally: the mooncake
        # cross-check must not silently degrade to a no-op on other backends.
        self.dtype = dtype
        if self.encoder_transfer_backend == "mooncake":
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
            # Index original req_id -> list of part_req_ids, so cleanup can drop
            # a request's parts without scanning every in-flight buffer.
            self._buffer_index = dict()
            self._use_rdma_pool = rdma_pool_enabled()
            self._rdma_pool = (
                RdmaBufferPool(self.embeddings_engine) if self._use_rdma_pool else None
            )
            logger.info(
                "mm receiver: encode drain bound is %.0fs "
                "(MC_TRANSFER_TIMEOUT=%s on this process; encoder processes "
                "MUST set the same value, and on gRPC receivers "
                "MC_TRANSFER_TIMEOUT + margin must stay under "
                "SGLANG_ENCODER_GRPC_TIMEOUT_SECS or this bound is wrong)",
                _ENCODE_DRAIN_TIMEOUT_S,
                os.environ.get("MC_TRANSFER_TIMEOUT", "30"),
            )
        elif self.encoder_transfer_backend == "zmq_to_scheduler":
            self.pp_rank = pp_rank
            self.tp_rank = tp_rank
            self.tp_size = server_args.tp_size
            self.tp_group = tp_group
            self.receive_count = self.tp_size
            self.status_reduce_groups = (
                [tp_group.cpu_group] if tp_group is not None else []
            )
            self.nnodes = server_args.nnodes
            self.hostname = get_local_ip_auto()
            self.waiting_list: List[WaitingImageRequest] = []
            self.scheduler = scheduler
            self.wait_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
            if scheduler is not None and server_args.enable_dp_attention:
                self.receive_count = scheduler.attn_tp_size * scheduler.attn_cp_size
                self.status_reduce_groups = [
                    scheduler.attn_tp_cpu_group,
                    scheduler.attn_cp_cpu_group,
                ]
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
            # The payloads in mm_data now own the only references the encode
            # path needs (extraction shares the strings, it does not copy).
            # With the mooncake backend there is exactly one extraction per
            # request -- the chat pre-validation pass caches its result in
            # obj._precomputed_mm_inputs and /generate is single-pass -- so
            # the raw fields on the request object can be released here,
            # instead of pinning the media bytes through the whole encoder
            # wait. Other backends may re-extract (zmq_to_tokenizer re-entry)
            # and must keep the originals.
            if mm_data and self.encoder_transfer_backend == "mooncake":
                self._release_raw_media_after_extract(request_obj)
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
            # the encode/E stage). Abort fast, but hand drain+cleanup to a
            # detached background task so in-flight RDMA writes finish before
            # the buffers are deregistered; then re-raise to preserve
            # cancellation semantics.
            self._schedule_drained_cleanup(encode_task, req_id, recv_task)
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

    def _schedule_drained_cleanup(self, encode_task, req_id, recv_task=None):
        """Run drain-then-deregister in a detached background task.

        Used on the request-cancellation path: the parent task is being
        cancelled and cannot wait for in-flight /send to drain, but
        deregistering immediately would tear down a live MR while E is still
        writing (the mooncake local/remote protection error). The detached
        task waits (bounded by the RDMA-write timeout) for the encode task,
        then deregisters, so cancellation latency and MR safety no longer
        trade off. It is referenced from _cleanup_tasks so it is never
        garbage-collected mid-flight.
        """
        if req_id is None and encode_task is None and recv_task is None:
            return

        async def _drain_and_cleanup():
            try:
                await self._abort_encode_and_cleanup(encode_task, req_id, recv_task)
            except BaseException:
                logger.exception(
                    "mm: background drain+cleanup failed for req_id=%s", req_id
                )

        cleanup_task = asyncio.create_task(_drain_and_cleanup())
        self._cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._cleanup_tasks.discard)
        logger.info(
            "mm-cleanup: detached drain+cleanup scheduled for req_id=%s "
            "(caller is being cancelled; in-flight RDMA writes must still "
            "finish before deregister)",
            req_id,
        )

    def _schedule_quiesced_deregister(self, req_id):
        """Defer the quiesce hold + deregister to a detached background task.

        Used when the parent task is cancelled again MID-QUIESCE (or
        mid-drain): deregistering right now would skip the very hold that
        keeps E's residual posted slices away from a torn rkey, re-opening
        the deregister-while-writing race. The detached task sleeps the full
        write-timeout window (conservative -- the parent lost track of how
        much of the hold elapsed) and only then discards the buffers.
        """

        async def _quiesce_then_deregister():
            try:
                await asyncio.sleep(_ENCODE_DRAIN_TIMEOUT_S)
                self._cleanup_mooncake_buffer(req_id)
            except BaseException:
                logger.exception(
                    "mm: background quiesced deregister failed for req_id=%s",
                    req_id,
                )

        cleanup_task = asyncio.create_task(_quiesce_then_deregister())
        cleanup_tasks = getattr(self, "_cleanup_tasks", None)
        if cleanup_tasks is not None:
            cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(cleanup_tasks.discard)
        logger.info(
            "mm-cleanup: detached quiesced deregister scheduled for req_id=%s "
            "(re-cancelled mid-quiesce/mid-drain; holding one full window "
            "before deregister)",
            req_id,
        )

    async def _abort_encode_and_cleanup(self, encode_task, req_id, recv_task=None):
        """Stop the recv task, drain the encode task, then discard the buffers.

        The encode task owns every /send HTTP request, and E completes (or
        times out) the RDMA write into a part's buffer *before* that part's
        /send returns. Cancelling encode_task and deregistering immediately
        would tear down a live MR while E is still writing, which is the
        local/remote protection error seen in production. So wait -- bounded
        by the RDMA-write timeout -- for the encode task to finish, then
        deregister. Draining also guarantees no _encode_then_send registers a
        buffer after cleanup, and every task's exception is retrieved.

        On the request-cancellation path the caller cannot await this (the
        parent is already being cancelled); it goes through
        _schedule_drained_cleanup instead, which runs this detached.
        """
        failed = False
        force_cancelled = False
        try:
            if recv_task is not None and not recv_task.done():
                recv_task.cancel()

            if encode_task is not None and not encode_task.done():
                # Let in-flight /send drain. E's write is bounded by mooncake's
                # MC_TRANSFER_TIMEOUT, so this normally returns quickly.
                cancelled_mid_drain = False
                try:
                    _, pending = await asyncio.wait(
                        {encode_task}, timeout=_ENCODE_DRAIN_TIMEOUT_S
                    )
                except asyncio.CancelledError:
                    # Cancelled mid-drain. We must NOT fall through to an
                    # inline deregister: a part's buffer may already be in
                    # E's hands with its transfer_sync still running. Defer
                    # quiesce+deregister to a detached task (if any buffer
                    # was handed over) and let the cancellation propagate.
                    cancelled_mid_drain = True
                    pending = {encode_task}
                if pending:
                    # A sibling /send did not drain in time (e.g. a hung
                    # /encode); force it down. encode()'s finally still
                    # cancels/drains every _encode_then_send, so none registers
                    # a buffer after cleanup. Cancelling only interrupts the
                    # HTTP call: if a part's /encode completed near the end of
                    # the drain window its buffer is already allocated and its
                    # /send dispatched, and E's transfer_sync keeps running on
                    # an executor thread -- treat like a failure below.
                    encode_task.cancel()
                    force_cancelled = True
                    logger.warning(
                        "mm-cleanup: drain bound %.0fs exceeded; force-cancelling "
                        "stuck encode task for req_id=%s",
                        _ENCODE_DRAIN_TIMEOUT_S,
                        req_id,
                    )
                if cancelled_mid_drain:
                    # We are about to propagate without awaiting these tasks;
                    # attach retrieval callbacks so their eventual exceptions
                    # do not surface as "Task exception was never retrieved".
                    for task in (encode_task, recv_task):
                        if task is not None and not task.done():
                            task.add_done_callback(_log_task_exception)
                    if req_id is not None and self._request_has_live_buffers(req_id):
                        self._schedule_quiesced_deregister(req_id)
                        req_id = None  # inline cleanup below must not run
                    raise

            for task in (encode_task, recv_task):
                if task is None:
                    continue
                try:
                    await task
                except asyncio.CancelledError:
                    # Self-inflicted (our cancel above, or the parent's): the
                    # task stopped before reporting anything; quiesce need is
                    # decided below from the request's own buffers.
                    pass
                except BaseException:
                    # encode task failed on its own (non-200 /encode or /send),
                    # or recv task surfaced an encoder error frame (E-side RDMA
                    # write failure arrives as a ZMQ error frame while every
                    # /send still returned 200 -- encode task looks
                    # "successful"). Either way E may hold this request's
                    # buffers/rkeys with residual posted slices in flight.
                    failed = True
        finally:
            # Always deregister, even if we were cancelled mid-drain (the drain
            # window is up to _ENCODE_DRAIN_TIMEOUT_S); otherwise the part
            # buffers and their MRs leak permanently.
            if req_id is not None:
                # Quiesce whenever any buffer was handed to E -- i.e. whenever
                # a /send was dispatched, so E holds this request's rkeys.
                # We deliberately do NOT require a reported failure: on the
                # timeout and client-disconnect paths the recv task (the only
                # consumer of E's ZMQ error frame) is cancelled BEFORE the
                # drain, and the /send failure signal can equally be lost, so
                # "no exception seen" does not mean "no write in flight".
                # Once we are in this cleanup the request is dead either way;
                # the only cost of the hold is a bounded async wait. Pure
                # /encode-dispatch failures allocate no buffer and skip the
                # hold, preserving fail-fast.
                quiesce = self._request_has_live_buffers(req_id)
                if quiesce:
                    # transferSync returning -1 (or an error frame) only means
                    # the waiter gave up; posted slices can still be in flight
                    # (observed landing ~5s later). Hold the MRs for one more
                    # write-timeout window before deregistering so those
                    # residual slices hit a live rkey instead of a torn one.
                    n_buffers = len(getattr(self, "_buffer_index", {}).get(req_id, ()))
                    if failed or force_cancelled:
                        logger.info(
                            "mm-cleanup: quiesce hold %.0fs before deregister "
                            "for req_id=%s (%d part buffer(s), failure reported)",
                            _ENCODE_DRAIN_TIMEOUT_S,
                            req_id,
                            n_buffers,
                        )
                    else:
                        logger.info(
                            "mm-cleanup: quiesce hold %.0fs before deregister "
                            "for req_id=%s (%d part buffer(s), no failure signal "
                            "-- timeout/disconnect path)",
                            _ENCODE_DRAIN_TIMEOUT_S,
                            req_id,
                            n_buffers,
                        )
                    try:
                        await asyncio.sleep(_ENCODE_DRAIN_TIMEOUT_S)
                    except asyncio.CancelledError:
                        # Cancelled again mid-quiesce: deregistering now would
                        # skip the hold entirely and re-open the race. Defer
                        # the remaining window + deregister to a detached task
                        # and let the cancellation propagate.
                        for task in (encode_task, recv_task):
                            if task is not None and not task.done():
                                task.add_done_callback(_log_task_exception)
                        self._schedule_quiesced_deregister(req_id)
                        req_id = None  # suppress the inline cleanup below
                        raise
                self._cleanup_mooncake_buffer(req_id)

    def _request_has_live_buffers(self, req_id) -> bool:
        """Whether any per-part buffer for this request is still held.

        This is the sole quiesce criterion in _abort_encode_and_cleanup:
        holding a buffer means a /send was dispatched, so E holds this
        request's rkeys and residual posted slices may be in flight
        regardless of whether a failure signal reached us.

        A failure before any /send (e.g. /encode dispatch failure, bad image,
        encoder unreachable) allocates no buffer, so there is nothing on E's
        side still writing and the quiesce hold would only delay the client
        error response for no safety benefit.
        """
        if self.encoder_transfer_backend != "mooncake":
            return False
        index = getattr(self, "_buffer_index", None)
        if index is None:
            return False
        return bool(index.get(req_id))

    def _cleanup_mooncake_buffer(self, req_id):
        if self.encoder_transfer_backend != "mooncake":
            return
        if not hasattr(self, "embeddings_buffer"):
            return
        # Buffers are keyed by part req_id ({req_id}_local_part_{idx}); drop
        # every part belonging to this request via the req_id index. The index
        # only exists once the first buffer was stored (and on receivers built
        # via __new__ in tests), so tolerate its absence.
        part_req_ids = getattr(self, "_buffer_index", {}).pop(req_id, [])
        released = 0
        for part_req_id in part_req_ids:
            entry = self.embeddings_buffer.pop(part_req_id, None)
            if entry is None:
                continue
            embeddings, _expected_bytes = entry
            try:
                if self._use_rdma_pool:
                    self._rdma_pool.discard(embeddings)
                else:
                    self.embeddings_engine.deregister(embeddings.data_ptr())
                released += 1
            except Exception:
                logger.exception(
                    "mooncake: failed to discard/deregister buffer for "
                    "part_req_id=%s",
                    part_req_id,
                )
        if released:
            logger.info(
                "mm-cleanup: released %d part buffer(s) for req_id=%s " "(pool=%s)",
                released,
                req_id,
                self._use_rdma_pool,
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
                # TODO(P0-5, follow-up): see the sync-path note -- pickle.loads
                # on a pod-IP-bound PULL socket is an in-cluster RCE surface;
                # gate it on a per-port token sent with the /encode payload.
                recv_obj: EmbeddingData = pickle.loads(parts[0])
                # Frames reach this socket purely by which (per-request, randomly
                # bound) port they land on. A cancelled/timed-out request frees
                # its port, which the OS can recycle to a later request; a late or
                # in-flight frame from the old request then arrives here. Validate
                # the embedded req_id and drop foreign frames -- otherwise they are
                # aggregated by part_idx and this request's RDMA buffer is later
                # read with the wrong request's shape metadata -> silent NaN (or an
                # add()/bounds assert).
                part_req_id = recv_obj.req_id
                original_req_id = extract_original_req_id(part_req_id)
                if original_req_id != req_id:
                    logger.warning(
                        "mm: dropping foreign frame on reused port req_id=%s "
                        "(expected %s, part_req_id=%s)",
                        original_req_id,
                        req_id,
                        part_req_id,
                    )
                    continue
                if getattr(recv_obj, "error_msg", None) is not None:
                    error_code = getattr(recv_obj, "error_code", None)
                    logger.warning(
                        f"Encoder error for req_id={req_id}: {recv_obj.error_msg} "
                        f"error_code={error_code}"
                    )
                    # Don't deregister here: recv_mm_data routes this through
                    # _abort_encode_and_cleanup, which first drains the encode
                    # task so no RDMA write is still in flight when the buffers
                    # are deregistered.
                    # Propagate the encoder's real error code (e.g. 500 for an
                    # RDMA write failure) instead of collapsing into a generic
                    # recv-timeout 504.
                    raise EncoderError(
                        f"Encoder error: {recv_obj.error_msg}",
                        status_code=int(error_code) if error_code else 500,
                    )
                # Cross-check the element type: bf16 vs fp16 both have 2-byte
                # items, so the byte-size guard alone would let a mismatched
                # buffer be reinterpreted silently into corrupted output.
                # Empty video shards are exempt: their placeholder embedding
                # carries zero bytes and is never reinterpreted (its dtype is
                # best-effort metadata on the encoder, not a buffer promise).
                frame_shape = getattr(recv_obj, "shape", None)
                frame_is_empty = not frame_shape or frame_shape[0] == 0
                frame_dtype = getattr(recv_obj, "dtype", None)
                expected_dtype = getattr(self, "dtype", None)
                if (
                    not frame_is_empty
                    and frame_dtype is not None
                    and expected_dtype is not None
                    and frame_dtype != expected_dtype
                ):
                    raise EncoderError(
                        f"Encoder dtype mismatch for req_id={req_id}: encoder "
                        f"sent {frame_dtype}, model expects {expected_dtype}",
                        status_code=500,
                    )
                logger.debug("recv_obj=%s", recv_obj)
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
                # Each part owns its own RDMA buffer (keyed by part req_id),
                # allocated and /sent independently during the pipelined
                # encode. Pop and decode each part's buffer separately.
                for part_idx in range(recv_embedding_data.num_parts):
                    shape = recv_embedding_data.embedding_shape_list[part_idx]
                    if shape is None:
                        # Defensive: nothing to decode for this part, but still
                        # release its buffer (if one was allocated) instead of
                        # leaking it past this request.
                        entry = self._pop_embedding_buffer(
                            create_part_req_id(req_id, part_idx)
                        )
                        if entry is not None:
                            orphan, _ = entry
                            if self._use_rdma_pool:
                                self._rdma_pool.release(orphan)
                            else:
                                self.embeddings_engine.deregister(orphan.data_ptr())
                        continue
                    part_req_id = create_part_req_id(req_id, part_idx)
                    entry = self._pop_embedding_buffer(part_req_id)
                    if entry is None:
                        logger.error(
                            "mooncake: embeddings_buffer missing part_req_id=%s",
                            part_req_id,
                        )
                        raise EncoderError(
                            f"mooncake: embeddings_buffer missing part_req_id={part_req_id}"
                        )
                    raw_buffer, expected_bytes = entry
                    try:
                        # Guard against an RDMA-layout mismatch (embedding_size
                        # != actual write): the byte length must exactly match
                        # the expected shape before reinterpreting the buffer.
                        expected_numel = 1
                        for dim in shape:
                            expected_numel *= dim
                        if expected_bytes != expected_numel * self.dtype.itemsize:
                            raise EncoderError(
                                f"mooncake: buffer size mismatch for "
                                f"part_req_id={part_req_id}: expected "
                                f"{expected_numel * self.dtype.itemsize} bytes "
                                f"for shape {shape}, got {expected_bytes}"
                            )
                        read_buffer = raw_buffer[:expected_bytes]
                        part_embedding = (
                            read_buffer.view(self.dtype).reshape(*shape).clone()
                        )
                        recv_embedding_data.embedding_list[part_idx] = part_embedding
                    finally:
                        if self._use_rdma_pool:
                            self._rdma_pool.release(raw_buffer)
                        else:
                            self.embeddings_engine.deregister(raw_buffer.data_ptr())
                recv_embedding = recv_embedding_data.get_embedding(is_concat=True)
            else:
                recv_embedding = recv_embedding_data.get_embedding(is_concat=True)

            if envs.SGLANG_ENCODER_CHECK_NAN.get():
                _check_recv_embedding_nan(req_id, recv_embedding)

            mm_inputs = mm_processor.get_mm_data(
                prompt,
                recv_embedding,
                **recv_embedding_data.get_mm_extra_meta(),
            )
            recv_embedding_data.inject_item_hashes(mm_inputs)
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
                    receive_count=self.receive_count,
                )
                # TODO(P0-4, follow-up): send_encode_request() runs inline in
                # the scheduler's synchronous main loop: it does
                # asyncio.run(...) + an aiohttp call with total=1800s, so a
                # black-holed encoder connection can freeze the whole loop
                # well past the 300s watchdog and take the replica down, and
                # any exception here (fd exhaustion, bind failure) kills the
                # scheduler process. Move the registration out of the loop
                # and wrap it in try/except that fails only the request.
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

        for group in self.status_reduce_groups:
            torch.distributed.all_reduce(
                local_status,
                op=torch.distributed.ReduceOp.MIN,
                group=group,
            )

        new_waiting = []
        abort_reqs = []
        for i, waiting_req in enumerate(self.waiting_list):
            status_value = local_status[i].item()
            if status_value == WaitingImageRequestStatus.SUCCESS:
                waiting_req.recv_req.need_wait_for_mm_inputs = False
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

    def _store_embedding_buffer(self, part_req_id, embeddings, expected_bytes):
        self.embeddings_buffer[part_req_id] = (embeddings, expected_bytes)
        original_req_id = extract_original_req_id(part_req_id)
        # The index is normally created in __init__ (mooncake branch); use
        # setdefault-style access so test doubles built via __new__ still work.
        if not hasattr(self, "_buffer_index"):
            self._buffer_index = {}
        self._buffer_index.setdefault(original_req_id, []).append(part_req_id)

    def _pop_embedding_buffer(self, part_req_id):
        entry = self.embeddings_buffer.pop(part_req_id, None)
        if entry is not None:
            original_req_id = extract_original_req_id(part_req_id)
            parts = self._buffer_index.get(original_req_id)
            if parts is not None:
                if part_req_id in parts:
                    parts.remove(part_req_id)
                if not parts:
                    self._buffer_index.pop(original_req_id, None)
        return entry

    def _encoder_send_slot(self, encoder_idx: int) -> "asyncio.Semaphore":
        """Per-encoder in-flight /send limiter (see SGLANG_ENCODER_MAX_INFLIGHT_SENDS).

        Lazily created so receivers built via __new__ in tests keep working.
        """
        slots = getattr(self, "_encoder_send_slots", None)
        if slots is None:
            slots = {}
            self._encoder_send_slots = slots
        semaphore = slots.get(encoder_idx)
        if semaphore is None:
            semaphore = asyncio.Semaphore(
                max(1, envs.SGLANG_ENCODER_MAX_INFLIGHT_SENDS.get())
            )
            slots[encoder_idx] = semaphore
        return semaphore

    def _pool_acquire_executor(self) -> "concurrent.futures.ThreadPoolExecutor":
        """Worker threads for the (potentially blocking) pool acquire.

        Lazily created so receivers built via __new__ in tests keep working.
        """
        executor = getattr(self, "_pool_acquire_workers", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=16, thread_name_prefix="mm-pool-acquire"
            )
            self._pool_acquire_workers = executor
        return executor

    async def _acquire_pool_buffer(self, total_bytes) -> torch.Tensor:
        # GLM NOTE: pool.acquire blocks for budget (admission control). This
        # coroutine runs on the http worker's event loop -- calling it inline
        # froze the whole worker (including /health) for the full wait, so
        # run it in a worker thread instead.
        future = self._pool_acquire_executor().submit(
            self._rdma_pool.acquire, total_bytes
        )
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # The thread cannot be interrupted; if it still completes the
            # acquire, hand the buffer straight back so an abandoned wait
            # does not leak pool budget. release() (not discard) is right:
            # the address was never exposed to the encoder, so no write can
            # be in flight against it.
            def _reclaim(f):
                if f.cancelled() or f.exception() is not None:
                    return
                self._rdma_pool.release(f.result())

            future.add_done_callback(_reclaim)
            raise

    async def allocate_embedding_buffer(self, req_id, total_bytes):
        # NOTE: per-part buffers cost one mooncake register/deregister per part
        # (the pre-pipelining contiguous layout did a single registration per
        # request). This is the price of pipelining /send per part; enable the
        # RDMA pool (SGLANG_MC_RDMA_POOL_MAX_MB)
        # for multi-image workloads so buffers are reused via size classes
        # instead of re-registered per request.
        #
        # Empty video shards carry 0 bytes; mooncake rejects zero-length regions
        # so allocate at least 1 byte, and the recorded write-length stays 0 so
        # the read slices to empty. The pool path floors to its size class.
        if self._use_rdma_pool:
            embeddings = await self._acquire_pool_buffer(total_bytes)
        else:
            embeddings = torch.empty(max(1, total_bytes), dtype=torch.uint8)
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
        # INVARIANT: a torch allocation never returns a NULL data_ptr, and the
        # gRPC Encode path derives `meta_only` from `buffer_address == 0`
        # (encode_grpc_server.py). Check BEFORE storing so a hypothetical
        # violation does not leak an entry into embeddings_buffer.
        assert embeddings.data_ptr() != 0, (
            "allocate_embedding_buffer must return a non-zero data_ptr; "
            "gRPC meta_only detection relies on it"
        )
        # Keep the FULL tensor (pool release/discard key on its numel()) plus the
        # expected write length, so the read can slice to the written region.
        self._store_embedding_buffer(req_id, embeddings, total_bytes)
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

    _RAW_MM_RELEASED_MARKER = "<released-to-encoder>"

    def _release_raw_media_after_extract(self, request_obj) -> None:
        """Drop raw media payload references once _extract_url_data owns them.

        _extract_url_data builds payloads that share (not copy) the base64/URL
        string references, so after it returns the request object's raw fields
        have no remaining reader on the encode path. Replacing them here frees
        the media bytes for the whole encoder wait (up to
        SGLANG_ENCODER_RECV_TIMEOUT) instead of holding them until scheduler
        dispatch. A truthy, length-preserving marker per item keeps
        contains_mm_input(), mm-limit validation and item counting truthful;
        the tokenizer manager's _release_raw_multimodal_payload nulls the
        fields for real after dispatch.
        """
        for attr in ("image_data", "video_data", "audio_data"):
            data = getattr(request_obj, attr, None)
            if data is None:
                continue
            if isinstance(data, list):
                setattr(request_obj, attr, [self._RAW_MM_RELEASED_MARKER] * len(data))
            else:
                setattr(request_obj, attr, self._RAW_MM_RELEASED_MARKER)

    def _extract_url_data(self, request_obj) -> List[Dict]:
        def is_video_frame_sequence(items):
            return (
                isinstance(items, (list, tuple))
                and bool(items)
                and all(
                    isinstance(item, dict) and "url" in item and "timestamp" in item
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

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=1800
            )  # Add timeout for request reliability
        ) as session:

            async def post_encoder_request(payload, endpoint, phase):
                encoder_idx = payload["encoder_idx"]
                encoder_session_id = encoder_session_ids.get(payload["part_idx"])
                affinity_shard = None
                affinity_shards = envs.GLM_ENCODER_AFFINITY_SHARDS.get()
                if encoder_session_id:
                    affinity_shard, affinity_shards = _encoder_affinity_shard_id(
                        payload["req_id"]
                    )
                url = f"{self.encode_urls[encoder_idx]}/{endpoint}"
                headers = _encoder_request_headers(
                    payload["req_id"], encoder_session_id
                )
                # GLM Note: Log encoder routing metadata so operators can verify
                # per-media parallel dispatch and Session-Id affinity in production.
                logger.info(
                    "Sending request to encoder: phase=%s url=%s req_id=%s "
                    "part_idx=%s num_parts=%s modality=%s encoder_idx=%s "
                    "session_id=%s affinity_shard=%s affinity_shards=%s",
                    phase,
                    url,
                    payload["req_id"],
                    payload["part_idx"],
                    payload["num_parts"],
                    payload["modality"],
                    encoder_idx,
                    encoder_session_id,
                    affinity_shard,
                    affinity_shards,
                )
                return await session.post(url, json=payload, headers=headers)

            # Send encode requests. Session-Id provides media affinity while
            # Request-Id retains the unique request/part identity.
            # Each part is encoded and immediately sent (pipelined) instead of
            # gathering all /encode responses first: an early part's embedding
            # must not sit in the encoder waiting for every other part to
            # finish encoding, or it can exceed the orphan-sweeper TTL and be
            # reclaimed before its /send arrives.
            async def _encode_then_send(encode_request):
                try:
                    response = await post_encoder_request(
                        encode_request, endpoint_encode, "encode"
                    )
                except Exception as e:
                    raise EncoderError(
                        f"Encoder request failed for {req_id}: {e}",
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    ) from e
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
                response_json = await response.json()
                if self.meta_only:
                    return
                # zmq backend: return is None
                if response_json is None:
                    return

                # mooncake backend: allocate a per-part RDMA buffer and send
                # immediately, keyed by the part req_id so recv can pop each
                # part independently.
                part_idx = response_json["part_idx"]
                embedding_size = response_json["embedding_size"]
                part_req_id = create_part_req_id(req_id, part_idx)
                buffer_address = await self.allocate_embedding_buffer(
                    part_req_id, embedding_size
                )
                response_json.update(
                    {
                        "session_id": self.embeddings_engine.session_id,
                        "buffer_address": buffer_address,
                    }
                )
                # Bound concurrent in-flight /send per encoder: the encoder's
                # transfer executor has 10 workers, and a queued transfer_sync
                # can wait a full MC_TRANSFER_TIMEOUT behind a stalled batch --
                # longer than the quiesce window budgets, reviving the
                # deregister-while-writing race under saturation.
                semaphore = self._encoder_send_slot(encode_request["encoder_idx"])
                try:
                    async with semaphore:
                        send_response = await post_encoder_request(
                            response_json, endpoint_send, "send"
                        )
                except Exception as e:
                    raise EncoderError(
                        f"Encoder /send request failed for {req_id}: {e}",
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    ) from e
                if send_response.status != 200:
                    try:
                        err_data = await send_response.json()
                        msg = err_data.get("message", "Unknown encoder error")
                    except Exception:
                        msg = await send_response.text()
                    raise EncoderError(
                        f"Encoder returned error {send_response.status} on /send: {msg}",
                        status_code=send_response.status,
                    )

            tasks = [
                asyncio.ensure_future(_encode_then_send(er)) for er in encode_requests
            ]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                first_exc = None
                for t in done:
                    t_exc = t.exception()
                    if t_exc is not None:
                        first_exc = t_exc
                        break
                if first_exc is not None:
                    # Let siblings finish their /send (draining their in-flight
                    # RDMA writes) before surfacing the failure, so the caller's
                    # cleanup deregisters buffers only after every write has
                    # drained. Bounded like _abort_encode_and_cleanup: anything
                    # still pending past the RDMA-write bound is a stuck /encode
                    # (which owns no buffer), so force it down.
                    pending = [t for t in tasks if not t.done()]
                    if pending:
                        _, still_pending = await asyncio.wait(
                            pending, timeout=_ENCODE_DRAIN_TIMEOUT_S
                        )
                        for t in still_pending:
                            t.cancel()
                        if still_pending:
                            await asyncio.gather(*still_pending, return_exceptions=True)
                    raise first_exc
            finally:
                # Parent cancellation cannot wait for in-flight /send to drain
                # (we are already being cancelled), so cancel and drain every
                # task so none registers a buffer after _cleanup_mooncake_buffer
                # has run.
                for p in tasks:
                    p.cancel()
                for p in tasks:
                    try:
                        await p
                    except BaseException:
                        pass


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
                    "req_id": create_part_req_id(req_id, cum_idx),
                    "prefill_host": self.host,
                    "embedding_port": embedding_port,
                }
            )
            cum_idx += 1
            cum_num_items += assigned_num

        # Pipeline each part independently: encode, then (mooncake) allocate a
        # per-part RDMA buffer and /send as soon as that part's Encode returns,
        # so no embedding waits on its siblings and gets reclaimed by the
        # orphan sweeper before its /send arrives.
        async def _encode_then_send(encode_request):
            target = _grpc_target(self.encode_urls[encode_request["encoder_idx"]])
            response = await asyncio.to_thread(
                _grpc_encode_request, target, encode_request
            )
            if self.encoder_transfer_backend != "mooncake":
                # zmq backends: the encoder's Encode RPC already sent the data
                # over ZMQ; nothing further to do on the receiver side.
                return
            part_req_id = encode_request["req_id"]
            if self.meta_only:
                # Decode-role receiver: no RDMA buffer to allocate. Still issue
                # a bufferless /send so the encoder pushes the metadata frame
                # over ZMQ (the gRPC Encode handler has no role field, so the
                # encoder still runs the ViT here -- unlike the HTTP decode
                # path, which routes to encode_metadata server-side).
                await asyncio.to_thread(
                    _grpc_send_request,
                    target,
                    {
                        "req_id": part_req_id,
                        "prefill_host": encode_request["prefill_host"],
                        "embedding_port": encode_request["embedding_port"],
                        "session_id": "",
                        "buffer_address": 0,
                    },
                )
                return
            buffer_address = await self.allocate_embedding_buffer(
                part_req_id, response.embedding_size
            )
            await asyncio.to_thread(
                _grpc_send_request,
                target,
                {
                    "req_id": part_req_id,
                    "prefill_host": encode_request["prefill_host"],
                    "embedding_port": encode_request["embedding_port"],
                    "session_id": self.embeddings_engine.session_id,
                    "buffer_address": buffer_address,
                },
            )

        tasks = [asyncio.ensure_future(_encode_then_send(er)) for er in encode_requests]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            first_exc = None
            for t in done:
                t_exc = t.exception()
                if t_exc is not None:
                    first_exc = t_exc
                    break
            if first_exc is not None:
                # Let siblings finish their /send (draining their in-flight RDMA
                # writes) before surfacing the failure, so the caller's cleanup
                # deregisters buffers only after every write has drained.
                # Bounded like _abort_encode_and_cleanup: anything still
                # pending past the RDMA-write bound is a stuck /encode (which
                # owns no buffer), so force it down.
                pending = [t for t in tasks if not t.done()]
                if pending:
                    _, still_pending = await asyncio.wait(
                        pending, timeout=_ENCODE_DRAIN_TIMEOUT_S
                    )
                    for t in still_pending:
                        t.cancel()
                    if still_pending:
                        await asyncio.gather(*still_pending, return_exceptions=True)
                raise first_exc
        finally:
            for p in tasks:
                p.cancel()
            for p in tasks:
                try:
                    await p
                except BaseException:
                    pass


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
