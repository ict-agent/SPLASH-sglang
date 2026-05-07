import logging
import os
import resource
import threading
import time
from collections import OrderedDict, deque
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.utils import get_int_env_var
from sglang.srt.utils.network import (
    config_socket,
    NetworkAddress,
)

logger = logging.getLogger(__name__)


ZMQ_SOCKET_SEND_TIMEOUT_MS = get_int_env_var(
    "SGLANG_DISAGGREGATION_ZMQ_SOCKET_SEND_TIMEOUT_MS", 40
)
ZMQ_SOCKET_RECEIVE_TIMEOUT_MS = get_int_env_var(
    "SGLANG_DISAGGREGATION_ZMQ_SOCKET_RECEIVE_TIMEOUT_MS", 50
)
ZMQ_SOCKET_DISABLE_ACK_CHECK: bool = (
    get_int_env_var("SGLANG_DISAGGREGATION_ZMQ_SOCKET_DISABLE_ACK_CHECK", 0) == 1
)
ZMQ_SOCKET_MAX_RETRIES = get_int_env_var(
    "SGLANG_DISAGGREGATION_ZMQ_SOCKET_MAX_RETRIES", 5
)
ZMQ_SOCKET_RETRY_DELAY_MS = get_int_env_var(
    "SGLANG_DISAGGREGATION_ZMQ_SOCKET_RETRY_DELAY_MS", 50
)
ZMQ_SOCKET_CACHE_SIZE = get_int_env_var(
    "SGLANG_DISAGGREGATION_ZMQ_SOCKET_CACHE_SIZE", 1024
)

RLIMIT_SOFT, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
if ZMQ_SOCKET_CACHE_SIZE > RLIMIT_SOFT:
    logger.warning(
        f"adjust ZMQ_SOCKET_CACHE_SIZE to {RLIMIT_SOFT}, since it should be <= rlimit_soft"
    )
    ZMQ_SOCKET_CACHE_SIZE = RLIMIT_SOFT

_cached_zmq_sockets = (
    OrderedDict()
)  # type: OrderedDict[Tuple[zmq.SocketType, str, bool, str], CachedZMQSocket]
_cached_zmq_sockets_lock = threading.RLock()  # reentrant lock

_shared_zmq_ctx: Optional[zmq.Context] = None
_shared_zmq_ctx_lock = threading.Lock()


def _get_shared_zmq_context() -> zmq.Context:
    # Reuse one process-level context for caller paths that do not pass an explicit context.
    global _shared_zmq_ctx
    if _shared_zmq_ctx is None:
        with _shared_zmq_ctx_lock:
            if _shared_zmq_ctx is None:
                _shared_zmq_ctx = zmq.Context()
                # GLM Fix: avoid ZMQ Too Many Open Files
                _shared_zmq_ctx.set(zmq.MAX_SOCKETS, 65535)
    return _shared_zmq_ctx


def get_cached_zmq_socket(
    ctx: Optional[zmq.Context],
    socket_type: zmq.SocketType,
    endpoint: str,
    is_ipv6: bool = False,
    desc: str = "",
    recreate_always: bool = False,
):
    with _cached_zmq_sockets_lock:
        if ctx is None:
            ctx = _get_shared_zmq_context()

        key = (socket_type, endpoint)

        if key in _cached_zmq_sockets:
            if recreate_always:
                # Remove the old socket to recreate a new one
                old_socket = _cached_zmq_sockets.pop(key)
                old_socket.close()
            else:
                _cached_zmq_sockets.move_to_end(
                    key, last=True
                )  # mark as most recently used
                return _cached_zmq_sockets[key]

        if len(_cached_zmq_sockets) >= ZMQ_SOCKET_CACHE_SIZE:
            # Remove the least recently used socket
            _, old_socket = _cached_zmq_sockets.popitem(last=False)
            old_socket.close()

        new_socket = CachedZMQSocket(ctx, socket_type, endpoint, is_ipv6, desc)
        _cached_zmq_sockets[key] = new_socket  # Add new socket to cache at last
        return new_socket


class CachedZMQSocket:
    def __init__(
        self,
        ctx: zmq.Context,
        socket_type: zmq.SocketType,
        endpoint: str,
        is_ipv6: bool = False,
        description: str = "",
    ):
        try:
            self.endpoint = endpoint
            self.description = description
            self.zmq_max_sockets = 0
            self.sock = None
            self.lock = None

            if ctx is None:
                ctx = zmq.Context()

            try:
                self.zmq_max_sockets = int(ctx.get(zmq.MAX_SOCKETS))
            except KeyError:
                self.zmq_max_sockets = 0

            if self.zmq_max_sockets < ZMQ_SOCKET_CACHE_SIZE:
                logger.debug(
                    f"enlarged zmq.MAX_SOCKETS to {ZMQ_SOCKET_CACHE_SIZE} since its original value {self.zmq_max_sockets} is smaller than cache size, "
                    f"rlimit_soft={RLIMIT_SOFT}, "
                    f"endpoint={self.endpoint}, "
                    f"description={self.description}"
                )
                self.zmq_max_sockets = ZMQ_SOCKET_CACHE_SIZE
                ctx.set(zmq.MAX_SOCKETS, ZMQ_SOCKET_CACHE_SIZE)

            self.lock = threading.Lock()

            self.sock = ctx.socket(socket_type)
            config_socket(self.sock, socket_type)

            if is_ipv6:
                self.sock.setsockopt(zmq.IPV6, 1)
            self.sock.connect(endpoint)

        except Exception as e:
            logger.error(
                f"failed to create zmq socket: {e}, "
                f"endpoint={self.endpoint}, "
                f"description={self.description}, "
                f"zmq_max_sockets={self.zmq_max_sockets}, "
                f"limit_soft={RLIMIT_SOFT}, "
                f"cur_fd_count={self._cur_fd_count}"
            )
            raise Exception(
                f"failed to create zmq socket: {e}, endpoint={self.endpoint}"
            )

    def close(self):
        try:
            if not hasattr(self, "sock") or self.sock is None:
                raise Exception("already closed or partially created")

            self.sock.setsockopt(zmq.LINGER, 0)  # without blocking
            self.sock.close()
            self.sock = None
            self.lock = None

            logger.info(
                f"successfully close zmq socket since lru_cache full, "
                f"endpoint={self.endpoint}, "
                f"description={self.description}, "
                f"lru_cache_size={ZMQ_SOCKET_CACHE_SIZE}, "
                f"zmq_max_sockets={self.zmq_max_sockets}, "
                f"limit_soft={RLIMIT_SOFT}, "
                f"cur_fd_count={self._cur_fd_count}"
            )

        except Exception as e:
            logger.error(
                f"failed to close zmq socket: {e}, "
                f"endpoint={self.endpoint}, "
                f"description={self.description}, "
                f"lru_cache_size={ZMQ_SOCKET_CACHE_SIZE}, "
                f"zmq_max_sockets={self.zmq_max_sockets}, "
                f"limit_soft={RLIMIT_SOFT}, "
                f"cur_fd_count={self._cur_fd_count}"
            )

    @property
    def _cur_fd_count(self):
        """Get the current process open FD count for diagnostics."""
        try:
            return len(os.listdir(f"/proc/{os.getpid()}/fd"))
        except Exception:
            return -1


def send_multipart_by_req_socket(
    ctx: Optional[zmq.Context],
    remote_ip: str,
    remote_port: int,
    multipart_data: List[bytes] = [],
    non_blocking: bool = False,
    max_retries: int = ZMQ_SOCKET_MAX_RETRIES,
    retry_delay_ms: int = ZMQ_SOCKET_RETRY_DELAY_MS,
    desc: str = "",
    bootstrap_room: Optional[int] = None,
):
    start_time = time.perf_counter()

    na = NetworkAddress(remote_ip, remote_port)
    is_ipv6 = na.is_ipv6

    def _config_req_socket(socket: zmq.Socket):
        socket.setsockopt(zmq.SNDTIMEO, ZMQ_SOCKET_SEND_TIMEOUT_MS)  # ms
        socket.setsockopt(zmq.RCVTIMEO, ZMQ_SOCKET_RECEIVE_TIMEOUT_MS)  # ms
        # FIX zmq.error.ZMQError: Operation cannot be accomplished in current state
        # Allow sending next request even if timeout happens on the previous request
        socket.setsockopt(zmq.REQ_RELAXED, 1)
        socket.setsockopt(zmq.REQ_CORRELATE, 1)

    send_success = False
    attempt = -1
    for attempt in range(max_retries):
        try:
            zmq_socket = get_cached_zmq_socket(
                ctx, zmq.REQ, na.to_tcp(), is_ipv6, desc, recreate_always=(attempt > 0)
            )
            with zmq_socket.lock:
                _config_req_socket(zmq_socket.sock)

                if non_blocking:
                    zmq_socket.sock.send_multipart(multipart_data, flags=zmq.NOBLOCK)
                else:
                    zmq_socket.sock.send_multipart(multipart_data)
                    if not ZMQ_SOCKET_DISABLE_ACK_CHECK:
                        resp = (
                            zmq_socket.sock.recv()
                        )  # wait for ack to ensure the request is processed
                        if resp != b"ACK":
                            raise Exception(f"unexpected ack: {resp}")

            send_success = True
            break
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed to send multipart data via ZMQ socket: {e}, "
                f"description={desc}, endpoint={na.to_tcp()}, bootstrap_room={bootstrap_room}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay_ms / 1000.0)

    elapsed_seconds = float(f"{time.perf_counter() - start_time:.4f}")

    if not send_success:
        logger.error(
            f"Failed to send multipart data via ZMQ socket after {max_retries} attempts, "
            f"description={desc}, endpoint={na.to_tcp()}, bootstrap_room={bootstrap_room}, elapsed_seconds={elapsed_seconds}"
        )


class FastQueue:
    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
    """Vectorised NumPy implementation."""
    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups
