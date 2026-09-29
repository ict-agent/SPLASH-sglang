import logging
import os
import resource
import struct
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def pack_list_of_buffers(buffers: List[bytes]) -> bytes:
    if not buffers:
        return b""
    n = len(buffers)
    header = struct.pack(f"<{n+1}I", n, *(len(b) for b in buffers))
    return header + b"".join(buffers)


def unpack_list_of_buffers(buf: bytes) -> List[bytes]:
    if buf == b"":
        return []
    (n,) = struct.unpack("<I", buf[:4])
    lens = struct.unpack(f"<{n}I", buf[4 : 4 + 4 * n])
    out = []
    offset = 4 + 4 * n
    for length in lens:
        out.append(buf[offset : offset + length])
        offset += length
    return out


def pack_int_lists(lists, fmt: str) -> bytes:
    return pack_list_of_buffers([struct.pack(f"<{len(a)}{fmt}", *a) for a in lists])


def unpack_int_lists(buf: bytes, fmt: str) -> List[List[int]]:
    width = struct.calcsize(fmt)
    return [
        list(struct.unpack(f"<{len(b)//width}{fmt}", b))
        for b in unpack_list_of_buffers(buf)
    ]


ZMQ_SOCKET_CACHE_SIZE = envs.SGLANG_DISAGGREGATION_ZMQ_SOCKET_CACHE_SIZE.get()
if ZMQ_SOCKET_CACHE_SIZE < 1:
    logger.warning(
        f"invalid SGLANG_DISAGGREGATION_ZMQ_SOCKET_CACHE_SIZE={ZMQ_SOCKET_CACHE_SIZE}, "
        f"falling back to 10240"
    )
    ZMQ_SOCKET_CACHE_SIZE = 10240

# Finite linger (ms) so close() gives just-queued control messages a chance to
# reach the wire instead of dropping them immediately. NOTE: this is best
# effort -- PUSH delivery here is at-most-once. Reliable delivery of critical
# control messages must come from an ACK/retry layer above, not from linger.
ZMQ_SOCKET_LINGER_MS = envs.SGLANG_DISAGGREGATION_ZMQ_SOCKET_LINGER_MS.get()
if ZMQ_SOCKET_LINGER_MS < 0:
    logger.warning(
        f"invalid SGLANG_DISAGGREGATION_ZMQ_SOCKET_LINGER_MS={ZMQ_SOCKET_LINGER_MS} "
        f"(negative means infinite linger and can hang close/term), falling back to 500"
    )
    ZMQ_SOCKET_LINGER_MS = 500

# How long lease() waits for an idle socket when the cache is full and every
# entry is busy, before giving up with an error.
ZMQ_SOCKET_LEASE_TIMEOUT_S = envs.SGLANG_DISAGGREGATION_ZMQ_SOCKET_LEASE_TIMEOUT_S.get()
if ZMQ_SOCKET_LEASE_TIMEOUT_S <= 0:
    logger.warning(
        f"invalid SGLANG_DISAGGREGATION_ZMQ_SOCKET_LEASE_TIMEOUT_S="
        f"{ZMQ_SOCKET_LEASE_TIMEOUT_S} (must be > 0), falling back to 30"
    )
    ZMQ_SOCKET_LEASE_TIMEOUT_S = 30

# Send timeout on cached PUSH sockets. Without it, a dead/hung peer whose HWM
# fills up blocks send_multipart() forever -- pinning the lease (1 capacity
# slot) and the per-endpoint send_lock permanently. With it, the send raises
# zmq.Again and the lease is released. 0 = non-blocking send (fail
# immediately when the queue is full).
ZMQ_SOCKET_SNDTIMEO_MS = envs.SGLANG_DISAGGREGATION_ZMQ_SOCKET_SNDTIMEO_MS.get()
if ZMQ_SOCKET_SNDTIMEO_MS < 0:
    logger.warning(
        f"invalid SGLANG_DISAGGREGATION_ZMQ_SOCKET_SNDTIMEO_MS="
        f"{ZMQ_SOCKET_SNDTIMEO_MS} (negative means block forever and can pin "
        f"the lease on a dead peer), falling back to 5000"
    )
    ZMQ_SOCKET_SNDTIMEO_MS = 5000

# NOTE on send queue semantics: we deliberately do NOT override SNDHWM
# (libzmq default 1000). With a finite HWM, a full send queue BLOCKS the
# send; together with SNDTIMEO it raises zmq.Again after the timeout -- no
# silent drops, no unbounded memory growth for a dead/hung peer. (An earlier
# revision set SNDHWM=0/unlimited "so slow peers never lose messages"; that
# was wrong on two counts: PUSH+HWM does not silently drop, and an unlimited
# queue turns a dead peer into unbounded process memory growth. AUX data can
# also be non-trivial in size.) Callers must handle zmq.Again per
# request/room; see sync_status_to_decode_endpoint / aux-data senders.
#
# Control messages (status notifications) and AUX data to the SAME decode
# endpoint intentionally share one cached PUSH socket. We deliberately do NOT
# split them into separate sockets/caches, because that would double the
# socket/FD count per endpoint -- directly fighting this cache's whole reason
# to exist (strict capacity + FD budget, see compute_zmq_socket_cache_capacity).
# The trade-off: a slow peer whose AUX backlog fills the HWM can also push a
# small status notification into zmq.Again; such a room then relies on the
# upper-layer lease/request timeout to fail instead of OOMing or blocking
# forever. If control-message timeliness ever becomes a problem, add a
# channel dimension to the cache key (or a small control-only cache) and
# re-evaluate the FD budget.

# FD headroom reserved for everything that is not a cached PUSH socket
# (model files, logs, HTTP connections, ZMQ internal pipes, ...).
_FD_RESERVE_MIN = 256
# Sockets reserved inside the shared ZMQ context for non-cached use.
_CTX_SOCKET_RESERVE = 32
# Minimum interval between quarantine drain attempts on the lease-release
# path (the acquire miss path drains unconditionally).
_QUARANTINE_DRAIN_INTERVAL_S = 30.0


def _current_open_fds() -> Optional[int]:
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return None


def compute_zmq_socket_cache_capacity(
    configured: int = ZMQ_SOCKET_CACHE_SIZE,
    ctx_max_sockets: Optional[int] = None,
    fd_budget_share: int = 1,
) -> int:
    """Clamp the configured cache size so cached sockets can never exhaust
    the process FD limit or the ZMQ context's MAX_SOCKETS.

    capacity = min(
        configured,
        ctx_max_sockets - reserve,                            # context budget
        (rlimit_soft - open_fds - headroom) // fd_budget_share,  # FD budget
    )

    ``fd_budget_share`` is the number of socket caches sharing this process's
    FD budget (e.g. 2: CommonKVManager + CommonKVReceiver), so their combined
    capacity stays within the limit.

    NOTE: the returned capacity is a one-time snapshot — it is computed when the
    class-level caches are created at import time. FDs opened later (model
    weights, HTTP, ZMQ internal pipes, ...) are NOT reflected; the fixed
    ``headroom`` (max(256, rlimit_soft//10)) is what absorbs that later growth.
    """
    capacity = configured

    if ctx_max_sockets is not None and ctx_max_sockets > 0:
        capacity = min(capacity, ctx_max_sockets - _CTX_SOCKET_RESERVE)

    rlimit_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if rlimit_soft not in (resource.RLIM_INFINITY, -1) and rlimit_soft > 0:
        open_fds = _current_open_fds() or 0
        fd_reserve = max(_FD_RESERVE_MIN, rlimit_soft // 10)
        fd_budget = (rlimit_soft - open_fds - fd_reserve) // max(fd_budget_share, 1)
        capacity = min(capacity, fd_budget)

    if capacity < 1:
        logger.warning(
            "zmq socket cache capacity clamped to 1 "
            f"(configured={configured}, ctx_max_sockets={ctx_max_sockets}, "
            f"rlimit_soft={rlimit_soft})"
        )
        capacity = 1
    elif capacity < configured:
        logger.warning(
            f"zmq socket cache capacity clamped: {configured} -> {capacity} "
            f"(ctx_max_sockets={ctx_max_sockets}, rlimit_soft={rlimit_soft}, "
            f"fd_budget_share={fd_budget_share})"
        )
    return capacity


class _SocketEntry:
    __slots__ = ("sock", "send_lock", "users")

    def __init__(self, sock: "zmq.Socket"):
        self.sock = sock
        self.send_lock = threading.Lock()
        self.users = 0


class ZMQSocketCache:
    """Strict-capacity LRU cache of connected ZMQ sockets over one shared
    Context.

    Guarantees:
      - The number of sockets owned by this cache (live entries + quarantined
        sockets whose close failed) never exceeds ``capacity``: busy (leased)
        entries stay in the cache and count against capacity; only idle
        entries (users == 0) are evicted; a victim whose close() fails is
        quarantined and keeps counting against capacity instead of being
        replaced. NOTE: physical OS FDs may briefly exceed capacity during
        the LINGER window after eviction (libzmq's reaper holds the fd for
        up to ZMQ_SOCKET_LINGER_MS while flushing queued messages); the
        strict bound is on cache-owned sockets.
      - When the cache is full and every entry is busy, ``lease()`` blocks
        (backpressure) until an entry becomes idle, instead of opening an
        extra socket.
      - Sends on the same socket are serialized via a per-entry lock
        (ZMQ sockets are not thread-safe).

    Usage::

        with cache.lease(zmq.PUSH, endpoint, is_ipv6) as sock:
            sock.send_multipart(message)
    """

    def __init__(
        self,
        ctx: "zmq.Context",
        capacity: Optional[int] = None,
        desc: str = "",
        fd_budget_share: int = 1,
    ):
        assert ctx is not None, "ZMQSocketCache requires a shared zmq.Context"
        self._ctx = ctx
        try:
            ctx_max_sockets = ctx.get(zmq.MAX_SOCKETS)
        except Exception:
            ctx_max_sockets = None
        self._capacity = compute_zmq_socket_cache_capacity(
            configured=capacity if capacity is not None else ZMQ_SOCKET_CACHE_SIZE,
            ctx_max_sockets=ctx_max_sockets,
            fd_budget_share=fd_budget_share,
        )
        self._desc = desc
        # key: (socket_type, endpoint, is_ipv6)
        self._entries: "OrderedDict[Tuple, _SocketEntry]" = OrderedDict()
        # Sockets whose close() failed: state unknown, never handed out again,
        # but still counted against capacity until a later close succeeds.
        self._quarantine: List["zmq.Socket"] = []
        self._last_quarantine_drain = 0.0
        self._cond = threading.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    def open_socket_count(self) -> int:
        with self._cond:
            return len(self._entries) + len(self._quarantine)

    @contextmanager
    def lease(
        self,
        socket_type: "zmq.SocketType",
        endpoint: str,
        is_ipv6: bool = False,
        timeout: Optional[float] = None,
    ):
        entry = self._acquire(socket_type, endpoint, is_ipv6, timeout)
        acquired = False
        try:
            # The timeout also bounds send_lock acquisition: on a cache hit
            # _acquire() returns immediately, but another thread may hold this
            # entry's send_lock (e.g. a transfer worker mid AUX/status send).
            # Without this, a timeout=0 caller (staging prefetch on the scheduler
            # thread) could still block here, defeating the non-blocking promise.
            if timeout is not None and timeout <= 0:
                acquired = entry.send_lock.acquire(blocking=False)
                if not acquired:
                    raise RuntimeError(
                        f"zmq socket send lock busy "
                        f"(desc={self._desc}, endpoint={endpoint})"
                    )
            elif timeout is not None:
                acquired = entry.send_lock.acquire(timeout=timeout)
                if not acquired:
                    raise RuntimeError(
                        f"zmq socket send lock busy for {timeout}s "
                        f"(desc={self._desc}, endpoint={endpoint})"
                    )
            else:
                entry.send_lock.acquire()
                acquired = True
            yield entry.sock
        finally:
            if acquired:
                entry.send_lock.release()
            with self._cond:
                entry.users -= 1
                if entry.users == 0:
                    self._cond.notify_all()
                if (
                    self._quarantine
                    and time.monotonic() - self._last_quarantine_drain
                    > _QUARANTINE_DRAIN_INTERVAL_S
                ):
                    # Periodic reclaim so a quarantined slot is recovered even
                    # under an all-hit steady state (where _acquire's miss-path
                    # drain would never run). Reclaimed slots must wake
                    # waiters: the users==0 notify above only fires when THIS
                    # entry went idle, which does not help waiters blocked on
                    # other endpoints.
                    if self._drain_quarantine_locked() > 0:
                        self._cond.notify_all()

    def _acquire(
        self,
        socket_type: "zmq.SocketType",
        endpoint: str,
        is_ipv6: bool,
        timeout: Optional[float] = None,
    ) -> _SocketEntry:
        key = (socket_type, endpoint, is_ipv6)
        deadline = None
        wait_timeout = ZMQ_SOCKET_LEASE_TIMEOUT_S if timeout is None else timeout
        with self._cond:
            while True:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.users += 1
                    self._entries.move_to_end(key, last=True)
                    return entry

                if self._quarantine:
                    # Cache miss: retry closing quarantined sockets so their
                    # slots (and fds) are reclaimed. Kept off the cache-hit
                    # hot path above. Reclaimed slots must wake waiters.
                    if self._drain_quarantine_locked() > 0:
                        self._cond.notify_all()

                if len(self._entries) + len(self._quarantine) < self._capacity:
                    break  # room for a new socket

                evicted = False
                for k in list(self._entries.keys()):  # oldest first
                    e = self._entries[k]
                    if e.users != 0:
                        continue
                    victim = self._entries.pop(k)
                    if self._close_entry(victim, k):
                        evicted = True
                        break
                    # Close failed and the fd may still be open: keep it
                    # counted against capacity (quarantine), do NOT create
                    # a replacement for its slot, and keep scanning for
                    # another idle victim.
                    self._quarantine.append(victim.sock)
                    victim.sock = None
                if evicted:
                    break

                # Cache full and every usable entry is busy (or the eviction
                # failed into quarantine): backpressure. Wait for a lease
                # release instead of opening an extra socket.
                if deadline is None:
                    deadline = time.monotonic() + wait_timeout
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"zmq socket cache exhausted: {self._capacity} sockets "
                        f"all busy for {wait_timeout}s "
                        f"(desc={self._desc}, endpoint={endpoint}, "
                        f"quarantined={len(self._quarantine)})"
                    )
                self._cond.wait(timeout=remaining)

            # Create + connect under the condition lock. zmq connect() is
            # non-blocking (the actual TCP connect happens on the context IO
            # thread) and endpoints here are numeric IPs, so this cannot
            # stall; doing it under the lock keeps capacity exact and means
            # no other thread can ever observe a half-created entry.
            sock = None
            try:
                sock = self._ctx.socket(socket_type)
                if is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)
                # SNDHWM intentionally NOT set: libzmq default (1000). See
                # the NOTE above ZMQ_SOCKET_SNDTIMEO_MS.
                # Bound sends so a full/dead peer raises zmq.Again after the
                # timeout instead of blocking the lease + send_lock forever.
                sock.setsockopt(zmq.SNDTIMEO, ZMQ_SOCKET_SNDTIMEO_MS)
                sock.connect(endpoint)
            except BaseException:
                if sock is not None:
                    try:
                        sock.close(linger=0)
                    except Exception:
                        pass
                self._cond.notify_all()
                raise
            entry = _SocketEntry(sock)
            entry.users = 1
            self._entries[key] = entry
            return entry

    def _drain_quarantine_locked(self) -> int:
        """Retry closing quarantined sockets; drop the ones that finally
        closed so their capacity slots become available again.

        Returns the number of reclaimed slots. The caller MUST notify_all()
        when it is > 0: waiters blocked on a full cache are not otherwise
        woken (a lease release whose entry still has other users does not
        notify)."""
        self._last_quarantine_drain = time.monotonic()
        if not self._quarantine:
            return 0
        remaining = []
        reclaimed = 0
        for sock in self._quarantine:
            # quiet=True: the initial eviction failure already logged an
            # error; retry failures would otherwise spam on every full-cache
            # miss.
            if self._close_sock(sock, "quarantined", quiet=True):
                reclaimed += 1
            else:
                remaining.append(sock)
        self._quarantine = remaining
        return reclaimed

    def _close_entry(self, entry: _SocketEntry, key) -> bool:
        sock = entry.sock
        if sock is None:
            return True
        closed = self._close_sock(sock, f"endpoint={key[1]}")
        if closed:
            entry.sock = None
            logger.debug(
                f"closed idle zmq socket (lru cache full): endpoint={key[1]}, "
                f"desc={self._desc}, capacity={self._capacity}"
            )
        return closed

    def _close_sock(self, sock: "zmq.Socket", what: str, quiet: bool = False) -> bool:
        """Close a socket; return True only if it is actually closed."""
        try:
            # Single pyzmq call (sets linger internally): avoids a separate
            # setsockopt() whose persistent failure would keep close() from
            # ever being attempted.
            sock.close(linger=ZMQ_SOCKET_LINGER_MS)
        except Exception as e:
            if getattr(sock, "closed", False):
                return True  # raised but the socket is closed anyway
            if not quiet:
                logger.error(
                    f"failed to close zmq socket ({what}): {e}, desc={self._desc}"
                )
            return False
        return True


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


def group_concurrent_contiguous_ranges(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.intp],
]:
    """Return source starts, destination starts, and lengths of contiguous runs."""
    if src_indices.size == 0:
        return src_indices, dst_indices, np.empty(0, dtype=np.intp)

    breaks = (
        np.flatnonzero((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1)) + 1
    )
    starts = np.empty(breaks.size + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = breaks
    ends = np.empty_like(starts)
    ends[:-1] = breaks
    ends[-1] = src_indices.size
    return src_indices[starts], dst_indices[starts], ends - starts
