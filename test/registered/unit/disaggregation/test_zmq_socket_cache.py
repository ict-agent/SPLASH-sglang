"""Unit tests for ZMQSocketCache (disaggregation/common/utils.py).

Verifies the strict-capacity lease design:
- busy (leased) sockets stay in the cache and count against capacity, so the
  number of open sockets can never exceed ``capacity``;
- when the cache is full and every entry is busy, lease() blocks
  (backpressure) instead of opening an extra socket;
- only idle entries are LRU-evicted;
- sends on the same socket are serialized;
- creation failures restore capacity/condition state.
"""

import ast
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import zmq  # noqa: E402

import sglang.srt.disaggregation.common.utils as disagg_utils  # noqa: E402
from sglang.srt.disaggregation.common.utils import (  # noqa: E402
    ZMQSocketCache,
    compute_zmq_socket_cache_capacity,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class _FakeSocket:
    """Minimal zmq.Socket stand-in with real closed-state semantics, so
    tests can distinguish "close was called" from "close succeeded"."""

    def __init__(self, socket_type):
        self.socket_type = socket_type
        self.closed = False
        self.fail_close = False
        self.connect = MagicMock()
        self.send_multipart = MagicMock()
        self.setsockopt = MagicMock()

    def close(self, linger=None):
        if self.fail_close:
            raise RuntimeError("close failed")
        self.closed = True


class _FakeContext:
    """Stands in for zmq.Context: creates fake sockets, counts real opens."""

    def __init__(self, max_sockets=16384):
        self._max_sockets = max_sockets
        self.created = []
        self.fail_next_connect = False

    def get(self, opt):
        assert opt == zmq.MAX_SOCKETS
        return self._max_sockets

    def socket(self, socket_type):
        sock = _FakeSocket(socket_type)
        if self.fail_next_connect:
            sock.connect.side_effect = RuntimeError("connect refused")
        self.created.append(sock)
        return sock

    @property
    def open_count(self):
        """Number of sockets whose underlying fd is actually still open."""
        return sum(1 for s in self.created if not s.closed)


def _make_cache(capacity, ctx=None):
    ctx = ctx or _FakeContext()
    # Bypass the rlimit-based clamp so tests control capacity exactly.
    with patch.object(
        disagg_utils, "compute_zmq_socket_cache_capacity", return_value=capacity
    ):
        cache = ZMQSocketCache(ctx, capacity=capacity, desc="test")
    return cache, ctx


class TestLeaseBasics(unittest.TestCase):
    def test_same_endpoint_reuses_socket(self):
        cache, ctx = _make_cache(capacity=4)
        with cache.lease(zmq.PUSH, "tcp://1.1.1.1:1") as s1:
            pass
        with cache.lease(zmq.PUSH, "tcp://1.1.1.1:1") as s2:
            pass
        self.assertIs(s1, s2)
        self.assertEqual(len(ctx.created), 1)

    def test_send_serialized_across_threads(self):
        """Concurrent lease() of the same endpoint must serialize sends."""
        cache, ctx = _make_cache(capacity=2)
        in_critical = []
        overlap = []

        def worker():
            with cache.lease(zmq.PUSH, "tcp://1.1.1.1:1") as sock:
                in_critical.append(1)
                if len(in_critical) > 1:
                    overlap.append(True)
                time.sleep(0.05)
                in_critical.pop()
                sock.send_multipart([b"x"])

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(overlap, [], "sends on one socket must not overlap")
        self.assertEqual(len(ctx.created), 1)
        self.assertEqual(ctx.created[0].send_multipart.call_count, 4)

    def test_timeout_zero_fails_immediately_when_send_lock_busy(self):
        """A timeout=0 lease must not block on the entry's send_lock: if another
        thread is mid-send on the same endpoint, acquire the lock non-blockingly
        and raise instead of stalling the caller (e.g. the scheduler thread)."""
        cache, ctx = _make_cache(capacity=1)
        entered = threading.Event()
        release = threading.Event()

        def hold():
            with cache.lease(zmq.PUSH, "tcp://a:1"):
                entered.set()
                release.wait(timeout=10)

        ta = threading.Thread(target=hold)
        ta.start()
        self.assertTrue(entered.wait(timeout=5))

        start = time.monotonic()
        with self.assertRaises(RuntimeError):
            with cache.lease(zmq.PUSH, "tcp://a:1", timeout=0.0):
                pass
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 1.0, "timeout=0 must fail, not block on send_lock")

        release.set()
        ta.join(timeout=5)

        # The failed acquire must have rolled back users so the entry is
        # reusable once the holder releases it.
        with cache.lease(zmq.PUSH, "tcp://a:1"):
            pass


class TestStrictCapacity(unittest.TestCase):
    def test_busy_socket_not_evicted_capacity_1(self):
        """capacity=1: while A is leased, leasing B must BLOCK; open socket
        count never exceeds 1. After A is released, B closes A and proceeds."""
        cache, ctx = _make_cache(capacity=1)
        a_entered = threading.Event()
        a_release = threading.Event()
        b_got_socket = threading.Event()
        max_open = []

        def hold_a():
            with cache.lease(zmq.PUSH, "tcp://a:1"):
                a_entered.set()
                a_release.wait(timeout=10)

        def want_b():
            with cache.lease(zmq.PUSH, "tcp://b:1"):
                b_got_socket.set()

        ta = threading.Thread(target=hold_a)
        tb = threading.Thread(target=want_b)
        ta.start()
        self.assertTrue(a_entered.wait(timeout=5))
        tb.start()

        # B must be blocked while A is busy, and no extra socket opened.
        time.sleep(0.2)
        self.assertFalse(b_got_socket.is_set(), "B must wait while A is busy")
        self.assertEqual(ctx.open_count, 1)
        max_open.append(ctx.open_count)

        # Release A: B evicts idle A and proceeds.
        a_release.set()
        self.assertTrue(b_got_socket.wait(timeout=5))
        ta.join()
        tb.join()
        self.assertTrue(ctx.created[0].closed, "A must be closed")
        self.assertEqual(ctx.open_count, 1)
        self.assertLessEqual(max(max_open), 1)

    def test_lru_evicts_only_idle(self):
        """With capacity=2 and one busy entry, the idle entry is the victim."""
        cache, ctx = _make_cache(capacity=2)
        release_a = threading.Event()

        def hold_a():
            with cache.lease(zmq.PUSH, "tcp://a:1"):
                release_a.wait(timeout=10)

        ta = threading.Thread(target=hold_a)
        ta.start()
        time.sleep(0.1)
        # B idle in cache; A busy. Creating C must evict B (idle), not A.
        with cache.lease(zmq.PUSH, "tcp://b:1"):
            pass
        with cache.lease(zmq.PUSH, "tcp://c:1"):
            pass
        a_sock, b_sock, c_sock = ctx.created
        self.assertFalse(a_sock.closed, "busy A must not be evicted")
        self.assertTrue(b_sock.closed, "idle B must be the victim")
        self.assertFalse(c_sock.closed)
        self.assertEqual(cache.open_socket_count(), 2)
        release_a.set()
        ta.join()

    def test_all_busy_times_out(self):
        cache, ctx = _make_cache(capacity=1)
        release = threading.Event()

        def hold():
            with cache.lease(zmq.PUSH, "tcp://a:1"):
                release.wait(timeout=10)

        t = threading.Thread(target=hold)
        t.start()
        time.sleep(0.1)
        with patch.object(disagg_utils, "ZMQ_SOCKET_LEASE_TIMEOUT_S", 0.2):
            with self.assertRaises(RuntimeError):
                with cache.lease(zmq.PUSH, "tcp://b:1"):
                    pass
        self.assertEqual(ctx.open_count, 1)
        release.set()
        t.join()


class TestFailureRecovery(unittest.TestCase):
    def test_connect_failure_restores_capacity(self):
        cache, ctx = _make_cache(capacity=1)
        ctx.fail_next_connect = True
        with self.assertRaises(RuntimeError):
            with cache.lease(zmq.PUSH, "tcp://bad:1"):
                pass
        self.assertEqual(cache.open_socket_count(), 0)
        # The half-created socket must be closed, not leaked.
        self.assertTrue(ctx.created[0].closed)
        # Cache must still be usable afterwards.
        ctx.fail_next_connect = False
        with cache.lease(zmq.PUSH, "tcp://good:1") as sock:
            self.assertIsNotNone(sock)
        self.assertEqual(cache.open_socket_count(), 1)

    def test_close_failure_quarantines_and_blocks_replacement(self):
        """A victim whose close() fails must stay counted against capacity
        (quarantine) and must NOT be replaced by a new socket -- the real
        open fd count never exceeds capacity."""
        cache, ctx = _make_cache(capacity=1)
        with cache.lease(zmq.PUSH, "tcp://a:1"):
            pass
        ctx.created[0].fail_close = True
        # A's close fails -> quarantined, still counted -> B must NOT get a
        # slot; with nothing evictable, lease must backpressure then raise.
        with patch.object(disagg_utils, "ZMQ_SOCKET_LEASE_TIMEOUT_S", 0.2):
            with self.assertRaises(RuntimeError):
                with cache.lease(zmq.PUSH, "tcp://b:1"):
                    pass
        self.assertEqual(len(ctx.created), 1, "no replacement socket created")
        self.assertLessEqual(ctx.open_count, 1, "real opens never exceed capacity")
        self.assertEqual(cache.open_socket_count(), 1)  # the quarantined one

    def test_quarantined_socket_slot_recovered_after_close_succeeds(self):
        """Once a later close attempt succeeds, the quarantined slot is
        reclaimed and the cache is fully usable again."""
        cache, ctx = _make_cache(capacity=1)
        with cache.lease(zmq.PUSH, "tcp://a:1"):
            pass
        ctx.created[0].fail_close = True
        with patch.object(disagg_utils, "ZMQ_SOCKET_LEASE_TIMEOUT_S", 0.2):
            with self.assertRaises(RuntimeError):
                with cache.lease(zmq.PUSH, "tcp://b:1"):
                    pass
        # The fd becomes closable (e.g. transient EINTR-like failure cleared).
        ctx.created[0].fail_close = False
        with cache.lease(zmq.PUSH, "tcp://b:1") as sock:
            self.assertIsNotNone(sock)
        self.assertTrue(ctx.created[0].closed, "quarantined socket closed")
        self.assertEqual(cache.open_socket_count(), 1)
        self.assertLessEqual(ctx.open_count, 1)

    def test_quarantine_reclaim_wakes_waiter_despite_busy_entry(self):
        """A waiter blocked on a full cache must be woken the moment a
        quarantined slot is reclaimed -- even though the only lease release
        happening belongs to an entry with OTHER remaining users (so the
        users==0 notify never fires). Regression test: without the
        drain-reclaim notify, the waiter sleeps the full lease timeout."""
        cache, ctx = _make_cache(capacity=2)
        # Fill both slots; A will be quarantined later.
        with cache.lease(zmq.PUSH, "tcp://a:1"):
            pass
        release_b1 = threading.Event()
        b1_released = threading.Event()

        def hold_b_first():
            with cache.lease(zmq.PUSH, "tcp://b:1"):
                release_b1.wait(timeout=10)
                b1_released.set()

        tb = threading.Thread(target=hold_b_first)
        tb.start()
        time.sleep(0.1)

        # Make A's close fail, then force C to evict A into quarantine.
        ctx.created[0].fail_close = True
        with patch.object(disagg_utils, "ZMQ_SOCKET_LEASE_TIMEOUT_S", 0.5):
            with self.assertRaises(RuntimeError):
                # Evicts idle A -> close fails -> quarantine; B busy ->
                # backpressure -> timeout. Cache now: {B} + quarantine{A}.
                with cache.lease(zmq.PUSH, "tcp://c:1"):
                    pass

        # Second concurrent user on B: entry.users == 2 now.
        release_b2 = threading.Event()

        def hold_b_second():
            with cache.lease(zmq.PUSH, "tcp://b:1"):
                release_b2.wait(timeout=10)

        tb2 = threading.Thread(target=hold_b_second)
        tb2.start()
        time.sleep(0.1)

        # Instrument A's close so we can deterministically observe when the
        # waiter's own acquire-miss drain has run (it must fail while
        # fail_close is still True, forcing the waiter into _cond.wait()).
        a_sock = ctx.created[0]
        close_count = [0]
        orig_close = a_sock.close

        def counting_close(linger=None):
            close_count[0] += 1
            return orig_close(linger=linger)

        a_sock.close = counting_close

        c_acquired = threading.Event()
        errors = []

        def want_c():
            try:
                with cache.lease(zmq.PUSH, "tcp://c:1"):
                    c_acquired.set()
            except RuntimeError as e:
                errors.append(e)

        with patch.object(disagg_utils, "_QUARANTINE_DRAIN_INTERVAL_S", 0.0):
            baseline = close_count[0]
            tc = threading.Thread(target=want_c)
            tc.start()

            # Wait until the waiter has attempted (and failed) its drain, i.e.
            # it is now blocked in _cond.wait(). Only then is it safe to make A
            # closable -- this makes the test deterministic rather than relying
            # on a fixed sleep.
            deadline = time.monotonic() + 5
            while close_count[0] == baseline and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(
                c_acquired.is_set(),
                "C must block while A is quarantined and B is busy",
            )

            # A becomes closable. Releasing the FIRST B user drops users to 1
            # (not 0), so the users==0 notify never fires; the release-path
            # drain must reclaim A and wake the blocked C waiter immediately.
            ctx.created[0].fail_close = False
            release_b1.set()
            self.assertTrue(
                c_acquired.wait(timeout=3),
                "waiter must be woken by quarantine reclaim "
                "(not sleep the full lease timeout)",
            )
            tc.join(timeout=5)

        b1_released.wait(timeout=5)
        tb.join(timeout=5)
        release_b2.set()
        tb2.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertTrue(ctx.created[0].closed, "quarantined A reclaimed")
        self.assertLessEqual(ctx.open_count, 2)

    def test_quarantine_reclaimed_under_all_hit_steady_state(self):
        """A quarantined slot must be reclaimed even when every subsequent
        lease is a cache hit (the acquire miss-path drain never runs) --
        the lease-release path drains periodically."""
        cache, ctx = _make_cache(capacity=2)
        release_b = threading.Event()

        def hold_b():
            with cache.lease(zmq.PUSH, "tcp://b:1"):
                release_b.wait(timeout=10)

        with cache.lease(zmq.PUSH, "tcp://a:1"):  # A idle in cache
            pass
        tb = threading.Thread(target=hold_b)
        tb.start()
        time.sleep(0.1)  # B leased (busy)
        ctx.created[0].fail_close = True
        with patch.object(disagg_utils, "ZMQ_SOCKET_LEASE_TIMEOUT_S", 0.2):
            with self.assertRaises(RuntimeError):
                # Evicts idle A -> close fails -> quarantine; B busy -> no
                # other victim -> backpressure -> timeout.
                with cache.lease(zmq.PUSH, "tcp://c:1"):
                    pass
        release_b.set()
        tb.join()
        self.assertEqual(cache.open_socket_count(), 2)  # B + quarantined A
        # A becomes closable; workload is now ALL HITS on b (steady state:
        # the acquire miss-path drain never runs again).
        ctx.created[0].fail_close = False
        with patch.object(disagg_utils, "_QUARANTINE_DRAIN_INTERVAL_S", 0.0):
            with cache.lease(zmq.PUSH, "tcp://b:1"):  # pure hit
                pass
        self.assertTrue(
            ctx.created[0].closed,
            "quarantined socket must be reclaimed on the release path",
        )
        self.assertEqual(cache.open_socket_count(), 1)

    def test_caller_exception_releases_lease(self):
        """An exception inside the lease body must release the refcount so
        the entry is evictable/reusable afterwards."""
        cache, ctx = _make_cache(capacity=1)
        with self.assertRaises(ValueError):
            with cache.lease(zmq.PUSH, "tcp://a:1"):
                raise ValueError("caller blew up")
        # Entry must be idle again: creating B evicts A without blocking.
        with cache.lease(zmq.PUSH, "tcp://b:1") as sock:
            self.assertIsNotNone(sock)
        self.assertTrue(ctx.created[0].closed, "A must be evictable")

    def test_connect_failure_under_contention_recovers(self):
        """A waiter that wins a freed slot but then fails connect must raise,
        close the half-built socket, restore capacity, and leave the cache
        usable (covers the notify_all on the connect-failure path)."""
        cache, ctx = _make_cache(capacity=1)
        a_entered = threading.Event()
        a_release = threading.Event()
        errors = []

        def hold_a():
            with cache.lease(zmq.PUSH, "tcp://a:1"):
                a_entered.set()
                a_release.wait(timeout=10)

        def want_b():
            try:
                with cache.lease(zmq.PUSH, "tcp://b:1"):
                    pass
            except RuntimeError as e:
                errors.append(e)

        ta = threading.Thread(target=hold_a)
        tb = threading.Thread(target=want_b)
        ta.start()
        self.assertTrue(a_entered.wait(timeout=5))
        tb.start()
        time.sleep(0.1)  # B is now waiting on the condition
        ctx.fail_next_connect = True
        a_release.set()  # B wakes, evicts idle A, create fails
        tb.join(timeout=5)
        ta.join(timeout=5)
        self.assertEqual(len(errors), 1, "B's create must fail loudly")
        self.assertEqual(cache.open_socket_count(), 0)
        self.assertTrue(ctx.created[0].closed, "A must be evicted")
        self.assertTrue(ctx.created[1].closed, "half-built B closed")
        # Cache must still be usable afterwards.
        ctx.fail_next_connect = False
        with cache.lease(zmq.PUSH, "tcp://c:1") as sock:
            self.assertIsNotNone(sock)


class TestKeyIsolation(unittest.TestCase):
    def test_socket_type_and_ipv6_are_separate_keys(self):
        cache, ctx = _make_cache(capacity=4)
        with cache.lease(zmq.PUSH, "tcp://a:1", is_ipv6=False):
            pass
        with cache.lease(zmq.PUSH, "tcp://a:1", is_ipv6=True):
            pass
        with cache.lease(zmq.PULL, "tcp://a:1", is_ipv6=False):
            pass
        self.assertEqual(len(ctx.created), 3)
        self.assertEqual(cache.open_socket_count(), 3)

    def test_sndtimeo_is_set(self):
        """Cached sockets must have a bounded send timeout so a dead peer
        cannot pin the lease forever."""
        cache, ctx = _make_cache(capacity=1)
        with cache.lease(zmq.PUSH, "tcp://a:1"):
            pass
        calls = {c.args[0]: c.args[1] for c in ctx.created[0].setsockopt.call_args_list}
        self.assertIn(zmq.SNDTIMEO, calls)
        self.assertGreater(calls[zmq.SNDTIMEO], 0)


class TestCapacityComputation(unittest.TestCase):
    def test_clamped_by_ctx_max_sockets(self):
        cap = compute_zmq_socket_cache_capacity(
            configured=100000, ctx_max_sockets=1000
        )
        self.assertLessEqual(cap, 1000 - 32)

    def test_clamped_by_rlimit(self):
        with patch.object(
            disagg_utils.resource, "getrlimit", return_value=(1024, 1024)
        ), patch.object(disagg_utils, "_current_open_fds", return_value=100):
            cap = compute_zmq_socket_cache_capacity(
                configured=100000, ctx_max_sockets=None
            )
        # 1024 - 100 open - max(256, 102) reserve = 668
        self.assertEqual(cap, 1024 - 100 - 256)

    def test_fd_budget_split_across_caches(self):
        """Two caches sharing the budget must each get half."""
        with patch.object(
            disagg_utils.resource, "getrlimit", return_value=(1024, 1024)
        ), patch.object(disagg_utils, "_current_open_fds", return_value=100):
            cap = compute_zmq_socket_cache_capacity(
                configured=100000, ctx_max_sockets=None, fd_budget_share=2
            )
        self.assertEqual(cap, (1024 - 100 - 256) // 2)

    def test_rlimit_infinity_ignored(self):
        with patch.object(
            disagg_utils.resource,
            "getrlimit",
            return_value=(disagg_utils.resource.RLIM_INFINITY, -1),
        ):
            cap = compute_zmq_socket_cache_capacity(
                configured=5000, ctx_max_sockets=None
            )
        self.assertEqual(cap, 5000)

    def test_never_below_one(self):
        with patch.object(
            disagg_utils.resource, "getrlimit", return_value=(64, 64)
        ), patch.object(disagg_utils, "_current_open_fds", return_value=60):
            cap = compute_zmq_socket_cache_capacity(
                configured=5000, ctx_max_sockets=None
            )
        self.assertEqual(cap, 1)


class TestSharedContexts(unittest.TestCase):
    def test_manager_and_receiver_use_separate_shared_contexts(self):
        """Each role owns one shared context; caches never create contexts."""
        ctx_m = _FakeContext()
        ctx_r = _FakeContext()
        cache_m, _ = _make_cache(capacity=2, ctx=ctx_m)
        cache_r, _ = _make_cache(capacity=2, ctx=ctx_r)
        with cache_m.lease(zmq.PUSH, "tcp://x:1"):
            pass
        with cache_r.lease(zmq.PUSH, "tcp://x:1"):
            pass
        self.assertEqual(len(ctx_m.created), 1)
        self.assertEqual(len(ctx_r.created), 1)

    def test_production_caches_split_fd_budget(self):
        """Lock down the production wiring in common/conn.py.

        The two class-level caches (CommonKVManager._push_socket_cache and
        CommonKVReceiver._push_socket_cache) must each pass fd_budget_share=2
        and reuse a per-role shared Context, otherwise their combined socket
        count can silently exceed the process FD budget (the core guarantee of
        this MR). We assert this from the source AST: importing common/conn
        here would pull torch/transformers into this lightweight test.
        """
        conn_path = os.path.join(os.path.dirname(disagg_utils.__file__), "conn.py")
        with open(conn_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        caches = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name not in ("CommonKVManager", "CommonKVReceiver"):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if any(
                    isinstance(t, ast.Name) and t.id == "_push_socket_cache"
                    for t in stmt.targets
                ):
                    caches[node.name] = stmt.value

        self.assertEqual(
            set(caches),
            {"CommonKVManager", "CommonKVReceiver"},
            "both roles must define a class-level _push_socket_cache",
        )
        for cls_name, call in caches.items():
            self.assertIsInstance(call, ast.Call, cls_name)
            self.assertEqual(call.func.id, "ZMQSocketCache", cls_name)
            kwargs = {kw.arg: kw.value for kw in call.keywords}
            self.assertIn("fd_budget_share", kwargs, cls_name)
            self.assertEqual(
                kwargs["fd_budget_share"].value,
                2,
                f"{cls_name} must split the process FD budget (fd_budget_share=2)",
            )
            ctx_arg = call.args[0]
            expected_ctx = "_push_ctx" if cls_name == "CommonKVManager" else "_ctx"
            self.assertEqual(ctx_arg.id, expected_ctx, cls_name)


if __name__ == "__main__":
    unittest.main()
