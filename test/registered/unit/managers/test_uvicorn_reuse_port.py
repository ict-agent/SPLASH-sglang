"""Unit tests for per-worker uvicorn SO_REUSEPORT hooks."""

import asyncio
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import uvicorn
from uvicorn.server import Server

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.multi_tokenizer_mixin as multi_tokenizer_mixin
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _FakeSocket:
    def __init__(self, *args, family=socket.AF_INET, **kwargs):
        self.family = family
        self.options = []
        self.bound_address = None
        self.closed = False

    def setsockopt(self, level, option, value):
        self.options.append((level, option, value))

    def bind(self, address):
        self.bound_address = address

    def close(self):
        self.closed = True


class _FakeZmqSocket:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)


class TestUvicornReusePort(CustomTestCase):
    def setUp(self):
        self.original_bind_socket = uvicorn.Config.bind_socket
        self.original_startup = Server.startup
        multi_tokenizer_mixin._reuse_port_parent_patched = False
        multi_tokenizer_mixin._reuse_port_child_patched = False

    def tearDown(self):
        uvicorn.Config.bind_socket = self.original_bind_socket
        Server.startup = self.original_startup
        multi_tokenizer_mixin._reuse_port_parent_patched = False
        multi_tokenizer_mixin._reuse_port_child_patched = False

    @unittest.skipUnless(hasattr(socket, "SO_REUSEPORT"), "SO_REUSEPORT unavailable")
    def test_parent_hook_sets_reuseport_during_bind(self):
        created = []

        def fake_bind_socket(_config):
            sock = socket.socket(family=socket.AF_INET)
            created.append(sock)
            return sock

        uvicorn.Config.bind_socket = fake_bind_socket
        multi_tokenizer_mixin.monkey_patch_uvicorn_parent_reuse_port()

        with patch.object(socket, "socket", side_effect=_FakeSocket):
            result = uvicorn.Config.bind_socket(SimpleNamespace())

        self.assertIs(result, created[0])
        self.assertIn((socket.SOL_SOCKET, socket.SO_REUSEPORT, 1), result.options)

    @unittest.skipUnless(hasattr(socket, "SO_REUSEPORT"), "SO_REUSEPORT unavailable")
    def test_child_hook_replaces_inherited_socket(self):
        async def fake_startup(_server, sockets=None):
            return sockets

        Server.startup = fake_startup
        multi_tokenizer_mixin.monkey_patch_uvicorn_child_reuse_port()

        inherited = _FakeSocket()
        fake_server = SimpleNamespace(
            config=SimpleNamespace(
                workers=4,
                uds=None,
                fd=None,
                host="127.0.0.1",
                port=0,
            )
        )
        sockets = asyncio.run(Server.startup(fake_server, sockets=[inherited]))

        self.assertTrue(inherited.closed)
        self.assertEqual(len(sockets), 1)
        own_socket = sockets[0]
        self.assertEqual(own_socket.getsockname()[0], "127.0.0.1")
        self.assertGreater(own_socket.getsockname()[1], 0)
        self.assertEqual(
            own_socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT), 1
        )
        own_socket.close()

    def test_child_hook_leaves_single_worker_socket_unchanged(self):
        async def fake_startup(_server, sockets=None):
            return sockets

        Server.startup = fake_startup
        multi_tokenizer_mixin.monkey_patch_uvicorn_child_reuse_port()

        inherited = _FakeSocket()
        fake_server = SimpleNamespace(
            config=SimpleNamespace(
                workers=1,
                uds=None,
                fd=None,
                host="127.0.0.1",
                port=30000,
            )
        )
        sockets = asyncio.run(Server.startup(fake_server, sockets=[inherited]))

        self.assertEqual(sockets, [inherited])
        self.assertFalse(inherited.closed)

    def test_attach_ipc_updates_cached_batch_items(self):
        req = GenerateReqInput(text=["hello", "world"], sampling_params=[{}, {}])
        req.normalize_batch_and_arguments()
        first_cached = req[0]

        worker = SimpleNamespace(tokenizer_ipc_name="ipc://worker-0")
        multi_tokenizer_mixin.TokenizerWorker._attach_multi_http_worker_info(
            worker, req
        )

        self.assertEqual(req.http_worker_ipc, "ipc://worker-0")
        self.assertEqual(first_cached.http_worker_ipc, "ipc://worker-0")
        self.assertEqual(req[1].http_worker_ipc, "ipc://worker-0")

    def test_sender_wrapper_attaches_ipc_to_tokenized_batch(self):
        socket = _FakeZmqSocket()
        wrapper = multi_tokenizer_mixin.SenderWrapper(
            SimpleNamespace(tokenizer_ipc_name="ipc://worker-0"), socket
        )
        batch = BatchTokenizedGenerateReqInput(
            batch=[
                TokenizedGenerateReqInput(
                    "hello",
                    [1],
                    None,
                    {},
                    False,
                    -1,
                    0,
                    None,
                    False,
                    rid="rid-0",
                ),
                TokenizedGenerateReqInput(
                    "world",
                    [2],
                    None,
                    {},
                    False,
                    -1,
                    0,
                    None,
                    False,
                    rid="rid-1",
                ),
            ]
        )

        wrapper.send_pyobj(batch)

        self.assertIs(socket.sent[0], batch)
        self.assertEqual(batch.rids, ["rid-0", "rid-1"])
        self.assertEqual(batch.http_worker_ipcs, ["ipc://worker-0", "ipc://worker-0"])
        self.assertEqual(batch.batch[0].http_worker_ipc, "ipc://worker-0")
        self.assertEqual(batch.batch[1].http_worker_ipc, "ipc://worker-0")


if __name__ == "__main__":
    unittest.main()
