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


if __name__ == "__main__":
    unittest.main()
