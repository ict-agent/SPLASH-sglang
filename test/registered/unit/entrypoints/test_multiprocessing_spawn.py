"""Regression tests for the lightweight multiprocessing spawn bootstrap."""

import importlib.machinery
import multiprocessing.spawn
import runpy
import subprocess
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BOOTSTRAP_FILE = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "entrypoints"
    / "multiprocessing_spawn.py"
)

# Load the registry without importing the SGLang package. This test deliberately
# remains stdlib-only so it can detect accidental heavyweight bootstrap imports.
_CI_REGISTER = runpy.run_path(
    str(_REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py")
)
register_cpu_ci = _CI_REGISTER["register_cpu_ci"]

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

_SPAWN_MODULE = runpy.run_path(
    str(_BOOTSTRAP_FILE), run_name="_multiprocessing_spawn_test"
)
_USE_MULTIPROCESSING_SPAWN_BOOTSTRAP = _SPAWN_MODULE[
    "use_multiprocessing_spawn_bootstrap"
]
_MULTIPROCESS_WITH_STARTUP_WAIT = _SPAWN_MODULE[
    "_multiprocess_with_startup_wait"
]
_USE_UVICORN_WORKER_STARTUP_WAIT = _SPAWN_MODULE[
    "use_uvicorn_worker_startup_wait"
]
_MISSING = object()


def _restore_attribute(module, name, original_value):
    if original_value is _MISSING:
        try:
            delattr(module, name)
        except AttributeError:
            pass
    else:
        setattr(module, name, original_value)


class TestMultiprocessingSpawnBootstrap(unittest.TestCase):
    def test_registered_ci_metadata_is_collectable(self):
        registries = _CI_REGISTER["collect_tests"]([str(Path(__file__).resolve())])

        self.assertEqual(len(registries), 1)
        self.assertEqual(registries[0].effective_suite, "stage-a-test-cpu")

    def test_bootstrap_does_not_import_sglang(self):
        """The pre-target child bootstrap must not initialize SGLang/HIP."""
        bootstrap_file = _BOOTSTRAP_FILE.resolve()
        check_code = (
            "import runpy, sys; "
            "runpy.run_path(sys.argv[1], run_name='__mp_main__'); "
            "bad = sorted(name for name in sys.modules "
            "if name == 'sglang' or name.startswith('sglang.')); "
            "assert not bad, bad"
        )

        result = subprocess.run(
            [sys.executable, "-I", "-c", check_code, str(bootstrap_file)],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_all_entrypoints_use_bootstrap_and_restore_main(self):
        """Cover both ``python -m`` and the recommended ``sglang serve``."""
        main_module = sys.modules["__main__"]
        saved_spec = getattr(main_module, "__spec__", _MISSING)
        saved_file = getattr(main_module, "__file__", _MISSING)
        bootstrap_file = _BOOTSTRAP_FILE.resolve()
        entrypoints = (
            (
                "python -m sglang.launch_server",
                importlib.machinery.ModuleSpec("sglang.launch_server", loader=None),
                "python/sglang/launch_server.py",
            ),
            ("sglang serve", None, "/usr/local/bin/sglang"),
        )

        try:
            for name, original_spec, original_file in entrypoints:
                with self.subTest(entrypoint=name):
                    main_module.__spec__ = original_spec
                    main_module.__file__ = original_file

                    with _USE_MULTIPROCESSING_SPAWN_BOOTSTRAP():
                        self.assertIsNone(main_module.__spec__)
                        self.assertEqual(
                            Path(main_module.__file__).resolve(), bootstrap_file
                        )
                        preparation_data = multiprocessing.spawn.get_preparation_data(
                            "uvicorn-worker"
                        )
                        self.assertNotIn("init_main_from_name", preparation_data)
                        self.assertEqual(
                            Path(preparation_data["init_main_from_path"]).resolve(),
                            bootstrap_file,
                        )

                    self.assertIs(main_module.__spec__, original_spec)
                    self.assertEqual(main_module.__file__, original_file)
        finally:
            _restore_attribute(main_module, "__spec__", saved_spec)
            _restore_attribute(main_module, "__file__", saved_file)

    def test_initial_workers_reach_ready_before_runtime_health_checks(self):
        events = []

        class FakeProcess:
            def __init__(self, pid, readiness):
                self.pid = pid
                self.readiness = iter(readiness)
                self.process = self

            def is_alive(self):
                return True

            def is_ready(self, timeout):
                events.append((self.pid, timeout))
                return next(self.readiness)

        class FakeMultiprocess:
            def __init__(self, ready_states):
                self.processes = [
                    FakeProcess(pid, ready)
                    for pid, ready in enumerate(ready_states, start=100)
                ]
                self.should_exit = threading.Event()

            def init_processes(self):
                events.append("started")

            def handle_signals(self):
                pass

        class FakeLogger:
            def __init__(self):
                self.errors = []

            def info(self, *args):
                pass

            def error(self, *args):
                self.errors.append(args)

        logger = FakeLogger()
        multiprocess_class = _MULTIPROCESS_WITH_STARTUP_WAIT(
            FakeMultiprocess, 30, logger
        )

        supervisor = multiprocess_class([[False, True], [True]])
        supervisor.init_processes()

        self.assertEqual(events[0], "started")
        self.assertEqual([event[0] for event in events[1:]], [100, 100, 101])
        self.assertTrue(all(0 < event[1] <= 1 for event in events[1:]))
        self.assertFalse(supervisor.should_exit.is_set())
        self.assertEqual(logger.errors, [])

    def test_initial_worker_startup_failure_stops_parent(self):
        waited_pids = []

        class FakeProcess:
            def __init__(self, pid, alive):
                self.pid = pid
                self.alive = alive
                self.process = self

            def is_alive(self):
                waited_pids.append(self.pid)
                return self.alive

            def is_ready(self, timeout):
                raise AssertionError("dead worker must not be pinged")

        class FakeMultiprocess:
            def __init__(self):
                self.processes = [
                    FakeProcess(100, False),
                    FakeProcess(101, True),
                ]
                self.should_exit = threading.Event()

            def init_processes(self):
                pass

            def handle_signals(self):
                pass

        class FakeLogger:
            def __init__(self):
                self.errors = []

            def info(self, *args):
                pass

            def error(self, *args):
                self.errors.append(args)

        logger = FakeLogger()
        multiprocess_class = _MULTIPROCESS_WITH_STARTUP_WAIT(
            FakeMultiprocess, 30, logger
        )

        supervisor = multiprocess_class()
        supervisor.init_processes()

        self.assertEqual(waited_pids, [100])
        self.assertTrue(supervisor.should_exit.is_set())
        self.assertEqual(len(logger.errors), 1)

    def test_uvicorn_supervisor_patch_is_scoped(self):
        class FakeMultiprocess:
            pass

        fake_uvicorn = types.ModuleType("uvicorn")
        fake_uvicorn.__path__ = []
        fake_uvicorn_main = types.ModuleType("uvicorn.main")
        fake_uvicorn_main.Multiprocess = FakeMultiprocess

        with mock.patch.dict(
            sys.modules,
            {
                "uvicorn": fake_uvicorn,
                "uvicorn.main": fake_uvicorn_main,
            },
        ):
            with _USE_UVICORN_WORKER_STARTUP_WAIT(30):
                patched = fake_uvicorn_main.Multiprocess
                self.assertTrue(issubclass(patched, FakeMultiprocess))
                self.assertIsNot(patched, FakeMultiprocess)

            self.assertIs(fake_uvicorn_main.Multiprocess, FakeMultiprocess)


if __name__ == "__main__":
    unittest.main()
