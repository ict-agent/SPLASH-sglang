import runpy
import sys
from pathlib import Path


def test_launch_server_spawn_bootstrap_is_lightweight():
    """Uvicorn spawn must reach its child target before importing SGLang."""
    repo_root = Path(__file__).resolve().parents[4]
    launch_server = repo_root / "python" / "sglang" / "launch_server.py"
    modules_before = set(sys.modules)

    runpy.run_path(str(launch_server), run_name="__mp_main__")

    imported = set(sys.modules) - modules_before
    assert not any(name == "sglang" or name.startswith("sglang.") for name in imported)
