#!/usr/bin/env bash
# =============================================================================
# tvm_ffi ROCm 兼容性补丁
#
# 容器内的 apache-tvm-ffi==0.1.9 已有 extension.load_inline(..., backend=None)
# 原生 HIP 支持, 但缺少可 import 的 tvm_ffi.cpp.load_inline / tvm_ffi.cpp.load
# 子模块。本脚本写入 callable shim, 使 sglang 的 JIT 路径能正常 import,
# 同时标记 _sglang_hipcc_patched=True 跳过旧的 HIP monkey patch。
#
# 参考: scripts/hcu/README_tvm_ffi_fix.md
# =============================================================================
set -euo pipefail

log() { printf '[fix-tvm-ffi] %s\n' "$*"; }

log "writing tvm_ffi callable shims"

python3 - <<'PY'
from pathlib import Path
import tvm_ffi.cpp as cpp

cpp_dir = Path(cpp.__file__).resolve().parent

load_inline_shim = r'''# Compatibility shim for SGLang ROCm JIT with tvm-ffi 0.1.9 extension API.
import sys
import types
from . import extension as _extension

_extension._sglang_hipcc_patched = True

class _CallableExtensionProxy(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return _extension.load_inline(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_extension, name)

    def __setattr__(self, name, value):
        if name.startswith("__"):
            super().__setattr__(name, value)
        else:
            setattr(_extension, name, value)

_proxy = _CallableExtensionProxy(__name__)
_proxy.__dict__.update({"__file__": _extension.__file__, "__package__": __package__})
sys.modules[__name__] = _proxy
'''

load_shim = r'''# Compatibility shim for SGLang/tvm-ffi 0.1.9 extension API.
import sys
import types
from . import extension as _extension

class _CallableExtensionProxy(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return _extension.load(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_extension, name)

    def __setattr__(self, name, value):
        if name.startswith("__"):
            super().__setattr__(name, value)
        else:
            setattr(_extension, name, value)

_proxy = _CallableExtensionProxy(__name__)
_proxy.__dict__.update({"__file__": _extension.__file__, "__package__": __package__})
sys.modules[__name__] = _proxy
'''

(cpp_dir / "load_inline.py").write_text(load_inline_shim)
(cpp_dir / "load.py").write_text(load_shim)
print(f"wrote {cpp_dir / 'load_inline.py'}")
print(f"wrote {cpp_dir / 'load.py'}")
PY

log "verifying shims"

python3 - <<'PY'
import importlib

import tvm_ffi
print("tvm_ffi", getattr(tvm_ffi, "__version__", None), tvm_ffi.__file__)

for name in ["tvm_ffi.cpp.load_inline", "tvm_ffi.cpp.load"]:
    m = importlib.import_module(name)
    print("IMPORT_OK", name, getattr(m, "__file__", None), "callable=", callable(m))

from sglang.jit_kernel.utils import _patch_tvm_ffi_load_inline_for_hip
_patch_tvm_ffi_load_inline_for_hip()

import tvm_ffi.cpp.extension as ext
print("SGLANG_PATCH_OK", getattr(ext, "_sglang_hipcc_patched", False))
PY

log "done"
