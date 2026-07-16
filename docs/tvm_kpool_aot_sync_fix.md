# TVM/KPool AOT Sync Fix Guide

This note records the fix flow used to make the GLM5 MTP KPool path work in a
fresh DCU container. Use it when syncing the same patch set to another
environment.

## Symptom

The server may fail before model loading with:

```text
ValueError: Cannot find model module. 'Glm5NextForCausalLM' is not a registered model ...
```

Or the server starts, but profiling still shows the old JIT
`update_kpool_write_plan_cuda_graph` path instead of the AOT
`kpool_write_plan_kernel`.

## Root Causes

1. `sglang.srt.models.glm5_next` import can fail if
   `python/sglang/srt/layers/attention/nsa/kpool/kernels.py` does not export
   `kpool_dequantize_fp8_paged_kv_cache`. ModelRegistry then skips
   `Glm5NextForCausalLM`, and SGLang falls through to the Transformers
   compatibility path.

2. On DCU/HIP, `sgl-kernel` is installed through:

```bash
python3 setup_hip.py install
```

Adding the AOT operator only to the CUDA/CMake path is not enough. The HIP build
must also compile and register the operator.

3. The Python planner opens the AOT path only when this succeeds:

```python
from sgl_kernel import kpool_write_plan
```

If the installed package does not export it, `_get_aot_kpool_write_plan()`
returns `None` and the code intentionally falls back to the JIT kernel.

## Files To Sync

Sync these files together:

```text
python/sglang/srt/layers/attention/nsa/kpool/planner.py
python/sglang/srt/layers/attention/nsa/kpool/kernels.py
sgl-kernel/csrc/elementwise/kpool_write_plan.cu
sgl-kernel/csrc/common_extension.cc
sgl-kernel/csrc/common_extension_rocm.cc
sgl-kernel/include/sgl_kernel_ops.h
sgl-kernel/python/sgl_kernel/elementwise.py
sgl-kernel/python/sgl_kernel/__init__.py
sgl-kernel/setup_hip.py
sgl-kernel/setup_rocm.py
sgl-kernel/CMakeLists.txt
```

For HIP/DCU, the critical additions are:

- `setup_hip.py` includes `csrc/elementwise/kpool_write_plan.cu` in `sources`.
- `common_extension_rocm.cc` registers `kpool_write_plan`.
- `sgl_kernel_ops.h` declares `kpool_write_plan`.
- `elementwise.py` and `__init__.py` export the Python wrapper.

## Install On DCU/HIP

Inside the container:

```bash
cd /work/codes/sglang-model/sgl-kernel
rm -rf build dist *.egg-info
python3 setup_hip.py install
```

Then verify:

```bash
python3 - <<'PY'
import torch
import sgl_kernel

print("sgl_kernel file:", sgl_kernel.__file__)
print("has python attr:", hasattr(sgl_kernel, "kpool_write_plan"))
print("has torch op:", hasattr(torch.ops.sgl_kernel, "kpool_write_plan"))

from sglang.srt.layers.attention.nsa.kpool.planner import _get_aot_kpool_write_plan
print("planner aot fn:", _get_aot_kpool_write_plan())
PY
```

Expected output:

```text
has python attr: True
has torch op: True
planner aot fn: <function kpool_write_plan ...>
```

## Smoke Test The Operator

```bash
python3 - <<'PY'
import torch
import sgl_kernel

B, N, max_pages = 2, 3, 8
pool_size, slots_per_page = 16, 4

write_start = torch.tensor([15, 33], dtype=torch.int32, device="cuda")
req_pool_indices = torch.tensor([7, 9], dtype=torch.int64, device="cuda")
real_page_table = torch.arange(
    B * N * max_pages, dtype=torch.int32, device="cuda"
).view(B * N, max_pages)

req_out = torch.empty(B, dtype=torch.int64, device="cuda")
ws_out = torch.empty(B, dtype=torch.int32, device="cuda")
tail_out = torch.empty(B, dtype=torch.int32, device="cuda")
loc_out = torch.empty(B, dtype=torch.int64, device="cuda")
pool_q = torch.empty(B * N, dtype=torch.int32, device="cuda")
seq_q = torch.empty(B * N, dtype=torch.int32, device="cuda")

sgl_kernel.kpool_write_plan(
    write_start,
    req_pool_indices,
    real_page_table,
    req_out,
    ws_out,
    tail_out,
    loc_out,
    pool_q,
    seq_q,
    pool_size,
    N,
    slots_per_page,
)
torch.cuda.synchronize()
print(req_out.cpu().tolist(), ws_out.cpu().tolist(), tail_out.cpu().tolist())
PY
```

## Validate GLM5 Registration

Before launching the full service, check:

```bash
cd /work/codes/sglang-model
export PYTHONPATH=/work/codes/sglang-model/python:${PYTHONPATH:-}
python3 - <<'PY'
from sglang.srt.models.registry import ModelRegistry

archs = ModelRegistry.get_supported_archs()
print("registered_causal", "Glm5NextForCausalLM" in archs)
print("registered_vlm", "Glm5NextForConditionalGeneration" in archs)
PY
```

Both values should be `True`.

## Launch Validation

```bash
cd /work/codes/sglang-model
wrapper=/tmp/glm5_mtp_kpool_start.sh
sed '$ s/[[:space:]]*\\$//' /work/glm5_n/glm5_mtp_kpool.sh > "$wrapper"
chmod +x "$wrapper"

export PYTHONPATH=/work/codes/sglang-model/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/dtk/lib:/opt/dtk/hip/lib:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}

mkdir -p /work/glm5_n/logs
log=/work/glm5_n/logs/glm5_mtp_$(date +%Y%m%d_%H%M%S).log
nohup bash "$wrapper" > "$log" 2>&1 &
echo "$log"
```

Check:

```bash
curl -sf http://127.0.0.1:30000/health
grep -Ei 'Cannot find model module|Ignore import error when loading sglang.srt.models.glm5_next|Scheduler hit an exception|Traceback' "$log" | tail
```

The old `Cannot find model module` error should be gone. In a new profile, the
write-plan update should appear as the AOT custom op path rather than the JIT
fallback.
