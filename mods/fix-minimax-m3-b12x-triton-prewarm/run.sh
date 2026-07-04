#!/bin/bash
set -euo pipefail
# Precompile serving-config-specific Triton helpers used by the MiniMax-M3 b12x
# stack before vLLM enables jit_monitor / accepts inference traffic.
#
# The heavy first-request JITs this covers include:
#   * vLLM slot mapping: _compute_slot_mapping_kernel
#   * b12x nvfp4 main-KV write: _nvfp4_write_triton
#   * b12x nvfp4 index-K write: _cache_write_nvfp4_kernel
#   * b12x MSA prefill/verify metadata Triton helpers
#   * EAGLE3 padded-input / slot-mapping metadata helpers
#
# The implementation lives in ../b12x/b12x/vllm/minimax_m3/triton_prewarm.py
# and is called from GPUWorker.compile_or_warm_up_model immediately after vLLM's
# stock kernel_warmup(), before CUDA graph capture and before jit_monitor starts.

PYTHON=${PYTHON:-python3}
$PYTHON - <<'PY'
import importlib.util, os, py_compile, shutil
from pathlib import Path

assert importlib.util.find_spec("b12x") is not None, "b12x not installed in image"
assert importlib.util.find_spec("b12x.vllm.minimax_m3.triton_prewarm") is not None, (
    "b12x MiniMax-M3 Triton prewarm module missing from image; rebuild "
    "vllm-node-minimax-m3-b12x from Dockerfile.b12x"
)
vroot = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
GPU_WORKER = vroot / "v1/worker/gpu_worker.py"
assert GPU_WORKER.exists(), f"missing {GPU_WORKER}"

t = GPU_WORKER.read_text()
anchor = (
    "        # Warmup and tune the kernels used during model execution before\n"
    "        # cuda graph capture.\n"
    "        kernel_warmup(self)\n"
)
assert t.count(anchor) == 1, f"gpu_worker kernel_warmup anchor count={t.count(anchor)}"
insert = anchor + (
    "\n"
    "        # MiniMax-M3 b12x: vLLM's generic dummy runs do not hit several\n"
    "        # serving-shape Triton helpers (nvfp4 cache writes, MSA metadata,\n"
    "        # EAGLE3 metadata). Prewarm them here, before CUDA graph capture and\n"
    "        # before jit_monitor is activated, so the first real request does not\n"
    "        # spend minutes compiling kernels.\n"
    "        if os.environ.get(\"B12X_TRITON_PREWARM\", \"1\") != \"0\":\n"
    "            try:\n"
    "                from b12x.vllm.minimax_m3.triton_prewarm import (\n"
    "                    prewarm_triton_kernels_for_worker,\n"
    "                )\n"
    "                prewarm_triton_kernels_for_worker(self)\n"
    "            except Exception:\n"
    "                logger.exception(\"MiniMax-M3 b12x Triton prewarm failed\")\n"
    "                if os.environ.get(\"B12X_TRITON_PREWARM_STRICT\", \"0\") == \"1\":\n"
    "                    raise\n"
)
if "prewarm_triton_kernels_for_worker(self)" not in t:
    t = t.replace(anchor, insert, 1)
    GPU_WORKER.write_text(t)
    print(f"Patched {GPU_WORKER}: b12x Triton prewarm hook installed")
else:
    print(f"{GPU_WORKER}: b12x Triton prewarm hook already installed")

py_compile.compile(str(GPU_WORKER), doraise=True)
for d in (GPU_WORKER.parent / "__pycache__",):
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
print("MiniMax-M3 b12x Triton prewarm mod applied.")
PY
