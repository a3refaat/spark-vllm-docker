#!/bin/bash
# Skip CUDA-graph memory pre-profiling. With
# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 the estimate is never
# subtracted, but the capture still runs INSIDE the memory-profiling
# window and its residue inflates non_torch_increase (double charge).
# Actual capture (post-KV-alloc) measured only 0.06 GiB.
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import py_compile
P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")
s = P.read_text()
old = "                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()"
if old in s:
    s = s.replace(old,
        "                import os as _os\n"
        "                if _os.environ.get('VLLM_SKIP_CUDAGRAPH_MEM_PROFILE', '0') != '1':\n"
        "                    cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()", 1)
    P.write_text(s)
    py_compile.compile(str(P), doraise=True)
print("cudagraph mem-profile skip patch applied")
PY
# Also promote the per-component memory breakdown to INFO (diagnosis aid).
python3 - <<'PY'
from pathlib import Path
import py_compile
P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")
s = P.read_text()
old = "        logger.debug(profile_result)\n"
if old in s:
    s = s.replace(old, "        logger.info(profile_result)\n", 1)
    P.write_text(s)
    py_compile.compile(str(P), doraise=True)
print("profile breakdown promoted to INFO")
PY
