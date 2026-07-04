#!/bin/bash
set -euo pipefail
# MiniMax-M3 b12x dense full-attention backend (SM120/SM121 / GB10).
#
# Registers a proper vLLM v1 AttentionBackend under `--attention-backend b12x`
# (enum member B12X -> b12x_backend.B12XAttentionBackend) and moves the dense
# M3 layers (0-2) onto b12x's dense paged path (msa_block_sparse=False). No
# Triton: with `--attention-backend b12x` the dense layers select B12X, and the
# sparse layers select the b12x MSA impl (mod: fix-minimax-m3-b12x-msa).
#
#   * installs a vLLM-local b12x_backend.py shim that re-exports the source of
#     truth in ../b12x/b12x/vllm/minimax_m3/backend.py
#   * registry.py: add `B12X = "...b12x_backend.B12XAttentionBackend"` enum
#     member so `AttentionBackendEnum["B12X"]` (i.e. `--attention-backend b12x`)
#     resolves -- same model-driven path pattern as MINIMAX_M3_SPARSE.
#
# Apply AFTER mods/fix-minimax-m3-b12x-msa (reuses its sparse metadata/builder
# and the shared fp8 KV contract). Recipe must pass `--attention-backend b12x`.

PYTHON=${PYTHON:-python3}
$PYTHON - <<'PY'
import importlib.util, py_compile, shutil
from pathlib import Path

assert importlib.util.find_spec("b12x") is not None, "b12x not installed in image"
assert importlib.util.find_spec("b12x.vllm.minimax_m3.backend") is not None, (
    "b12x MiniMax-M3 vLLM dense backend glue missing from image; rebuild "
    "vllm-node-minimax-m3-b12x from Dockerfile.b12x"
)
vroot = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
common = vroot / "models/minimax_m3/common"
REGISTRY = vroot / "v1/attention/backends/registry.py"
assert common.exists() and REGISTRY.exists()

# 1) install a tiny vLLM-local compatibility shim. The implementation source of
# truth lives in ../b12x/b12x/vllm/minimax_m3/backend.py.
dst = common / "b12x_backend.py"
dst.write_text("from b12x.vllm.minimax_m3.backend import *  # noqa: F401,F403\n")
py_compile.compile(str(dst), doraise=True)
print(f"Installed shim {dst} -> b12x.vllm.minimax_m3.backend")

# 2) registry: add the B12X enum member (model-driven path, like MINIMAX_M3_SPARSE)
r = REGISTRY.read_text()
anchor = (
    "    MINIMAX_M3_SPARSE = (\n"
    "        \"vllm.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseBackend\"\n"
    "    )\n"
)
assert r.count(anchor) == 1, f"registry MINIMAX_M3_SPARSE anchor count={r.count(anchor)}"
if "b12x_backend.B12XAttentionBackend" not in r:
    r = r.replace(
        anchor,
        anchor
        + "    B12X = (\n"
        "        \"vllm.models.minimax_m3.common.b12x_backend.B12XAttentionBackend\"\n"
        "    )\n",
        1,
    )
    REGISTRY.write_text(r)
py_compile.compile(str(REGISTRY), doraise=True)
assert "B12X = (" in REGISTRY.read_text()
print("Patched registry.py: added AttentionBackendEnum.B12X")

for d in (common / "__pycache__", REGISTRY.parent / "__pycache__"):
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
print("MiniMax-M3 b12x dense backend mod applied.")
PY
