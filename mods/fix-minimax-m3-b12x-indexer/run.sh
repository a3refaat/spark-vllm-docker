#!/bin/bash
set -euo pipefail
# MiniMax-M3 b12x CuTe MSA lightning indexer (SM120/SM121 / GB10).
#
# Self-contained b12x integration (does NOT depend on the legacy fp8-kv /
# indexer-fp8 mods). Replaces the Triton index score+top-k with b12x's MSA
# indexer kernels (paged decode + contiguous prefill) over a PACKED page-64
# fp8+per-token-scale side cache (b12x's required byte layout), reusing the SAME
# indexer KV-cache group (block_size unchanged; only dtype/width + read/write
# change). q2k output ([num_kv_heads, q_rows, 16], ascending, -1 padded at end)
# is consumed directly by the b12x MSA attention (decode + extend).
#
# This mod patches indexer.py ONLY (cache spec + impl selector) and installs a
# vLLM-local shim to the implementation source of truth in
# ../b12x/b12x/vllm/minimax_m3/indexer.py. The model.py forward / _insert_kv
# (packed index write + main fp8 KV write + norm+rope-only fused bypass) is
# owned by the companion mod fix-minimax-m3-b12x-msa, applied AFTER this one.

PYTHON=${PYTHON:-python3}
$PYTHON - <<'PY'
import importlib.util, py_compile, shutil
from pathlib import Path

assert importlib.util.find_spec("b12x") is not None, "b12x not installed in image"
assert importlib.util.find_spec("b12x.vllm.minimax_m3.indexer") is not None, (
    "b12x MiniMax-M3 vLLM indexer glue missing from image; rebuild "
    "vllm-node-minimax-m3-b12x from Dockerfile.b12x"
)
vroot = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
common = vroot / "models/minimax_m3/common"
IDX = common / "indexer.py"
assert IDX.exists()

# Install a tiny vLLM-local compatibility shim. The implementation source of
# truth lives in ../b12x/b12x/vllm/minimax_m3/indexer.py.
dst = common / "b12x_indexer.py"
dst.write_text("from b12x.vllm.minimax_m3.indexer import *  # noqa: F401,F403\n")
py_compile.compile(str(dst), doraise=True)
print(f"Installed shim {dst} -> b12x.vllm.minimax_m3.indexer")

t = IDX.read_text()

# 1) unlock fp8 indexer-kv-dtype -> PACKED uint8 side cache (132 bytes/token:
#    128 fp8 K + 4 fp32 scale). The page-major page-64 layout is produced by the
#    packed write; the spec only needs the right byte budget + uint8 dtype.
gate = (
    '        if indexer_kv_dtype != "bf16":\n'
    "            raise NotImplementedError(\n"
    '                f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported yet "\n'
    "                \"for the MiniMax M3 indexer cache (only 'bf16').\"\n"
    "            )\n"
    "        self.kv_cache = torch.tensor([])\n"
    "        self.head_dim = head_dim\n"
    "        self.indexer_kv_dtype = indexer_kv_dtype\n"
    "        # Storage dtype for the side cache (bf16 today; quantized layouts later).\n"
    "        self.dtype = torch.bfloat16\n"
)
assert t.count(gate) == 1, f"indexer cache dtype gate anchor count={t.count(gate)}"
t = t.replace(
    gate,
    '        if indexer_kv_dtype not in ("bf16", "fp8", "nvfp4"):\n'
    "            raise NotImplementedError(\n"
    '                f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported "\n'
    "                \"for the MiniMax M3 indexer cache (only 'bf16', 'fp8', 'nvfp4').\"\n"
    "            )\n"
    "        self.kv_cache = torch.tensor([])\n"
    "        self.head_dim = head_dim\n"
    "        self.indexer_kv_dtype = indexer_kv_dtype\n"
    "        # b12x packed page-64 side cache (uint8 storage). fp8: head_dim(128\n"
    "        # fp8 K) + 4 (fp32 per-token scale) = 132 B/tok. nvfp4 (S5): per-16\n"
    "        # e2m1 + e4m3 = head_dim//2 + head_dim//16 = 72 B/tok. (mod: b12x-indexer)\n"
    "        self._b12x_packed = indexer_kv_dtype in (\"fp8\", \"nvfp4\")\n"
    "        self._packed_head_bytes = (\n"
    "            head_dim // 2 + head_dim // 16 if indexer_kv_dtype == \"nvfp4\"\n"
    "            else head_dim + 4\n"
    "        )\n"
    "        self.dtype = torch.uint8 if self._b12x_packed else torch.bfloat16\n",
    1,
)

# 2) packed cache spec: uint8, head_size = packed bytes (132).
spec_old = (
    "        return MLAAttentionSpec(\n"
    "            block_size=vllm_config.cache_config.block_size,\n"
    "            num_kv_heads=1,\n"
    "            head_size=self.head_dim,\n"
    "            dtype=self.dtype,\n"
    "        )\n"
)
assert t.count(spec_old) == 1, "indexer get_kv_cache_spec anchor"
t = t.replace(
    spec_old,
    "        return MLAAttentionSpec(\n"
    "            block_size=vllm_config.cache_config.block_size,\n"
    "            num_kv_heads=1,\n"
    "            head_size=(self._packed_head_bytes if getattr(self, \"_b12x_packed\", False) else self.head_dim),\n"
    "            dtype=self.dtype,\n"
    "        )\n",
    1,
)

# 3) backend get_supported_head_sizes: allow the packed 132-byte head.
heads_old = (
    "    @classmethod\n"
    "    def get_supported_head_sizes(cls) -> list[int]:\n"
    "        return [128]\n"
)
assert t.count(heads_old) >= 1, "indexer get_supported_head_sizes anchor"
t = t.replace(
    heads_old,
    "    @classmethod\n"
    "    def get_supported_head_sizes(cls) -> list[int]:\n"
    "        return [128, 132, 72]\n",
    1,
)

# 3b) drop the nvfp4 "not-yet-added" guard in select_indexer_impl_cls (S5 ships
#     the b12x CuteDSL indexer impl). Keep mxfp4 blocked.
nv_guard = (
    '    if indexer_kv_dtype in ("mxfp4", "nvfp4"):\n'
    "        raise NotImplementedError(\n"
    '            f"indexer_kv_dtype={indexer_kv_dtype!r} needs the (not-yet-added) "\n'
    '            "CuteDSL indexer impl."\n'
    "        )\n"
)
assert t.count(nv_guard) == 1, "nvfp4 not-yet-added guard anchor"
t = t.replace(
    nv_guard,
    '    if indexer_kv_dtype == "mxfp4":\n'
    "        raise NotImplementedError(\n"
    '            f"indexer_kv_dtype={indexer_kv_dtype!r} needs the (not-yet-added) "\n'
    '            "CuteDSL indexer impl."\n'
    "        )\n",
    1,
)

# 4) impl selector: b12x MSA indexer on SM120/121 for the fp8/nvfp4 packed cache.
sel_old = (
    "    if indexer_kv_dtype != \"bf16\":\n"
    "        raise NotImplementedError(\n"
    "            f\"indexer_kv_dtype={indexer_kv_dtype!r} is not supported by the \"\n"
    "            \"Triton indexer impl.\"\n"
    "        )\n"
    "    return MiniMaxM3IndexerTritonImpl\n"
)
assert t.count(sel_old) == 1, "select_indexer_impl_cls anchor"
t = t.replace(
    sel_old,
    "    if indexer_kv_dtype in (\"fp8\", \"nvfp4\"):\n"
    "        from vllm.platforms import current_platform as _cp\n"
    "        if _cp.is_cuda() and _cp.is_device_capability_family(120):\n"
    "            from vllm.models.minimax_m3.common.b12x_indexer import (\n"
    "                MiniMaxM3IndexerB12xImpl,\n"
    "            )\n"
    "            return MiniMaxM3IndexerB12xImpl\n"
    "        raise NotImplementedError(\n"
    "            \"fp8 b12x indexer requires SM120/121 (GB10)\"\n"
    "        )\n"
    "    if indexer_kv_dtype != \"bf16\":\n"
    "        raise NotImplementedError(\n"
    "            f\"indexer_kv_dtype={indexer_kv_dtype!r} is not supported by the \"\n"
    "            \"Triton indexer impl.\"\n"
    "        )\n"
    "    return MiniMaxM3IndexerTritonImpl\n",
    1,
)

IDX.write_text(t)
py_compile.compile(str(IDX), doraise=True)
for d in (common / "__pycache__",):
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
print("Patched indexer.py: packed fp8 cache spec + b12x impl selector.")
print("MiniMax-M3 b12x MSA indexer mod applied.")
PY
