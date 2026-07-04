#!/bin/bash
set -euo pipefail

# KVarN KV-cache quantization backend (vllm-project/vllm PR #46812 backport).
#
# KVarN (huawei-csl, arXiv:2606.03458): calibration-free 4-bit-K / 2-bit-V
# KV-cache quantization — per 128-token tile: Hadamard rotation + log-domain
# Sinkhorn variance normalization + asymmetric RTN, with an fp16 sink/tail
# pool. Selected via --kv-cache-dtype kvarn_k4v2_g128 (block_size must equal
# the g<N> group). Dense/GQA full-attention layers only; MiniMax-M3 sparse MSA
# layers keep their own spec/backend (see docs/b12x-kvarn-integration-plan.md).
#
# The PR is pure Python (Triton JIT kernels) — no wheel rebuild. 16 of the 19
# code files apply verbatim (kvarn_pr46812_clean.diff, applied with patch);
# the 3 below drifted vs this image's base (g979b56a66, ~2 weeks older than
# the PR base) and are ported as anchor edits:
#   * config/cache.py            — image lacks "int4_per_token_head" context
#   * layers/attention/attention.py — image lacks `cast` in the typing import
#   * v1/core/kv_cache_utils.py  — older unify_kv_cache_spec_page_size shape
#
# Diff snapshot pinned at PR state of 2026-07-03; refresh deliberately.

PYTHON=${PYTHON:-python3}
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOD_DIR
$PYTHON - <<'PY'
import os
import py_compile
import subprocess
from pathlib import Path

pkg = Path("/usr/local/lib/python3.12/dist-packages")
mod_dir = Path(os.environ["MOD_DIR"])

# ── idempotence guard ────────────────────────────────────────────────────────
kvarn_cfg = pkg / "vllm/model_executor/layers/quantization/kvarn/config.py"
if kvarn_cfg.exists():
    print("KVarN backport already applied:", kvarn_cfg)
    raise SystemExit(0)

# ── 1) 16 clean files via patch ─────────────────────────────────────────────
diff = mod_dir / "kvarn_pr46812_clean.diff"
subprocess.run(
    ["patch", "-p1", "-f", "--no-backup-if-mismatch", "-i", str(diff)],
    cwd=pkg, check=True,
)
print("Applied kvarn_pr46812_clean.diff (16 files)")

# ── 2) config/cache.py: register the kvarn cache dtypes ─────────────────────
p = pkg / "vllm/config/cache.py"
t = p.read_text()
anchor = '    "turboquant_3bit_nc",\n'
assert t.count(anchor) == 1, "cache.py CacheDType anchor not found"
t = t.replace(
    anchor,
    anchor
    + '    "kvarn_k4v2_g128",\n'
    + '    "kvarn_k4v4_g128",\n'
    + '    "kvarn_k4v2_g64",\n'
    + '    "kvarn_k4v4_g64",\n',
    1,
)
p.write_text(t)
print("Patched", p)

# ── 3) attention.py: PR hunk #1 (import contextlib); hunks 2-4 via patch ────
p = pkg / "vllm/model_executor/layers/attention/attention.py"
t = p.read_text()
anchor = "from typing import TYPE_CHECKING, Any\n"
assert t.count(anchor) == 1, "attention.py import anchor not found"
t = t.replace(anchor, "import contextlib\nfrom typing import TYPE_CHECKING, Any\n", 1)
p.write_text(t)
# apply the remaining hunks of the attention.py chunk from the full diff
attn_diff = mod_dir / "kvarn_attention_py.diff"
subprocess.run(
    ["patch", "-p1", "-f", "--no-backup-if-mismatch", "-i", str(attn_diff)],
    cwd=pkg, check=True,
)
print("Patched", p)

# ── 4) kv_cache_utils.py: group-locked specs pad the page, never scale ──────
p = pkg / "vllm/v1/core/kv_cache_utils.py"
t = p.read_text()
old = (
    "            ratio = max_page_size // layer_page_size\n"
    "            new_block_size = layer_spec.block_size * ratio\n"
    "            new_spec = replace(layer_spec, block_size=new_block_size)\n"
    "            assert new_spec.page_size_bytes == max_page_size\n"
)
new = (
    "            # KVarN/TQ specs are group-locked: block_size must equal the\n"
    "            # variance-normalization tile size, so block_size cannot be\n"
    "            # scaled to grow the page. Pad the page instead (strided view,\n"
    "            # like MLA), keeping block_size fixed. (mod: add-kvarn-kv-quant)\n"
    "            if getattr(layer_spec, \"tq_slot_size\", 0) > 0:\n"
    "                new_spec = replace(  # type: ignore[call-arg]\n"
    "                    layer_spec, page_size_padded=max_page_size\n"
    "                )\n"
    "            else:\n"
    "                ratio = max_page_size // layer_page_size\n"
    "                new_block_size = layer_spec.block_size * ratio\n"
    "                new_spec = replace(layer_spec, block_size=new_block_size)\n"
    "            assert new_spec.page_size_bytes == max_page_size\n"
)
assert t.count(old) == 1, "kv_cache_utils.py unify anchor not found"
t = t.replace(old, new, 1)
p.write_text(t)
print("Patched", p)

# ── compile gate ─────────────────────────────────────────────────────────────
for rel in [
    "vllm/config/cache.py",
    "vllm/model_executor/layers/attention/attention.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/attention/backends/kvarn_attn.py",
    "vllm/v1/attention/ops/triton_kvarn_decode.py",
    "vllm/model_executor/layers/quantization/kvarn/config.py",
]:
    py_compile.compile(str(pkg / rel), doraise=True)
print("KVarN backport applied + compiled.")
PY
