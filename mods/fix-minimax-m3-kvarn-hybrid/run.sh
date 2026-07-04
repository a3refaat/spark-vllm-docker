#!/bin/bash
set -euo pipefail

# MiniMax-M3 × KVarN hybrid (Phase 1 — see docs/b12x-kvarn-integration-plan.md).
#
# With --kv-cache-dtype kvarn_k4v2_g128 the standard Attention layers (3 dense
# M3 layers + any spec-decode draft) route to the KVarN backend natively (mod:
# add-kvarn-kv-quant). The 57 sparse MSA layers have their own backend/impl and
# do NOT understand KVarN tiles yet (that is Phase 2): under a global kvarn
# dtype they must degrade to plain bf16 ("auto").
#
# One strategic anchor does the whole job: MiniMaxM3SparseAttention.__init__
# derives EVERYTHING from self.kv_cache_dtype — kv_cache_torch_dtype (spec
# bytes), select_main_impl_cls (Triton vs MSA), the impl's fp8 view, the spec's
# kv_quant_mode, and the dtype string passed to reshape_and_cache_flash /
# the fused qknorm-rope op. Mapping kvarn_* -> "auto" at that single point
# makes the sparse path behave exactly like the validated bf16-KV config.
#
# The indexer side cache is hardcoded bf16 (indexer.py: "self.dtype =
# torch.bfloat16") and keyed off attention_config.indexer_kv_dtype, not the
# main kv-cache dtype — no change needed.
#
# Apply AFTER mods/add-kvarn-kv-quant (needs the kvarn CacheDType strings).

PYTHON=${PYTHON:-python3}
$PYTHON - <<'PY'
import py_compile
from pathlib import Path

pkg = Path("/usr/local/lib/python3.12/dist-packages")

# ── 1) sparse layer: global kvarn dtype degrades to bf16 ("auto") ────────────
p = pkg / "vllm/models/minimax_m3/nvidia/model.py"
t = p.read_text()
if "kvarn dtype is Phase-2" in t:
    print("kvarn-hybrid already applied:", p)
    raise SystemExit(0)

old = (
    '        self.kv_cache_dtype = (\n'
    '            cache_config.cache_dtype if cache_config is not None else "auto"\n'
    '        )\n'
)
new = (
    '        self.kv_cache_dtype = (\n'
    '            cache_config.cache_dtype if cache_config is not None else "auto"\n'
    '        )\n'
    '        # KVarN: the sparse-MSA backend reading kvarn tiles is Phase-2;\n'
    '        # until then a global kvarn dtype means "quantize the standard\n'
    '        # Attention layers (dense + draft), sparse stays bf16". Mapping to\n'
    '        # "auto" here fixes the torch dtype, impl selection, spec quant\n'
    '        # mode, and the cache-op dtype strings in one place.\n'
    '        # (mod: fix-minimax-m3-kvarn-hybrid; kvarn dtype is Phase-2)\n'
    '        if str(self.kv_cache_dtype).startswith("kvarn"):\n'
    '            self.kv_cache_dtype = "auto"\n'
)
assert t.count(old) == 1, "sparse kv_cache_dtype anchor not found"
t = t.replace(old, new, 1)
p.write_text(t)
py_compile.compile(str(p), doraise=True)
print("Patched", p)

# ── 2) sparse backend ClassVar: accept the global kvarn strings (defensive;
#       any engine-level dtype-support validation checks this list) ──────────
p = pkg / "vllm/models/minimax_m3/common/sparse_attention.py"
t = p.read_text()
anchor = (
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "bfloat16",\n'
)
assert t.count(anchor) == 1, "sparse ClassVar anchor not found"
t = t.replace(
    anchor,
    anchor
    + '        "kvarn_k4v2_g128",\n'
    + '        "kvarn_k4v4_g128",\n'
    + '        "kvarn_k4v2_g64",\n'
    + '        "kvarn_k4v4_g64",\n',
    1,
)
p.write_text(t)
py_compile.compile(str(p), doraise=True)
print("Patched", p)
print("MiniMax-M3 kvarn-hybrid (Phase 1: dense+draft kvarn, sparse bf16) applied.")
PY
