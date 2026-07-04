#!/bin/bash
set -euo pipefail

# MiniMax-M3 compressed-tensors MoE SWIGLU clamp fix.
#
# MiniMax-M3 uses the swigluoai_uninterleave activation, which requires
# clamp_limit/alpha/beta during vLLM's profile_run. RoutedExperts already stores
# these as layer.swiglu_limit / layer.swiglu_alpha / layer.swiglu_beta, but the
# compressed-tensors WNA16 Marlin MoE quant config builder does not copy them
# into FusedMoEQuantConfig. Without this, startup fails with:
#   AssertionError: SWIGLUOAI_UNINTERLEAVE requires clamp_limit

PYTHON=${PYTHON:-python3}
$PYTHON - <<'PY'
from pathlib import Path
import importlib.util, py_compile, shutil

spec = importlib.util.find_spec("vllm")
if spec is None or spec.submodule_search_locations is None:
    raise SystemExit("vLLM is not importable")
vroot = Path(spec.submodule_search_locations[0])
p = vroot / "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py"
assert p.exists(), f"missing {p}"
t = p.read_text()

if "MiniMax-M3 swigluoai_uninterleave clamp fields" in t:
    print("MiniMax-M3 CT WNA16 MoE swiglu clamp already patched:", p)
    raise SystemExit(0)

old = '''    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return make_wna16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            group_size=self.group_size,
            num_bits=self.num_bits,
            w1_zp=getattr(layer, "w13_weight_zero_point", None),
            w2_zp=getattr(layer, "w2_weight_zero_point", None),
        )
'''
new = '''    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        moe_quant_config = make_wna16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            group_size=self.group_size,
            num_bits=self.num_bits,
            w1_zp=getattr(layer, "w13_weight_zero_point", None),
            w2_zp=getattr(layer, "w2_weight_zero_point", None),
        )
        # MiniMax-M3 swigluoai_uninterleave clamp fields.  RoutedExperts has
        # these values, but the compressed-tensors MoE config otherwise drops
        # them, causing profile_run to assert on clamp_limit=None.
        moe_quant_config.gemm1_clamp_limit = getattr(layer, "swiglu_limit", None)
        moe_quant_config.gemm1_alpha = getattr(layer, "swiglu_alpha", None)
        moe_quant_config.gemm1_beta = getattr(layer, "swiglu_beta", None)
        return moe_quant_config
'''
if old not in t:
    raise SystemExit("anchor not found in compressed_tensors_moe_wna16_marlin.py")
t = t.replace(old, new, 1)
p.write_text(t)
py_compile.compile(str(p), doraise=True)
cache = p.parent / "__pycache__"
if cache.exists():
    shutil.rmtree(cache, ignore_errors=True)
print("Patched MiniMax-M3 compressed-tensors WNA16 MoE swiglu clamp:", p)
PY
