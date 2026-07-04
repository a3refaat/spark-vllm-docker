#!/bin/bash
set -euo pipefail
# MiniMax-M3 b12x CuTe MSA main attention (SM120/SM121 / GB10) -- self-contained.
#
# Owns the b12x main-attention integration end to end (no dependency on the
# legacy fp8-kv / indexer-fp8 mods):
#   * model.py __init__: fp8 KV per-tensor scale buffers (_k_scale/_v_scale;
#     VLLM_M3_FP8_KV_TEST_SCALE, default 1.0). b12x reads the main fp8 cache with
#     these as descales; the write divides by them -> exact round trip.
#   * model.py forward: on the fp8 path run the fused qknorm+rope op in
#     norm+rope-ONLY mode (None caches) and route the writes through _insert_kv.
#   * model.py _insert_kv: write the main fp8 KV via reshape_and_cache_flash
#     (scaled), and the single-head index_k into the packed page-64 b12x side
#     cache (write_packed_index_cache, from the b12x-indexer mod).
#   * sparse_attention.py select_main_impl_cls: pick the b12x MSA impl on
#     SM120/121 for fp8/bf16/nvfp4 KV.
#   * installs a vLLM-local b12x_msa_attn.py shim that re-exports the source of
#     truth in ../b12x/b12x/vllm/minimax_m3/msa_attn.py.
#
# Apply AFTER mods/fix-minimax-m3-b12x-indexer (which installs the b12x_indexer.py
# shim and the packed indexer cache spec).

PYTHON=${PYTHON:-python3}
$PYTHON - <<'PY'
import importlib.util, py_compile, shutil
from pathlib import Path

assert importlib.util.find_spec("b12x") is not None, "b12x not installed in image"
assert importlib.util.find_spec("b12x.vllm.minimax_m3.msa_attn") is not None, (
    "b12x MiniMax-M3 vLLM MSA glue missing from image; rebuild "
    "vllm-node-minimax-m3-b12x from Dockerfile.b12x"
)
vroot = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
common = vroot / "models/minimax_m3/common"
IMPL = common / "sparse_attention.py"
MODEL = vroot / "models/minimax_m3/nvidia/model.py"
assert IMPL.exists() and MODEL.exists()

# 1) install a tiny vLLM-local compatibility shim. The implementation source of
# truth lives in ../b12x/b12x/vllm/minimax_m3/msa_attn.py.
dst = common / "b12x_msa_attn.py"
dst.write_text("from b12x.vllm.minimax_m3.msa_attn import *  # noqa: F401,F403\n")
py_compile.compile(str(dst), doraise=True)
print(f"Installed shim {dst} -> b12x.vllm.minimax_m3.msa_attn")

# 2) selector: b12x MSA on SM120/121 for fp8/bf16 KV (nvfp4 excluded).
t = IMPL.read_text()
anchor = (
    "        return MiniMaxM3SparseMSAImpl\n"
    "    return MiniMaxM3SparseTritonImpl\n"
)
assert t.count(anchor) == 1, f"selector anchor count={t.count(anchor)}"
t = t.replace(
    anchor,
    "        return MiniMaxM3SparseMSAImpl\n"
    "    if (\n"
    "        current_platform.is_cuda()\n"
    "        and current_platform.is_device_capability_family(120)\n"
    "        and topk_blocks in (4, 8, 16, 32)\n"
    "    ):\n"
    "        from vllm.models.minimax_m3.common.b12x_msa_attn import (\n"
    "            MiniMaxM3SparseB12xImpl,\n"
    "        )\n"
    "        return MiniMaxM3SparseB12xImpl\n"
    "    return MiniMaxM3SparseTritonImpl\n",
    1,
)

# 2a) nvfp4 main KV: advertise the dtype, pack the cache last dim (hd//2 +
# hd//16), and keep the uint8 cache (b12x reads the per-16 e4m3 block scales
# from the cache itself -> use_fp8_kv must be False so it is NOT viewed as fp8).
t = t.replace(
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "bfloat16",\n        "fp8",\n        "fp8_e4m3",\n        "fp8_e5m2",\n    ]\n',
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "bfloat16",\n        "fp8",\n        "fp8_e4m3",\n        "fp8_e5m2",\n'
    '        "nvfp4",\n    ]\n',
    1,
)
t = t.replace(
    '        cache_dtype_str: str = "auto",\n'
    "    ) -> tuple[int, ...]:\n"
    "        return (num_blocks, 2, block_size, num_kv_heads, head_size)\n",
    '        cache_dtype_str: str = "auto",\n'
    "    ) -> tuple[int, ...]:\n"
    '        if cache_dtype_str == "nvfp4":\n'
    "            from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim\n"
    "            head_size = nvfp4_kv_cache_full_dim(head_size)\n"
    "        return (num_blocks, 2, block_size, num_kv_heads, head_size)\n",
    1,
)
t = t.replace(
    "        self.use_fp8_kv = is_quantized_kv_cache(kv_cache_dtype)\n",
    "        self.use_fp8_kv = (\n"
    '            is_quantized_kv_cache(kv_cache_dtype) and kv_cache_dtype != "nvfp4"\n'
    "        )\n",
    1,
)
IMPL.write_text(t)
py_compile.compile(str(IMPL), doraise=True)
print("Patched select_main_impl_cls + nvfp4 dtype/shape/uint8-keep")

# 3) model.py: fp8 scales, norm+rope-only fused bypass, _insert_kv writes.
m = MODEL.read_text()

# 3a) __init__ fp8 scale buffers.
init_old = (
    "        compilation_config.static_forward_context[self.layer_name] = self\n"
    "        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache\n"
)
assert m.count(init_old) == 1, "model __init__ anchor"
m = m.replace(
    init_old,
    "        compilation_config.static_forward_context[self.layer_name] = self\n"
    "        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache\n"
    "        # b12x fp8 KV: per-tensor scale used as BOTH the write divisor\n"
    "        # (reshape_and_cache_flash) and the b12x read descale. Default 1.0\n"
    "        # (== unscaled e4m3). (mod: b12x-msa)\n"
    "        self._b12x_fp8 = \"fp8\" in self.kv_cache_dtype\n"
    "        self._b12x_nvfp4 = self.kv_cache_dtype == \"nvfp4\"\n"
    "        # any quantized KV (fp8 or nvfp4) routes the writes through\n"
    "        # _insert_kv (the fused op can't scale/pack the cache).\n"
    "        self._b12x_quant = self._b12x_fp8 or self._b12x_nvfp4\n"
    "        # The b12x packed page-64 index cache can NEVER be written by the\n"
    "        # fused op's insert mode (dtype/layout mismatch) -> ALWAYS run the\n"
    "        # fused op norm+rope-only and write both caches via _insert_kv\n"
    "        # (bf16/auto main KV writes natively via reshape_and_cache_flash).\n"
    "        self._b12x_route_insert = True\n"
    "        import os as _os\n"
    "        _b12x_s = float(_os.environ.get(\"VLLM_M3_FP8_KV_TEST_SCALE\", \"1.0\"))\n"
    "        self.register_buffer(\n"
    "            \"_k_scale\", torch.full((), _b12x_s, dtype=torch.float32), persistent=False\n"
    "        )\n"
    "        self.register_buffer(\n"
    "            \"_v_scale\", torch.full((), _b12x_s, dtype=torch.float32), persistent=False\n"
    "        )\n",
    1,
)

# 3b) forward: fused op -> norm+rope only on fp8; extract k/v/index_k; _insert_kv.
fwd_old = (
    "            self.kv_cache,\n"
    "            self.indexer.index_cache.kv_cache,\n"
    "            self.kv_cache.size(2),  # paged-cache block size\n"
    "            q,\n"
    "            index_q,\n"
    "            self.kv_cache_dtype,\n"
    "        )\n"
    "\n"
    "        output = torch.empty_like(q)\n"
)
assert m.count(fwd_old) == 1, "model forward fused-call anchor"
m = m.replace(
    fwd_old,
    "            # b12x quant path: fused op can't apply a KV scale / pack nvfp4\n"
    "            # and can't write the packed index cache, so run it norm+rope-\n"
    "            # ONLY (None caches) and write both caches via _insert_kv.\n"
    "            self.kv_cache if not self._b12x_route_insert else None,\n"
    "            self.indexer.index_cache.kv_cache if not self._b12x_route_insert else None,\n"
    "            self.kv_cache.size(2),  # paged-cache block size\n"
    "            q,\n"
    "            index_q,\n"
    "            self.kv_cache_dtype,\n"
    "        )\n"
    "        if self._b12x_route_insert:\n"
    "            _kv = self.num_kv_heads * self.head_dim\n"
    "            _k = qkv[:, self.q_size : self.q_size + _kv]\n"
    "            _v = qkv[:, self.q_size + _kv : self.q_size + 2 * _kv]\n"
    "            # index_k is a SINGLE shared head (idx_head_dim wide).\n"
    "            _ik0 = self.q_size + 2 * _kv + self.index_q_size\n"
    "            _index_k = qkv[:, _ik0 : _ik0 + self.idx_head_dim]\n"
    "            self._insert_kv(_k, _v, _index_k)\n"
    "\n"
    "        output = torch.empty_like(q)\n",
    1,
)

# 3c) _insert_kv: scaled main fp8 write + packed index-K write.
ins_old = (
    "        # Identity scale: unused for the bf16 cache, required arg of the op.\n"
    "        key_cache, value_cache = self.kv_cache.unbind(1)\n"
    "        scale = torch.ones((), device=key.device)\n"
    "        ops.reshape_and_cache_flash(\n"
    "            key.view(-1, self.num_kv_heads, self.head_dim),\n"
    "            value.view(-1, self.num_kv_heads, self.head_dim),\n"
    "            key_cache,\n"
    "            value_cache,\n"
    "            main_meta.slot_mapping,\n"
    "            self.kv_cache_dtype,\n"
    "            scale,\n"
    "            scale,\n"
    "        )\n"
    "\n"
    "        # Index-key cache: single vector per token, scatter by slot.\n"
    "        idx_cache = self.indexer.index_cache.kv_cache.view(-1, self.idx_head_dim)\n"
    "        idx_cache[index_meta.slot_mapping] = index_key.to(idx_cache.dtype)\n"
)
assert m.count(ins_old) == 1, "model _insert_kv anchor"
m = m.replace(
    ins_old,
    "        # Main KV write. fp8: reshape_and_cache_flash DIVIDES by the\n"
    "        # per-tensor scale before the e4m3 cast (b12x reads the same\n"
    "        # descale). nvfp4: native packed block-quant write (b12x reads the\n"
    "        # per-16 e4m3 block scales from the cache). (mod: b12x-msa)\n"
    "        key_cache, value_cache = self.kv_cache.unbind(1)\n"
    "        if self._b12x_nvfp4:\n"
    "            from vllm.models.minimax_m3.common.b12x_backend import (\n"
    "                nvfp4_block_quant_write,\n"
    "            )\n"
    "            nvfp4_block_quant_write(\n"
    "                key, key_cache, main_meta.slot_mapping,\n"
    "                self.num_kv_heads, self.head_dim)\n"
    "            nvfp4_block_quant_write(\n"
    "                value, value_cache, main_meta.slot_mapping,\n"
    "                self.num_kv_heads, self.head_dim)\n"
    "        else:\n"
    "            k_scale = self._k_scale.to(key.device)\n"
    "            v_scale = self._v_scale.to(key.device)\n"
    "            ops.reshape_and_cache_flash(\n"
    "                key.view(-1, self.num_kv_heads, self.head_dim),\n"
    "                value.view(-1, self.num_kv_heads, self.head_dim),\n"
    "                key_cache,\n"
    "                value_cache,\n"
    "                main_meta.slot_mapping,\n"
    "                self.kv_cache_dtype,\n"
    "                k_scale,\n"
    "                v_scale,\n"
    "            )\n"
    "        # Index-K side cache: packed page-64 fp8 + per-token scale (b12x)\n"
    "        # when uint8; else the plain bf16 scatter. (mod: b12x-msa)\n"
    "        idx_cache = self.indexer.index_cache.kv_cache\n"
    "        if idx_cache.dtype == torch.uint8:\n"
    "            from vllm.models.minimax_m3.common.b12x_indexer import (\n"
    "                write_packed_index_cache, write_packed_index_cache_nvfp4,\n"
    "            )\n"
    "            if idx_cache.shape[-1] == self.idx_head_dim // 2 + self.idx_head_dim // 16:\n"
    "                write_packed_index_cache_nvfp4(idx_cache, index_key, index_meta.slot_mapping)\n"
    "            else:\n"
    "                write_packed_index_cache(idx_cache, index_key, index_meta.slot_mapping)\n"
    "        else:\n"
    "            idx_cache.view(-1, self.idx_head_dim)[index_meta.slot_mapping] = (\n"
    "                index_key.to(idx_cache.dtype)\n"
    "            )\n"
)

MODEL.write_text(m)
py_compile.compile(str(MODEL), doraise=True)
for d in (common / "__pycache__", MODEL.parent / "__pycache__"):
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
print("Patched nvidia/model.py: fp8 scales + norm+rope-only fused bypass + writes")
print("MiniMax-M3 b12x MSA attention mod applied.")
PY
