#!/bin/bash
set -euo pipefail

# MiniMax-M3 sparse MSA layers on KVarN tiles (Phase 2A).
#
# Replaces mods/fix-minimax-m3-kvarn-hybrid (which degraded sparse layers to
# bf16). The stock block-sparse Triton attend kernels stay UNCHANGED:
#   * store  : rotate K/V by H -> fp16 pool scatter (graph-safe); Sinkhorn+RTN
#              flush of full tiles runs in the metadata builder between graph
#              replays (kvarn_sparse.KVarNSparseGroup.builder_step).
#   * spec   : TQFullAttentionSpec, tq_slot_size = tile_bytes/128 (=108 B/token
#              /head for k4v2_g128); cache = uint8 [num_blocks, Hk, 13824].
#   * read   : Triton ws kernel dequants the indexer-SELECTED tiles (and copies
#              pool-resident ones) into a bounded bf16 workspace shaped like
#              the stock paged cache + scatters a parallel block table.
#              topk indices / positions / causal masks untouched. q @ H in,
#              o @ H out (H orthonormal => scores exact).
# Offline gates (kvarn_sparse_test.py): tile round-trip K .995 / V .898 (iid
# worst case), pool copy exact, bt scatter exact, attend parity POOL-ONLY
# cos=0.999989 (machinery exact; remaining delta is quantization by design).
#
# SPEC DECODE (EAGLE3, Phase 3): flush is COMMIT-gated — builder_step packs a
# tile only once every position is below num_computed_tokens (accepted), never
# on scheduled fill: verify-step draft tokens can be REJECTED and rewritten in
# the fp16 pool next step, while packing to int4 is permanent (mirrors the
# dense kvarn_attn builder's committed-boundary rule). Verify rows (uniform
# decode_query_len > 1) flow through the stock spec-as-decode classification;
# the fused/ws decode kernels and ws sizing (q_len_max = 1 + num_spec) handle
# them natively. Offline gates (kvarn_spec_test.py): CLEAN-vs-SPEC packed
# records byte-identical under rejection, uncommitted tile stays pool-resident,
# fused decode q_len=4 row-exact, reclaim flushes only committed sinks.
#
# Apply AFTER mods/add-kvarn-kv-quant. Do NOT combine with kvarn-hybrid.

PYTHON=${PYTHON:-python3}
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOD_DIR
$PYTHON - <<'PY'
import os
import py_compile
import shutil
from pathlib import Path

pkg = Path("/usr/local/lib/python3.12/dist-packages")
mod_dir = Path(os.environ["MOD_DIR"])
common = pkg / "vllm/models/minimax_m3/common"
MODEL = pkg / "vllm/models/minimax_m3/nvidia/model.py"
SPARSE = common / "sparse_attention.py"

if (common / "kvarn_sparse.py").exists():
    print("kvarn-sparse already applied")
    raise SystemExit(0)

assert (pkg / "vllm/model_executor/layers/quantization/kvarn/config.py").exists(), (
    "apply mods/add-kvarn-kv-quant first")

# ── 0) install the module ────────────────────────────────────────────────────
shutil.copy(mod_dir / "kvarn_sparse.py", common / "kvarn_sparse.py")
py_compile.compile(str(common / "kvarn_sparse.py"), doraise=True)
print("Installed", common / "kvarn_sparse.py")

# ── 1) model.py: layer init — kvarn state (no dtype downgrade) ──────────────
m = MODEL.read_text()
init_old = (
    '        self.kv_cache_dtype = (\n'
    '            cache_config.cache_dtype if cache_config is not None else "auto"\n'
    '        )\n'
)
assert m.count(init_old) == 1, "model init anchor"
m = m.replace(
    init_old,
    init_old
    + '        # KVarN sparse tiles (mod: kvarn-sparse). is_quantized_kv_cache()\n'
    + '        # is False for kvarn_* so impl selection / fp8 views are\n'
    + '        # unaffected; the uint8 tile cache never reaches the attend\n'
    + '        # kernels (workspace substitution in the impl forward).\n'
    + '        self._kvarn_state = None\n'
    + '        self._kvarn_group = None\n'
    + '        if str(self.kv_cache_dtype).startswith("kvarn"):\n'
    + '            from vllm.models.minimax_m3.common import kvarn_sparse as _ks\n'
    + '            self._kvarn_group = _ks.get_group(vllm_config)\n'
    + '            self._kvarn_state = _ks.KVarNSparseLayer(\n'
    + '                self.layer_name, self.num_kv_heads, self.head_dim,\n'
    + '                str(self.kv_cache_dtype))\n',
    1,
)

# ── 2) model.py: spec — TQFullAttentionSpec with tile slot size ──────────────
spec_old = (
    '        return FullAttentionSpec(\n'
    '            block_size=vllm_config.cache_config.block_size,\n'
    '            num_kv_heads=self.num_kv_heads,\n'
    '            head_size=self.head_dim,\n'
    '            head_size_v=self.head_dim,\n'
    '            dtype=self.kv_cache_torch_dtype,\n'
    '            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n'
    '        )\n'
)
spec_new = (
    '        if self._kvarn_state is not None:\n'
    '            from vllm.v1.kv_cache_interface import TQFullAttentionSpec\n'
    '            cfg = self._kvarn_state.cfg\n'
    '            assert vllm_config.cache_config.block_size == cfg.group\n'
    '            return TQFullAttentionSpec(\n'
    '                block_size=cfg.group,\n'
    '                num_kv_heads=self.num_kv_heads,\n'
    '                head_size=self.head_dim,\n'
    '                head_size_v=self.head_dim,\n'
    '                dtype=torch.uint8,\n'
    '                tq_slot_size=cfg.tile_bytes_aligned // cfg.group,\n'
    '            )\n'
) + spec_old
assert m.count(spec_old) == 1, "model spec anchor"
m = m.replace(spec_old, spec_new, 1)

# ── 3) model.py: forward — norm+rope-only fused op under kvarn ───────────────
fwd_old = (
    '            self.kv_cache,\n'
    '            self.indexer.index_cache.kv_cache,\n'
    '            self.kv_cache.size(2),  # paged-cache block size\n'
)
assert m.count(fwd_old) == 1, "model forward fused-call anchor"
m = m.replace(
    fwd_old,
    '            # kvarn: fused op cannot pack tiles -> norm+rope ONLY (None\n'
    '            # caches); both caches written via _insert_kv. (mod: kvarn-sparse)\n'
    '            self.kv_cache if self._kvarn_state is None else None,\n'
    '            self.indexer.index_cache.kv_cache\n'
    '            if self._kvarn_state is None else None,\n'
    '            self.kv_cache.size(2),  # paged-cache block size (unused w/ None)\n',
    1,
)
fwd_out_old = '        output = torch.empty_like(q)\n'
assert m.count(fwd_out_old) == 1, "model forward output anchor"
m = m.replace(
    fwd_out_old,
    '        if self._kvarn_state is not None:\n'
    '            _kv = self.num_kv_heads * self.head_dim\n'
    '            _k = qkv[:, self.q_size : self.q_size + _kv]\n'
    '            _v = qkv[:, self.q_size + _kv : self.q_size + 2 * _kv]\n'
    '            _ik0 = self.q_size + 2 * _kv + self.index_q_size\n'
    '            _index_k = qkv[:, _ik0 : _ik0 + self.idx_head_dim]\n'
    '            self._insert_kv(_k, _v, _index_k)\n'
    '\n'
    '        output = torch.empty_like(q)\n',
    1,
)

# ── 4) model.py: _insert_kv — pool store instead of reshape_and_cache ───────
ins_old = (
    '        # Identity scale: unused for the bf16 cache, required arg of the op.\n'
    '        key_cache, value_cache = self.kv_cache.unbind(1)\n'
)
assert m.count(ins_old) == 1, "model _insert_kv anchor"
m = m.replace(
    ins_old,
    '        if self._kvarn_state is not None:\n'
    '            # KVarN: rotated fp16 pool write (tiles packed at flush by the\n'
    '            # metadata builder). Index-K cache scatter below is unchanged.\n'
    '            if self._kvarn_state.ensure(self.kv_cache, self._kvarn_group):\n'
    '                self._kvarn_state.store(\n'
    '                    key, value, main_meta.slot_mapping, self._kvarn_group)\n'
    '            idx_cache = self.indexer.index_cache.kv_cache.view(\n'
    '                -1, self.idx_head_dim)\n'
    '            idx_cache[index_meta.slot_mapping] = index_key.to(idx_cache.dtype)\n'
    '            return\n'
    + ins_old,
    1,
)
MODEL.write_text(m)
py_compile.compile(str(MODEL), doraise=True)
print("Patched", MODEL)

# ── 5) sparse_attention.py ───────────────────────────────────────────────────
s = SPARSE.read_text()

# 5a) accept the kvarn dtype strings
cv_old = (
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "bfloat16",\n'
)
assert s.count(cv_old) == 1, "ClassVar anchor"
s = s.replace(
    cv_old,
    cv_old
    + '        "kvarn_k4v2_g128",\n        "kvarn_k4v4_g128",\n'
    + '        "kvarn_k4v2_g64",\n        "kvarn_k4v4_g64",\n',
    1,
)

# 5b) cache shape: uint8 tile records under kvarn (+ module flag for strides)
shape_old = '        return (num_blocks, 2, block_size, num_kv_heads, head_size)\n'
assert s.count(shape_old) == 1, "get_kv_cache_shape anchor"
s = s.replace(
    shape_old,
    '        if str(cache_dtype_str).startswith("kvarn"):\n'
    '            from vllm.models.minimax_m3.common import kvarn_sparse as _ks\n'
    '            import vllm.models.minimax_m3.common.sparse_attention as _sa\n'
    '            _sa._KVARN_MAIN = True\n'
    '            cfg = _ks._cfg(str(cache_dtype_str), head_size)\n'
    '            assert block_size == cfg.group\n'
    '            return (num_blocks, num_kv_heads, cfg.tile_bytes_aligned)\n'
    + shape_old,
    1,
)

# 5c) stride order: 3-D identity when the kvarn record shape is active
stride_old = '        cache_layout = get_kv_cache_layout()\n'
assert s.count(stride_old) == 1, "stride order anchor"
s = s.replace(
    stride_old,
    '        if globals().get("_KVARN_MAIN", False):\n'
    '            return (0, 1, 2)\n'
    + stride_old,
    1,
)

# 5d) builder: flush/bookkeeping hook (eager, between replays; skipped during
#     cudagraph capture where the metadata is synthetic)
b_old = (
    '        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n'
    '            split_decodes_and_prefills(\n'
)
assert s.count(b_old) == 1, "builder anchor"
s = s.replace(
    b_old,
    '        from vllm.models.minimax_m3.common import kvarn_sparse as _ks\n'
    '        if _ks._STATE is not None and not getattr(\n'
    '                self, "_kvarn_in_capture", False):\n'
    '            _ks._STATE.builder_step(common_attn_metadata)\n'
    + b_old,
    1,
)
cls_impl_old = 'class MiniMaxM3SparseImpl(AttentionImplBase[MiniMaxM3SparseMetadata]):\n'
assert s.count(cls_impl_old) == 1, "impl class anchor"
s = s.replace(
    cls_impl_old,
    'def _kvarn_bfcc(self, common_attn_metadata):\n'
    '    # capture-time build: synthetic metadata -> no kvarn host bookkeeping\n'
    '    self._kvarn_in_capture = True\n'
    '    try:\n'
    '        return self.build(0, common_attn_metadata)\n'
    '    finally:\n'
    '        self._kvarn_in_capture = False\n'
    '\n\n'
    'MiniMaxM3SparseMetadataBuilder.build_for_cudagraph_capture = _kvarn_bfcc\n'
    '\n\n'
    + cls_impl_old,
    1,
)

# 5e) Triton impl forward: workspace substitution + q/out rotation
dec_old = (
    '            minimax_m3_sparse_attn_decode(\n'
    '                q[:nd],\n'
    '                kv_cache,\n'
    '                decode_topk,\n'
    '                d.block_table,\n'
    '                d.seq_lens,\n'
    '                self.num_kv_heads,\n'
    '                self.scale,\n'
    '                out[:nd],\n'
    '                d.decode_query_len,\n'
    '            )\n'
)
assert s.count(dec_old) == 1, "decode call anchor"
s = s.replace(
    dec_old,
    '            _st = getattr(layer, "_kvarn_state", None)\n'
    '            if _st is not None and _st.ensure(\n'
    '                    kv_cache, layer._kvarn_group):\n'
    '                from vllm.models.minimax_m3.common import kvarn_sparse as _ks\n'
    '                import os as _os\n'
    '                if _os.environ.get("KVARN_SPARSE_FUSED_DECODE", "1") == "1":\n'
    '                    # Phase 2C: dequant + q-rotation fused into the split\n'
    '                    # kernel (no decode workspace round-trip).\n'
    '                    _ks.kvarn_sparse_attn_decode_fused(\n'
    '                        q[:nd], _st, layer._kvarn_group, decode_topk,\n'
    '                        d.block_table, d.seq_lens, self.num_kv_heads,\n'
    '                        self.scale, out[:nd], d.decode_query_len)\n'
    '                else:\n'
    '                    _ws, _btws = _ks.build_ws_decode(\n'
    '                        _st, layer._kvarn_group, decode_topk, d.block_table,\n'
    '                        nd, d.decode_query_len)\n'
    '                    _q_rot = _ks.rotate_heads(q[:nd], _st._H16)\n'
    '                    _o_rot = torch.empty_like(_q_rot)\n'
    '                    minimax_m3_sparse_attn_decode(\n'
    '                        _q_rot, _ws, decode_topk, _btws, d.seq_lens,\n'
    '                        self.num_kv_heads, self.scale, _o_rot,\n'
    '                        d.decode_query_len)\n'
    '                    out[:nd] = _ks.rotate_heads(_o_rot, _st._H16)\n'
    '            else:\n'
    '                minimax_m3_sparse_attn_decode(\n'
    '                    q[:nd],\n'
    '                    kv_cache,\n'
    '                    decode_topk,\n'
    '                    d.block_table,\n'
    '                    d.seq_lens,\n'
    '                    self.num_kv_heads,\n'
    '                    self.scale,\n'
    '                    out[:nd],\n'
    '                    d.decode_query_len,\n'
    '                )\n',
    1,
)
pre_old = (
    '            minimax_m3_sparse_attn(\n'
    '                q[nd:],\n'
    '                kv_cache,\n'
    '                prefill_topk,\n'
    '                p.block_table,\n'
)
assert s.count(pre_old) == 1, "prefill call anchor"
pre_rest = (
    '                p.cu_seqlens_q,\n'
    '                p.seq_lens,\n'
    '                p.context_lens,\n'
    '                p.max_query_len,\n'
    '                self.num_kv_heads,\n'
    '                self.scale,\n'
    '                out[nd:],\n'
    '            )\n'
)
assert s.count(pre_rest) == 1, "prefill call tail anchor"
s = s.replace(
    pre_old + pre_rest,
    '            _st = getattr(layer, "_kvarn_state", None)\n'
    '            if _st is not None and _st.ensure(\n'
    '                    kv_cache, layer._kvarn_group):\n'
    '                from vllm.models.minimax_m3.common import kvarn_sparse as _ks\n'
    '                import os as _os\n'
    '                if _os.environ.get("KVARN_SPARSE_FUSED_PREFILL", "1") == "1":\n'
    '                    # 2D: in-kernel dequant + both rotations; no workspace\n'
    '                    _ks.kvarn_sparse_attn_prefill_fused(\n'
    '                        q[nd:], _st, layer._kvarn_group, prefill_topk,\n'
    '                        p.block_table, p.cu_seqlens_q, p.seq_lens,\n'
    '                        p.context_lens, p.max_query_len,\n'
    '                        self.num_kv_heads, self.scale, out[nd:])\n'
    '                else:\n'
    '                    _ws, _btws = _ks.build_ws_prefill(\n'
    '                        _st, layer._kvarn_group, p.block_table,\n'
    '                        p.block_table.shape[0])\n'
    '                    _q_rot = _ks.rotate_heads(q[nd:], _st._H16)\n'
    '                    _o_rot = torch.empty_like(_q_rot)\n'
    '                    minimax_m3_sparse_attn(\n'
    '                        _q_rot, _ws, prefill_topk, _btws,\n'
    '                        p.cu_seqlens_q, p.seq_lens, p.context_lens,\n'
    '                        p.max_query_len, self.num_kv_heads, self.scale,\n'
    '                        _o_rot)\n'
    '                    out[nd:] = _ks.rotate_heads(_o_rot, _st._H16)\n'
    '            else:\n'
    + pre_old.replace("            ", "                ", 1)
      .replace("\n            ", "\n                ")
    + pre_rest.replace("\n            ", "\n                "),
    1,
)

SPARSE.write_text(s)
py_compile.compile(str(SPARSE), doraise=True)
print("Patched", SPARSE)
print("MiniMax-M3 kvarn-sparse (Phase 2A: sparse MSA on KVarN tiles) applied.")
PY
