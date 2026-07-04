#!/bin/bash
set -euo pipefail

# MiniMax-M3 indexer (index-K side cache) on KVarN 4-bit tiles (Phase 2B).
#
# Enabled via --attention-config '{"indexer_kv_dtype": "kvarn"}'.
# The index-K cache goes from bf16 256 B/token to a KVarN K-only tile record
# (8960 B / 128-token tile = 70 B/token) and the scorer kernels dequantize
# IN-KERNEL (the indexer scans the whole context each decode step, so a
# workspace would re-materialize the full cache per step).
#   * decode scorer: records only. M3 (init=0, local=1) force-scores the tail
#     block without reading K, and every non-tail tile is flushed by the
#     builder before it stops being local.
#   * prefill scorer: per-block branch — rotated fp16 pool (current chunk,
#     unpacked) or record dequant (older context, incl. prefix-cache hits).
#   * index_q rotated by H before scoring ((qH)(kH)^T = qk^T); scores are
#     rank-only so no un-rotation is needed.
# Bookkeeping reuses kvarn_sparse.KVarNSparseGroup with a separate instance
# (indexer kv-cache group has its own block-id space + builder).
#
# Apply AFTER mods/add-kvarn-kv-quant AND mods/fix-minimax-m3-kvarn-sparse.

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
IDX = common / "indexer.py"
MODEL = pkg / "vllm/models/minimax_m3/nvidia/model.py"
ATTN_CFG = pkg / "vllm/config/attention.py"

if (common / "kvarn_indexer.py").exists():
    print("kvarn-indexer already applied")
    raise SystemExit(0)

assert (common / "kvarn_sparse.py").exists(), (
    "apply mods/fix-minimax-m3-kvarn-sparse first")

shutil.copy(mod_dir / "kvarn_indexer.py", common / "kvarn_indexer.py")
py_compile.compile(str(common / "kvarn_indexer.py"), doraise=True)
print("Installed", common / "kvarn_indexer.py")

# ── 0) config: allow indexer_kv_dtype="kvarn" ────────────────────────────────
a = ATTN_CFG.read_text()
old = 'IndexerKVDType = Literal["bf16", "fp8", "mxfp4", "nvfp4"]'
assert a.count(old) == 1, "IndexerKVDType anchor"
a = a.replace(
    old, 'IndexerKVDType = Literal["bf16", "fp8", "mxfp4", "nvfp4", "kvarn"]', 1)
ATTN_CFG.write_text(a)
py_compile.compile(str(ATTN_CFG), doraise=True)
print("Patched", ATTN_CFG)

# ── 1) indexer.py ────────────────────────────────────────────────────────────
s = IDX.read_text()

# 1a) backend shape: flat uint8 record per block under kvarn
shape_old = "        return (num_blocks, block_size, head_size)\n"
assert s.count(shape_old) == 1, "indexer shape anchor"
s = s.replace(
    shape_old,
    "        # kvarn indexer cache is opted in per-layer (indexer_kv_dtype),\n"
    "        # NOT by the global --kv-cache-dtype: gate on the module flag so\n"
    "        # kvarn-main + bf16-indexer configs keep the stock shape.\n"
    "        import vllm.models.minimax_m3.common.indexer as _idxmod\n"
    '        if getattr(_idxmod, "_KVARN_IDX", False):\n'
    "            # K-only tile record (possibly stride-padded so N records\n"
    "            # tile the manager page exactly; set at spec time).\n"
    "            from vllm.models.minimax_m3.common.kvarn_indexer import (\n"
    "                rec_layout,\n"
    "            )\n"
    "            stride = getattr(\n"
    '                _idxmod, "_KVARN_IDX_REC_STRIDE",\n'
    '                rec_layout(head_size)["REC"])\n'
    "            return (num_blocks, stride)\n"
    + shape_old,
    1,
)
stride_old = (
    "        if include_num_layers_dimension:\n"
    "            # M3 does not use cross-layer (per-layer-stacked) KV blocks.\n"
    "            raise NotImplementedError\n"
    "        return (0, 1, 2)\n"
)
assert s.count(stride_old) == 1, "indexer stride anchor"
s = s.replace(
    stride_old,
    "        if include_num_layers_dimension:\n"
    "            # M3 does not use cross-layer (per-layer-stacked) KV blocks.\n"
    "            raise NotImplementedError\n"
    "        # kvarn record caches are 2-D; runner uses identity order then.\n"
    "        import vllm.models.minimax_m3.common.indexer as _idxmod\n"
    '        if getattr(_idxmod, "_KVARN_IDX", False):\n'
    "            return (0, 1)\n"
    "        return (0, 1, 2)\n",
    1,
)

# 1b) cache: accept kvarn dtype, uint8 storage dtype, TQ spec (108->70B slot)
cache_old = (
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
assert s.count(cache_old) == 1, "indexer cache init anchor"
s = s.replace(
    cache_old,
    '        if indexer_kv_dtype not in ("bf16", "kvarn"):\n'
    "            raise NotImplementedError(\n"
    '                f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported yet "\n'
    "                \"for the MiniMax M3 indexer cache ('bf16' or 'kvarn').\"\n"
    "            )\n"
    "        self.kv_cache = torch.tensor([])\n"
    "        self.head_dim = head_dim\n"
    "        self.indexer_kv_dtype = indexer_kv_dtype\n"
    "        # Storage dtype: bf16 vectors, or uint8 kvarn tile records.\n"
    "        self.dtype = (\n"
    '            torch.uint8 if indexer_kv_dtype == "kvarn" else torch.bfloat16\n'
    "        )\n"
    '        if indexer_kv_dtype == "kvarn":\n'
    "            import vllm.models.minimax_m3.common.indexer as _idxmod\n"
    "            _idxmod._KVARN_IDX = True\n",
    1,
)
spec_old = (
    "        # Key-only: MLAAttentionSpec budgets one vector/token (not 2x for K+V).\n"
    "        return MLAAttentionSpec(\n"
    "            block_size=vllm_config.cache_config.block_size,\n"
    "            num_kv_heads=1,\n"
    "            head_size=self.head_dim,\n"
    "            dtype=self.dtype,\n"
    "        )\n"
)
assert s.count(spec_old) == 1, "indexer spec anchor"
s = s.replace(
    spec_old,
    '        if self.indexer_kv_dtype == "kvarn":\n'
    "            from vllm.models.minimax_m3.common.kvarn_indexer import rec_layout\n"
    "            from vllm.v1.kv_cache_interface import TQFullAttentionSpec\n"
    "            import vllm.models.minimax_m3.common.indexer as _idxmod\n"
    "            block_size = vllm_config.cache_config.block_size\n"
    "            assert block_size == 128, \"kvarn indexer needs block_size 128\"\n"
    "            rec = rec_layout(self.head_dim)[\"REC\"]\n"
    "            # NOTE: keep block_size == 128 so ALL specs stay in vLLM's\n"
    "            # UniformTypeKVCacheSpecs single-group path, which allocates\n"
    "            # per-layer tensors at each layer's OWN page size -- the\n"
    "            # 8960-byte indexer page is NOT padded there. A block-384\n"
    "            # 'packing' variant was tried and REGRESSED capacity (230k vs\n"
    "            # 264k tokens): differing block sizes split the uniform group\n"
    "            # into 2 padded groups (57->60 layers, full 27648B slots).\n"
    "            assert rec % block_size == 0\n"
    "            _idxmod._KVARN_IDX_REC_STRIDE = rec\n"
    "            return TQFullAttentionSpec(\n"
    "                block_size=block_size,\n"
    "                num_kv_heads=1,\n"
    "                head_size=self.head_dim,\n"
    "                head_size_v=self.head_dim,\n"
    "                dtype=torch.uint8,\n"
    "                tq_slot_size=rec // block_size,\n"
    "            )\n"
    + spec_old,
    1,
)

# 1c) builder hook: kvarn bookkeeping for the indexer group (eager)
b_old = (
    "        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n"
    "            split_decodes_and_prefills(\n"
)
assert s.count(b_old) == 1, "indexer builder anchor"
s = s.replace(
    b_old,
    "        from vllm.models.minimax_m3.common import kvarn_indexer as _ki\n"
    "        if _ki._STATE is not None and not getattr(\n"
    '                self, "_kvarn_in_capture", False):\n'
    "            _ki._STATE.builder_step(common_attn_metadata)\n"
    + b_old,
    1,
)
impl_cls_old = "class MiniMaxM3IndexerImpl(nn.Module):\n"
assert s.count(impl_cls_old) == 1, "indexer impl class anchor"
s = s.replace(
    impl_cls_old,
    "def _kvarn_idx_bfcc(self, common_attn_metadata):\n"
    "    # capture-time build: synthetic metadata -> no kvarn host bookkeeping\n"
    "    self._kvarn_in_capture = True\n"
    "    try:\n"
    "        return self.build(0, common_attn_metadata)\n"
    "    finally:\n"
    "        self._kvarn_in_capture = False\n"
    "\n\n"
    "MiniMaxM3IndexerTritonMetadataBuilder.build_for_cudagraph_capture = (\n"
    "    _kvarn_idx_bfcc\n"
    ")\n"
    "\n\n"
    + impl_cls_old,
    1,
)

# 1d) impl selection: kvarn -> Triton impl (kvarn kernels chosen in forward)
sel_old = (
    '    if indexer_kv_dtype != "bf16":\n'
    "        raise NotImplementedError(\n"
    '            f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported by the "\n'
    '            "Triton indexer impl."\n'
    "        )\n"
)
assert s.count(sel_old) == 1, "impl select anchor"
s = s.replace(
    sel_old,
    '    if indexer_kv_dtype not in ("bf16", "kvarn"):\n'
    "        raise NotImplementedError(\n"
    '            f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported by the "\n'
    '            "Triton indexer impl."\n'
    "        )\n",
    1,
)

# 1e) impl init: kvarn per-layer state
impl_init_old = (
    "        # Owns the side cache (registers itself in the static forward context).\n"
    "        self.index_cache = MiniMaxM3IndexerCache(\n"
)
assert s.count(impl_init_old) == 1, "impl init anchor"
s = s.replace(
    impl_init_old,
    "        # KVarN 4-bit index-K tiles (mod: kvarn-indexer)\n"
    "        self._kvarn_state = None\n"
    "        self._kvarn_group = None\n"
    '        if indexer_kv_dtype == "kvarn":\n'
    "            from vllm.config.vllm import get_current_vllm_config\n"
    "            from vllm.models.minimax_m3.common import kvarn_indexer as _ki\n"
    "            self._kvarn_group = _ki.get_group(get_current_vllm_config())\n"
    "            self._kvarn_state = _ki.KVarNIndexerLayer(\n"
    '                f"{prefix}.index_cache", index_head_dim)\n'
    + impl_init_old,
    1,
)

# 1f) Triton forward: kvarn scorer variants (reuse stock topk on the scores)
fwd_old = (
    "        if index_md.num_decodes > 0:\n"
    "            d = index_md.decode\n"
    "            assert d is not None\n"
    "            decode_topk = minimax_m3_index_decode(\n"
)
assert s.count(fwd_old) == 1, "triton fwd decode anchor"
s = s.replace(
    fwd_old,
    "        _st = self._kvarn_state\n"
    "        if _st is not None and _st.ensure(kv, self._kvarn_group):\n"
    "            from vllm.models.minimax_m3.common import kvarn_indexer as _ki\n"
    "            iq = torch.matmul(\n"
    "                iq.to(torch.float16), _st._H16).to(iq.dtype)\n"
    "            return _kvarn_indexer_forward(self, iq, index_md, _st, _ki)\n"
    + fwd_old,
    1,
)

# forward body appended at module scope (kernels mirror index_topk wrappers)
s += '''

def _kvarn_indexer_forward(self, iq, index_md, _st, _ki):
    """(decode_topk, prefill_topk) with kvarn scorers + stock topk kernels."""
    import triton as _triton
    from vllm.models.minimax_m3.common.ops.index_topk import (
        SPARSE_BLOCK_SIZE,
        _topk_index_kernel,
        _topk_index_merge_kernel,
        _topk_index_partial_kernel,
    )
    from vllm.platforms import current_platform
    from vllm.utils.math_utils import round_up

    nd = index_md.num_decode_tokens
    decode_topk = prefill_topk = None
    if index_md.num_decodes > 0:
        d = index_md.decode
        idxq = iq[:nd]
        total_q = idxq.shape[0]
        h = self.num_index_heads
        topk = self.topk_blocks
        max_block = _triton.cdiv(d.max_seq_len, SPARSE_BLOCK_SIZE)
        use_pdl = current_platform.is_arch_support_pdl()
        pdl_launch = {"launch_pdl": True} if use_pdl else {}
        score_stride = round_up(max_block, 16)
        score = torch.empty((h, total_q, score_stride), dtype=torch.float32,
                            device=idxq.device)
        TARGET_GRID = 4096
        target = max(1, min(256, TARGET_GRID // max(1, total_q * h)))
        num_kv_chunks = 1 << (target.bit_length() - 1)
        _ki.kvarn_index_decode_score(
            idxq, _st, d.block_table, d.seq_lens, d.max_seq_len,
            self.init_blocks, self.local_blocks, self.scale,
            d.decode_query_len, score, num_kv_chunks, use_pdl, pdl_launch)
        # stock split top-k over the scores (unchanged)
        topk_idx = torch.empty((h, total_q, topk), dtype=torch.int32,
                               device=idxq.device)
        topk_target = max(1, min(16, 64 // max(1, total_q * h)))
        num_topk_chunks = 1 << (topk_target.bit_length() - 1)
        block_size_t = _triton.next_power_of_2(topk)
        chunk_blocks = (max_block + num_topk_chunks - 1) // num_topk_chunks
        sp = torch.empty(num_topk_chunks, h, total_q, block_size_t,
                         dtype=torch.float32, device=idxq.device)
        ip = torch.empty_like(sp, dtype=torch.int32)
        _topk_index_partial_kernel[(total_q, h, num_topk_chunks)](
            score, sp, ip, d.seq_lens, SPARSE_BLOCK_SIZE, topk, chunk_blocks,
            d.decode_query_len,
            score.stride(0), score.stride(1), score.stride(2),
            sp.stride(0), sp.stride(1), sp.stride(2), sp.stride(3),
            ip.stride(0), ip.stride(1), ip.stride(2), ip.stride(3),
            USE_PDL=use_pdl, **pdl_launch)
        _topk_index_merge_kernel[(total_q, h)](
            sp, ip, topk_idx, d.seq_lens, SPARSE_BLOCK_SIZE, topk,
            d.decode_query_len,
            sp.stride(0), sp.stride(1), sp.stride(2), sp.stride(3),
            ip.stride(0), ip.stride(1), ip.stride(2), ip.stride(3),
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            num_topk_chunks=num_topk_chunks, USE_PDL=use_pdl, **pdl_launch)
        decode_topk = topk_idx
    if index_md.num_prefills > 0:
        p = index_md.prefill
        idxq = iq[nd:]
        total_q = idxq.shape[0]
        h = self.num_index_heads
        topk = self.topk_blocks
        batch = p.cu_seqlens_q.shape[0] - 1
        max_block = _triton.cdiv(p.max_seq_len, SPARSE_BLOCK_SIZE)
        score_stride = round_up(max_block, 16)
        score = torch.empty((h, total_q, score_stride), dtype=torch.float32,
                            device=idxq.device)
        _ki.kvarn_index_prefill_score(
            idxq, _st, self._kvarn_group, p.block_table, p.cu_seqlens_q,
            p.seq_lens, p.context_lens, p.max_query_len, self.scale, score)
        topk_idx = torch.empty((h, total_q, topk), dtype=torch.int32,
                               device=idxq.device)
        _topk_index_kernel[(p.max_query_len, batch, h)](
            score, topk_idx, 1, SPARSE_BLOCK_SIZE,
            p.cu_seqlens_q, p.cu_seqlens_q, p.context_lens,
            topk, self.init_blocks, self.local_blocks,
            score.stride(0), score.stride(1), score.stride(2),
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            MASK_INIT=False, MASK_LOCAL=False)
        prefill_topk = topk_idx
    return decode_topk, prefill_topk
'''
IDX.write_text(s)
py_compile.compile(str(IDX), doraise=True)
print("Patched", IDX)

# ── 2) model.py: index-K store -> kvarn pool under kvarn ─────────────────────
m = MODEL.read_text()
ins_old = (
    "            idx_cache = self.indexer.index_cache.kv_cache.view(\n"
    "                -1, self.idx_head_dim)\n"
    "            idx_cache[index_meta.slot_mapping] = index_key.to(idx_cache.dtype)\n"
    "            return\n"
)
assert m.count(ins_old) == 1, "model kvarn _insert_kv anchor (needs kvarn-sparse)"
m = m.replace(
    ins_old,
    "            _ist = getattr(self.indexer.impl, \"_kvarn_state\", None)\n"
    "            if _ist is not None:\n"
    "                if _ist.ensure(self.indexer.index_cache.kv_cache,\n"
    "                               self.indexer.impl._kvarn_group):\n"
    "                    _ist.store(index_key, index_meta.slot_mapping,\n"
    "                               self.indexer.impl._kvarn_group)\n"
    "            else:\n"
    "                idx_cache = self.indexer.index_cache.kv_cache.view(\n"
    "                    -1, self.idx_head_dim)\n"
    "                idx_cache[index_meta.slot_mapping] = index_key.to(\n"
    "                    idx_cache.dtype)\n"
    "            return\n",
    1,
)
MODEL.write_text(m)
py_compile.compile(str(MODEL), doraise=True)
print("Patched", MODEL)
print("MiniMax-M3 kvarn-indexer (Phase 2B: 4-bit index-K tiles) applied.")
PY
