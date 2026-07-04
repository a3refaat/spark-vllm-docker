"""b12x CuTe MSA lightning-indexer impl for MiniMax-M3 on SM120/SM121 (GB10).

Replaces the Triton index score+top-k (``MiniMaxM3IndexerTritonImpl``) with
b12x's MSA indexer kernels:
  * decode  -> ``msa_paged_decode_block_scores`` (b12x fused CuTe score +
               per-head PAGE_HEAD_MAX block max-pool, page_size 64) followed by
               a fused Triton top-k select (``_msa_select_kernel``, bit-exact
               replacement for b12x's launch-bound PyTorch ``msa_topk_blocks``
               tail). Both graph-captured.
  * prefill -> ``msa_q2k_indices_prefill`` bound to ONE max-capacity contiguous
               scratch (``plan_indexer_contiguous_scratch`` at
               max_num_batched_tokens x max_model_len). The binding owns the
               metadata/query_positions/block_base/output; per call we only
               rebind k_start/k_end and refill the scratch K -> the CuTe scorer
               + tiled top-k compile EXACTLY once (fixed supertile_k=32768) and
               every later prefill runs warm (~1-5ms), any context length.

Both emit q2k block ids in the SAME contract the downstream attention wants:
``[num_kv_heads, q_rows, 16]`` int32, ascending, ``-1`` padded at the END
(valid ids contiguous at the front). That layout is consumed directly by BOTH
the b12x MSA decode attention AND the Triton block-sparse attend (which counts
``sum(idx>=0)`` leading entries) -- no reorder, no Triton indexer.

INDEX SIDE CACHE (b12x packed contract, reusing the SAME indexer KV group):
  The cache is allocated as uint8 with packed head bytes 132 = 128 (fp8 e4m3 K)
  + 4 (per-token fp32 scale), block_size 128 -> 16896 bytes / 128-block. b12x's
  paged indexer kernel HARD-REQUIRES page_size==64 and a page-MAJOR byte layout
  ``[num_pages64, 64*(128+4)=8448]`` (all 64 tokens' fp8 K, then all 64 fp32
  scales). One 128-block == two 64-pages, so we reinterpret the same storage as
  ``[num_blocks*2, 8448]`` and derive a 64-page table from the existing 128
  block table: ``page64[i, 2c+r] = 2*block_table[i,c] + r``. No new KV group,
  no block-size change -- only the cache dtype/width and the write/read change.

The packed write lives in the layer's ``_insert_kv`` (mod patches model.py); it
calls ``write_packed_index_cache`` here, which scatters via a fused Triton
quant+pack kernel (``_cache_write_kernel``, bit-exact replacement for the
launch-bound PyTorch byte-scatter). This module owns the read/score/top-k side.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.forward_context import get_forward_context
from vllm.config import get_current_vllm_config
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
)

from b12x.attention.indexer import (
    MSA_TOPK_BLOCKS,
    MSA_SM_SCALE,
    build_paged_mqa_schedule_metadata,
    quantize_msa_q_fp8,
    msa_paged_decode_block_scores,
    msa_decode_query_positions,
    msa_q2k_indices_decode,
    msa_q2k_indices_prefill,
    msa_topk_blocks,
)
from b12x.attention.indexer.contiguous_kernel import (
    run_contiguous_block_scores_kernel_nvf4,
)

# Bisect toggles (default ON). Set B12X_TRITON_SELECT=0 / B12X_TRITON_CACHE_WRITE=0
# to fall back to the stock b12x PyTorch tails for isolation.
import os as _os
_USE_TRITON_SELECT = _os.getenv("B12X_TRITON_SELECT", "1") != "0"
_USE_TRITON_CACHE_WRITE = _os.getenv("B12X_TRITON_CACHE_WRITE", "1") != "0"
_USE_TRITON_QQUANT = _os.getenv("B12X_TRITON_QQUANT", "1") != "0"
# Phase 5: one-pass fused MSA indexer (score->block-max->topk->q2k in ONE
# kernel). Env B12X_MSA_FUSED_INDEXER=0 restores the scheduled scorer +
# select-tail chain. fp8 index-K only until the nvfp4 score stage lands.
from b12x.attention.paged.tuning.policy import msa_fused_indexer_enabled
_USE_FUSED_INDEXER = msa_fused_indexer_enabled()
# Native nvfp4 contiguous MSA scorer for prefill/verify (ZERO dequant->requant):
# byte-gather e2m1 K + per-16 e4m3 straight from the paged nvfp4 cache and score
# via the mxf4nvf4 MMA (validated bit-exact + compile-once). Default ON for the
# nvfp4 cache; set B12X_INDEXER_NATIVE_NVF4=0 to fall back to the dequant->fp8
# contiguous scorer.
_USE_NATIVE_NVF4 = _os.getenv("B12X_INDEXER_NATIVE_NVF4", "1") != "0"
# Verify-grouped fused indexer (q_len rows per K stream, one kernel).
# Bit-exact vs the multirow fused path (tests/test_fused_msa_verify_indexer);
# supersedes both the multirow fused verify AND the native-nvf4 verify chain
# when q_len*heads fits the 8-slot select cap. B12X_MSA_FUSED_VERIFY=0 rolls
# back to the previous per-quant verify routes.
_USE_FUSED_VERIFY = _os.getenv("B12X_MSA_FUSED_VERIFY", "1") != "0"
# Build the decode indexer metadata (page64 table + paged-MQA schedule + query
# positions) ONCE per forward instead of redundantly in all 57 sparse layers.
# The metadata depends only on the decode block_table/seq_lens, which are the
# SAME across layers. Graph-safe: the buffers are shared module-wide and the
# first-constructed indexer (layer 3, which also executes first) owns the build,
# so the build kernels are captured once and every later layer reuses them.
# Toggle OFF => every layer rebuilds into the shared buffers (bisect fallback).
_USE_META_ONCE = _os.getenv("B12X_META_ONCE", "1") != "0"
# Indexer prefill (contiguous) supertile width. The 296 MB prefill scratch is
# ~90% a [max_num_batched_tokens x supertile_k] f32 logits buffer, so it scales
# LINEARLY with supertile_k. The supertile loop is runtime (over k_start/k_end),
# so a fixed smaller value stays compile-once and keeps full max_model_len
# capacity -- it just scores the context in more chunks (slightly slower
# prefill). 32768 = b12x default (2 chunks @ 65536); 8192 -> ~95 MB (8 chunks).
_IDX_SUPERTILE_K = max(int(_os.getenv("B12X_IDX_SUPERTILE_K", "32768")), 1)
# plan_indexer_*_scratch + the caps live in the .scratch submodule (the package
# __init__ only re-exports the unified plan_indexer_scratch).
from b12x.attention.paged.tuning.policy import debug_paged_policy_enabled
from b12x.attention.indexer.scratch import (
    B12XIndexerPagedScratchCaps,
    B12XIndexerContiguousScratchCaps,
    plan_indexer_paged_scratch,
    plan_indexer_contiguous_scratch,
)

# Shared indexer decode scratch across ALL sparse layers (graph-safe: attention
# runs sequentially layer-by-layer; each layer's indexer writes the shared
# scratch -- incl. its q2k output -- and that layer's main MSA attention reads
# it before the next layer's indexer overwrites). Keyed by storage signature.
_IDX_DEC_STORE: dict = {}

# Shared decode metadata buffers across ALL sparse layers (page64 table +
# schedule + int32 seqlens + query positions), keyed by (batch, width64). Built
# once per forward by the metadata owner (see _USE_META_ONCE).
_FWD_META_STORE: dict = {}
# The first MiniMaxM3IndexerB12xImpl constructed (== lowest sparse layer ==
# first to execute) owns the per-forward metadata build.
_META_OWNER_TAKEN = [False]


def _fwd_meta(batch: int, width64: int, device) -> dict:
    key = (int(batch), int(width64), str(device))
    m = _FWD_META_STORE.get(key)
    if m is None:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        m = _FWD_META_STORE[key] = dict(
            page64=torch.zeros((batch, width64), dtype=torch.int32, device=device),
            sched=torch.empty((num_sms + 1, 2), dtype=torch.int32, device=device),
            seqlens=torch.zeros((batch,), dtype=torch.int32, device=device),
            qpos=torch.zeros((batch,), dtype=torch.int32, device=device),
        )
    return m


def _shared_idx_storage(key, spec, device):
    if key not in _IDX_DEC_STORE:
        _IDX_DEC_STORE[key] = torch.zeros(spec.shape, dtype=spec.dtype, device=device)
    return _IDX_DEC_STORE[key]


# ONE max-capacity contiguous MSA prefill scratch shared across ALL sparse
# layers. Bound at (max_num_batched_tokens x max_model_len): every prefill
# passes the SAME fixed q/k shapes and varies only k_start/k_end, so b12x's MSA
# prefill scorer + tiled top-k JIT-compile EXACTLY once and then serve any
# context length warm (~1-5ms) -- empirically proven compile-once-at-max. Shared
# across layers is graph-safe: layers run sequentially, each fully consuming the
# scratch (incl. its q2k output) before the next layer rebinds it.
_IDX_CTG_STORE: dict = {}


def _contiguous_ctx(qmax, kmax, heads, device):
    key = (int(qmax), int(kmax), int(heads), int(_IDX_SUPERTILE_K), str(device))
    ctx = _IDX_CTG_STORE.get(key)
    if ctx is None:
        caps = B12XIndexerContiguousScratchCaps(
            device=device, num_q_heads=1, num_idx_heads=int(heads),
            max_q_rows=int(qmax), max_k_rows=int(kmax),
            topk=MSA_TOPK_BLOCKS, score_mode="msa", supertile_k=_IDX_SUPERTILE_K)
        plan = plan_indexer_contiguous_scratch(caps)
        (spec,) = plan.scratch_specs()
        # Zero once: we only refill the k_quant prefix [:k_rows] per call, so a
        # zeroed tail keeps masked-region K finite (no NaN can sneak into the
        # block max-pool before the causal k_end mask applies).
        scratch = torch.zeros(spec.shape, dtype=spec.dtype, device=device)
        ctx = _IDX_CTG_STORE[key] = dict(plan=plan, scratch=scratch)
    return ctx


# Fixed-capacity native-nvf4 contiguous MSA buffers, shared across ALL sparse
# layers (K/Q/block_scores are transient -- consumed immediately by topk; only
# the per-layer q2k output is persistent). Bound at (qmax, kmax) so the native
# block-score scorer compiles ONCE and indexes by k_start/k_end at runtime --
# the same compile-once pattern as the fp8 contiguous scratch, minus the ~315MB
# logits buffer (block scores are tiny).
_IDX_NVF4_STORE: dict = {}
_IDX_NVF4_VERIFY_STORE: dict = {}


def _nvf4_prefill_buffers(qmax, kmax, heads, device):
    key = (int(qmax), int(kmax), int(heads), str(device))
    b = _IDX_NVF4_STORE.get(key)
    if b is None:
        half = _INDEX_HEAD_DIM // 2
        nbmax = (int(kmax) + _MSA_BLOCK_TOKENS - 1) // _MSA_BLOCK_TOKENS
        b = _IDX_NVF4_STORE[key] = dict(
            nbmax=int(nbmax),
            q_e2m1=torch.zeros((qmax, heads, half), dtype=torch.uint8, device=device),
            q_sfa=torch.zeros((qmax, heads, _NV_NBLK), dtype=torch.uint8, device=device),
            weights=torch.full((qmax, heads), MSA_SM_SCALE, dtype=torch.float32, device=device),
            k_e2m1=torch.zeros((kmax, half), dtype=torch.uint8, device=device),
            k_sfb=torch.zeros((kmax, _NV_NBLK), dtype=torch.uint8, device=device),
            k_start=torch.zeros((qmax,), dtype=torch.int32, device=device),
            k_end=torch.zeros((qmax,), dtype=torch.int32, device=device),
            bs=torch.full((heads, qmax, nbmax), float("-inf"), dtype=torch.float32, device=device),
        )
    return b


def _nvf4_verify_buffers(qmax, supertile_k, heads, topk, max_reqs, width128, device):
    """Fixed-capacity native-nvf4 VERIFY buffers.

    K is bounded to one supertile; global K coverage comes from the fixed
    max_chunks loop in _verify_nvf4_native.  q2k is persistent because the main
    MSA verify graph records its data_ptr.
    """
    qmax = int(qmax)
    supertile_k = int(supertile_k)
    heads = int(heads)
    topk = int(topk)
    max_reqs = int(max_reqs)
    width128 = int(width128)
    key = (qmax, supertile_k, heads, topk, max_reqs, width128, str(device))
    b = _IDX_NVF4_VERIFY_STORE.get(key)
    if b is None:
        half = _INDEX_HEAD_DIM // 2
        chunk_blocks = (supertile_k + _MSA_BLOCK_TOKENS - 1) // _MSA_BLOCK_TOKENS
        b = _IDX_NVF4_VERIFY_STORE[key] = dict(
            chunk_blocks=int(chunk_blocks),
            seq=torch.zeros((max_reqs,), dtype=torch.int32, device=device),
            block_table=torch.zeros((max_reqs, width128), dtype=torch.int32, device=device),
            base=torch.zeros((max_reqs,), dtype=torch.int32, device=device),
            cum=torch.zeros((max_reqs,), dtype=torch.int32, device=device),
            k_start_g=torch.zeros((qmax,), dtype=torch.int32, device=device),
            k_end_g=torch.zeros((qmax,), dtype=torch.int32, device=device),
            k_start=torch.zeros((qmax,), dtype=torch.int32, device=device),
            k_end=torch.zeros((qmax,), dtype=torch.int32, device=device),
            block_base=torch.zeros((qmax,), dtype=torch.int32, device=device),
            qpos=torch.zeros((qmax,), dtype=torch.int32, device=device),
            q_e2m1=torch.zeros((qmax, heads, half), dtype=torch.uint8, device=device),
            q_sfa=torch.zeros((qmax, heads, _NV_NBLK), dtype=torch.uint8, device=device),
            weights=torch.full((qmax, heads), MSA_SM_SCALE, dtype=torch.float32, device=device),
            k_e2m1=torch.zeros((supertile_k, half), dtype=torch.uint8, device=device),
            k_sfb=torch.zeros((supertile_k, _NV_NBLK), dtype=torch.uint8, device=device),
            bs=torch.full((heads, qmax, chunk_blocks), float("-inf"), dtype=torch.float32, device=device),
            carry_v=torch.full((2, heads, qmax, topk), float("-inf"), dtype=torch.float32, device=device),
            carry_i=torch.full((2, heads, qmax, topk), 2147483647, dtype=torch.int32, device=device),
            q2k=torch.full((heads, qmax, topk), -1, dtype=torch.int32, device=device),
        )
    return b


_INDEX_HEAD_DIM = 128
_MSA_BLOCK_TOKENS = 128               # MSA sparse block size (tokens per block)
_PACK_PAGE = 64                      # b12x paged-indexer page size (hard contract)
_PACK_ROW_BYTES = _PACK_PAGE * (_INDEX_HEAD_DIM + 4)   # 8448
_PACK_DATA_BYTES = _PACK_PAGE * _INDEX_HEAD_DIM        # 8192
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


# --------------------------------------------------------------------------
# Fused Triton helpers that replace b12x's launch-bound PyTorch tails (each
# collapses ~12-25 tiny graph-captured kernels into ONE). Same kernel class as
# b12x's own shipped Triton gather/pack/metadata helpers -- NOT a triton_attn
# fallback. Both are bit-exact vs the b12x reference (validated over random
# inputs across shapes) and CUDA-graph-capturable.
#   * _msa_select_kernel   <- b12x msa_topk_blocks  (force-local + top-k + sort)
#   * _cache_write_kernel  <- the PyTorch byte-scatter in write_packed_index_cache
# --------------------------------------------------------------------------
_ROW_F32 = _PACK_ROW_BYTES // 4       # 2112 f32 per packed page row
_DATA_F32 = _PACK_DATA_BYTES // 4     # 2048 f32 offset to the per-token scale

# S5 nvfp4 page-major index-K cache: per-16-block e2m1 + e4m3 (72 B/token vs the
# fp8 path's 132). Page row = [page*64 e2m1 | page*8 e4m3] = 4608 B. Matches the
# harness pack_index_k_cache_nvfp4 (the format gate C validated the kernel reads).
_NVFP4_E2M1_MAX = 6.0
_NVFP4_BLK = 16                                                      # channels / scale block
_NV_NBLK = _INDEX_HEAD_DIM // _NVFP4_BLK                             # 8 scale blocks / token
_PACK_DATA_BYTES_NV = _PACK_PAGE * (_INDEX_HEAD_DIM // 2)            # 64*64 = 4096
_PACK_SCALE_BYTES_NV = _PACK_PAGE * _NV_NBLK                         # 64*8  = 512
_PACK_ROW_BYTES_NV = _PACK_DATA_BYTES_NV + _PACK_SCALE_BYTES_NV      # 4608
_E2M1_MAG_T = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e2m1_decode_torch(codes: torch.Tensor) -> torch.Tensor:
    """e2m1 nibble (sign<<3 | mag-idx) -> signed magnitude (torch ref)."""
    sign = ((codes >> 3) & 1).to(torch.float32)
    mag = _E2M1_MAG_T.to(codes.device)[(codes & 0x7).long()]
    return torch.where(sign > 0, -mag, mag)


def _next_pow2(n: int) -> int:
    return 1 << (max(int(n), 1) - 1).bit_length()


@triton.jit
def _msa_select_kernel(bs_ptr, qpos_ptr, out_ptr, R, B,
                       TOPK: tl.constexpr, BLOCK: tl.constexpr,
                       MSA_BLOCK_TOKENS: tl.constexpr):
    pid = tl.program_id(0)                    # row = h*R + r  (block_scores [H,R,B])
    r = pid % R
    offs = tl.arange(0, BLOCK)
    m = offs < B
    scores = tl.load(bs_ptr + pid * B + offs, mask=m, other=float("-inf"))
    qpos = tl.load(qpos_ptr + r)
    local_block = qpos // MSA_BLOCK_TOKENS
    valid_local = (local_block >= 0) & (local_block < B)
    scores = tl.where((offs == local_block) & valid_local, float("inf"), scores)
    ar = tl.arange(0, TOPK)
    sel = tl.full([TOPK], 2147483647, tl.int32)
    for i in range(TOPK):
        mx = tl.max(scores, axis=0)
        am = tl.argmax(scores, axis=0).to(tl.int32)
        put = tl.where(mx > float("-inf"), am, 2147483647)
        sel = tl.where(ar == i, put, sel)
        scores = tl.where(offs == am, float("-inf"), scores)
    srt = tl.sort(sel)                        # ascending; invalid (INT32_MAX) -> tail
    out = tl.where(srt == 2147483647, -1, srt).to(tl.int32)
    tl.store(out_ptr + pid * TOPK + ar, out)


def _triton_msa_select(block_scores, query_positions, topk, out, block):
    """Bit-exact b12x msa_topk_blocks (decode, block_base=None) in one kernel.

    block_scores [H,R,B] f32 contiguous -> out [H,R,topk] i32 (ascending, -1
    padded). ``block`` is the FIXED pow2 >= max blocks so the kernel compiles
    ONCE (B is a runtime arg, masked)."""
    H, R, B = block_scores.shape
    if not block_scores.is_contiguous():
        block_scores = block_scores.contiguous()
    _msa_select_kernel[(H * R,)](
        block_scores, query_positions, out, R, B,
        int(topk), int(block), _MSA_BLOCK_TOKENS, num_warps=4)
    return out


@triton.jit(do_not_specialize=["T"])
def _cache_write_kernel(ik_ptr, slot_ptr, cache_u8_ptr, cache_f32_ptr, T,
                        HD: tl.constexpr, PAGE: tl.constexpr, ROWB: tl.constexpr,
                        ROWF: tl.constexpr, DATAF: tl.constexpr, FP8MAX: tl.constexpr):
    t = tl.program_id(0)
    slot128 = tl.load(slot_ptr + t)
    # slot_mapping == -1 marks padding tokens (e.g. the batch-pad row in a decode
    # CUDA graph captured at a larger batch than max_num_seqs). They have no cache
    # slot -- SKIP them. Writing them would compute page=-1//64=0, slot=-1%64=-1
    # (C truncation) -> a NEGATIVE byte offset -> out-of-bounds memory corruption.
    if slot128 >= 0:
        cols = tl.arange(0, HD)
        ik = tl.load(ik_ptr + t * HD + cols).to(tl.float32)
        amax = tl.max(tl.abs(ik), axis=0)
        scale = tl.where(amax > 0.0, amax / FP8MAX, 1.0)
        q = ik / scale
        q = tl.minimum(tl.maximum(q, -FP8MAX), FP8MAX)
        q_u8 = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        page = slot128 // PAGE
        slot = slot128 % PAGE
        tl.store(cache_u8_ptr + page * ROWB + slot * HD + cols, q_u8)
        tl.store(cache_f32_ptr + page * ROWF + DATAF + slot, scale)


@triton.jit
def _msa_qquant_kernel(q_ptr, quant_ptr, scale_ptr,
                       HD: tl.constexpr, FP8MAX: tl.constexpr):
    pid = tl.program_id(0)                    # = n*H + h over q [N, H, HD]
    cols = tl.arange(0, HD)
    q = tl.load(q_ptr + pid * HD + cols).to(tl.float32)
    amax = tl.max(tl.abs(q), axis=0)
    scale = tl.where(amax > 0.0, amax / FP8MAX, 1.0)
    qq = tl.minimum(tl.maximum(q / scale, -FP8MAX), FP8MAX)
    qq_u8 = qq.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(quant_ptr + pid * HD + cols, qq_u8)
    tl.store(scale_ptr + pid, scale)


def _triton_msa_qquant(q):
    """Bit-exact b12x quantize_msa_q_fp8 (per (token,head) amax/448 -> fp8+scale)
    in one kernel. ``q`` [N, H, 128] float -> (quant [N,H,128] fp8, scale [N,H]
    f32). Replaces b12x's ~10-launch PyTorch quantize chain."""
    N, H, HD = q.shape
    qc = q if q.is_contiguous() else q.contiguous()
    quant = torch.empty((N, H, HD), dtype=torch.float8_e4m3fn, device=q.device)
    scale = torch.empty((N, H), dtype=torch.float32, device=q.device)
    _msa_qquant_kernel[(N * H,)](
        qc, quant.view(torch.uint8), scale, HD, _FP8_MAX, num_warps=4)
    return quant, scale


@triton.jit
def _msa_qquant_nvfp4_kernel(q_ptr, e2m1_ptr, sfa_ptr,
                             HD: tl.constexpr, NBLK: tl.constexpr, BLK: tl.constexpr,
                             E2M1_MAX: tl.constexpr):
    pid = tl.program_id(0)                    # = n*H + h over q [N, H, HD]
    brow = tl.arange(0, NBLK)
    bcol = tl.arange(0, BLK)
    cols2d = brow[:, None] * BLK + bcol[None, :]
    q = tl.load(q_ptr + pid * HD + cols2d).to(tl.float32)
    amax = tl.max(tl.abs(q), axis=1)
    sr = tl.maximum(amax / E2M1_MAX, 1e-6)
    s_e4 = sr.to(tl.float8e4nv)
    v = q / s_e4.to(tl.float32)[:, None]
    av = tl.abs(v)
    idx = ((av > 0.25).to(tl.int32) + (av > 0.75).to(tl.int32)
           + (av > 1.25).to(tl.int32) + (av > 1.75).to(tl.int32)
           + (av > 2.5).to(tl.int32) + (av > 3.5).to(tl.int32)
           + (av > 5.0).to(tl.int32))
    codes = ((v < 0).to(tl.int32) << 3) | idx
    c3 = tl.reshape(codes, [NBLK, BLK // 2, 2])
    eo = tl.arange(0, 2)[None, None, :]
    lo = tl.sum(tl.where(eo == 0, c3, 0), axis=2)
    hi = tl.sum(tl.where(eo == 1, c3, 0), axis=2)
    packed = tl.reshape((lo | (hi << 4)).to(tl.uint8), [HD // 2])
    tl.store(e2m1_ptr + pid * (HD // 2) + tl.arange(0, HD // 2), packed)
    tl.store(sfa_ptr + pid * NBLK + tl.arange(0, NBLK), s_e4.to(tl.uint8, bitcast=True))


def _triton_msa_qquant_nvfp4(q):
    """Q -> nvfp4 (per-16 e2m1 + e4m3), SAME recipe as _cache_write_nvfp4_kernel.
    q [N, H, 128] float -> (q_e2m1 [N,H,64] u8, q_sfa [N,H,8] u8). Bit-exact vs
    the K cache write (tools/nvfp4-kv Phase 1)."""
    N, H, HD = q.shape
    qc = q if q.is_contiguous() else q.contiguous()
    e2m1 = torch.empty((N, H, HD // 2), dtype=torch.uint8, device=q.device)
    sfa = torch.empty((N, H, _NV_NBLK), dtype=torch.uint8, device=q.device)
    _msa_qquant_nvfp4_kernel[(N * H,)](
        qc, e2m1, sfa, HD, _NV_NBLK, _NVFP4_BLK, _NVFP4_E2M1_MAX, num_warps=4)
    return e2m1, sfa


@triton.jit
def _nvf4_verify_base_kernel(seq_ptr, base_ptr, cum_ptr,
                             BATCH, BLOCK_REQ: tl.constexpr):
    offs = tl.arange(0, BLOCK_REQ)
    seq = tl.load(seq_ptr + offs, mask=offs < BATCH, other=0).to(tl.int32)
    padded = ((seq + 127) // 128) * 128
    cs = tl.cumsum(padded, 0)
    tl.store(base_ptr + offs, cs - padded, mask=offs < BATCH)
    tl.store(cum_ptr + offs, cs, mask=offs < BATCH)


@triton.jit
def _nvf4_verify_query_meta_kernel(seq_ptr, base_ptr, ksg_ptr, keg_ptr,
                                   bb_ptr, qpos_ptr, Q_ROWS, Q_LEN: tl.constexpr,
                                   QCAP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_cap = offs < QCAP
    valid = offs < Q_ROWS
    req = offs // Q_LEN
    j = offs - req * Q_LEN
    base = tl.load(base_ptr + req, mask=valid, other=0).to(tl.int32)
    seq = tl.load(seq_ptr + req, mask=valid, other=0).to(tl.int32)
    end = base + (seq - Q_LEN) + j + 1
    end = tl.maximum(end, base)
    tl.store(ksg_ptr + offs, tl.where(valid, base, 0), mask=in_cap)
    tl.store(keg_ptr + offs, tl.where(valid, end, 0), mask=in_cap)
    tl.store(bb_ptr + offs, tl.where(valid, base // 128, 0), mask=in_cap)
    tl.store(qpos_ptr + offs, tl.where(valid, end - 1, -1), mask=in_cap)


@triton.jit
def _nvf4_verify_local_bounds_kernel(ksg_ptr, keg_ptr, ks_ptr, ke_ptr,
                                     Q_ROWS, CHUNK_START, SUPERTILE_K: tl.constexpr,
                                     QCAP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_cap = offs < QCAP
    valid = offs < Q_ROWS
    gs = tl.load(ksg_ptr + offs, mask=in_cap, other=0).to(tl.int32)
    ge = tl.load(keg_ptr + offs, mask=in_cap, other=0).to(tl.int32)
    cs = CHUNK_START
    ls = tl.minimum(tl.maximum(gs - cs, 0), SUPERTILE_K)
    le = tl.minimum(tl.maximum(ge - cs, 0), SUPERTILE_K)
    le = tl.maximum(le, ls)
    tl.store(ks_ptr + offs, tl.where(valid, ls, 0), mask=in_cap)
    tl.store(ke_ptr + offs, tl.where(valid, le, 0), mask=in_cap)


@triton.jit
def _nvf4_verify_gather_chunk_kernel(cache_ptr, block_table_ptr, seq_ptr, base_ptr, cum_ptr,
                                     ke2_ptr, ksf_ptr,
                                     CHUNK_START, BATCH, NPAGES, WIDTH128: tl.constexpr,
                                     BT_S0: tl.constexpr, BT_S1: tl.constexpr,
                                     REQ_LOG: tl.constexpr,
                                     ROWB: tl.constexpr, HALF: tl.constexpr,
                                     NBLK: tl.constexpr, DATAB: tl.constexpr):
    t = tl.program_id(0)
    gpos = CHUNK_START + t
    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), BATCH, tl.int32)
    for _ in range(REQ_LOG):
        mid = (lo + hi) // 2
        cm = tl.load(cum_ptr + mid, mask=mid < BATCH, other=2147483647).to(tl.int32)
        go_left = gpos < cm
        hi = tl.where(go_left, mid, hi)
        lo = tl.where(go_left, lo, mid + 1)
    req = lo
    req_safe = tl.minimum(req, BATCH - 1)
    base = tl.load(base_ptr + req_safe, mask=BATCH > 0, other=0).to(tl.int32)
    seq = tl.load(seq_ptr + req_safe, mask=BATCH > 0, other=0).to(tl.int32)
    pos = gpos - base
    valid = (req < BATCH) & (pos >= 0) & (pos < seq)
    b128 = tl.minimum(tl.maximum(pos // 128, 0), WIDTH128 - 1)
    sub = (pos % 128) // 64
    page = tl.load(block_table_ptr + req_safe * BT_S0 + b128 * BT_S1, mask=valid, other=0).to(tl.int64) * 2 + sub
    page = tl.minimum(tl.maximum(page, 0), NPAGES - 1)
    slot = tl.where(valid, pos % 64, 0)
    d = tl.arange(0, HALF)
    data = tl.load(cache_ptr + page * ROWB + slot * HALF + d, mask=valid, other=0)
    tl.store(ke2_ptr + t * HALF + d, data)
    s = tl.arange(0, NBLK)
    scales = tl.load(cache_ptr + page * ROWB + DATAB + slot * NBLK + s, mask=valid, other=0)
    tl.store(ksf_ptr + t * NBLK + s, scales)


@triton.jit
def _nvf4_topk_init_kernel(vals_ptr, inds_ptr, R, QCAP: tl.constexpr, TOPK: tl.constexpr):
    pid = tl.program_id(0)
    h = pid // R
    r = pid - h * R
    offs = tl.arange(0, TOPK)
    base = (h * QCAP + r) * TOPK + offs
    tl.store(vals_ptr + base, tl.full((TOPK,), float("-inf"), tl.float32))
    tl.store(inds_ptr + base, tl.full((TOPK,), 2147483647, tl.int32))


@triton.jit
def _nvf4_block_topk_merge_kernel(bs_ptr, carry_v_in, carry_i_in, carry_v_out, carry_i_out,
                                  qpos_ptr, R, QCAP: tl.constexpr, B: tl.constexpr,
                                  CHUNK_BLOCK_START, TOPK: tl.constexpr, BLOCK: tl.constexpr,
                                  MSA_BLOCK_TOKENS: tl.constexpr):
    pid = tl.program_id(0)
    h = pid // R
    r = pid - h * R
    bo = tl.arange(0, BLOCK)
    bm = bo < B
    bs_base = (h * QCAP + r) * B
    scores = tl.load(bs_ptr + bs_base + bo, mask=bm, other=float("-inf"))
    qpos = tl.load(qpos_ptr + r)
    forced_gb = qpos // MSA_BLOCK_TOKENS
    gb = CHUNK_BLOCK_START + bo
    scores = tl.where((gb == forced_gb) & bm & (qpos >= 0), float("inf"), scores)

    tk = tl.arange(0, TOPK)
    cand_v = tl.full((TOPK,), float("-inf"), tl.float32)
    cand_i = tl.full((TOPK,), 2147483647, tl.int32)
    for i in range(TOPK):
        mx = tl.max(scores, axis=0)
        am = tl.argmax(scores, axis=0).to(tl.int32)
        valid = mx > float("-inf")
        cand_v = tl.where(tk == i, mx, cand_v)
        cand_i = tl.where(tk == i, tl.where(valid, CHUNK_BLOCK_START + am, 2147483647), cand_i)
        scores = tl.where(bo == am, float("-inf"), scores)

    carry_base = (h * QCAP + r) * TOPK + tk
    cv = tl.load(carry_v_in + carry_base)
    ci = tl.load(carry_i_in + carry_base)
    nv = cand_v
    ni = cand_i
    for i in range(TOPK):
        cmx = tl.max(cv, axis=0)
        cpos = tl.argmax(cv, axis=0).to(tl.int32)
        nmx = tl.max(nv, axis=0)
        npos = tl.argmax(nv, axis=0).to(tl.int32)
        take_n = nmx > cmx
        oval = tl.where(take_n, nmx, cmx)
        nidx = tl.max(tl.where(tk == npos, ni, 0), axis=0)
        cidx = tl.max(tl.where(tk == cpos, ci, 0), axis=0)
        oidx = tl.where(take_n, nidx, cidx)
        tl.store(carry_v_out + (h * QCAP + r) * TOPK + i, oval)
        tl.store(carry_i_out + (h * QCAP + r) * TOPK + i,
                 tl.where(oval > float("-inf"), oidx, 2147483647))
        nv = tl.where((tk == npos) & take_n, float("-inf"), nv)
        cv = tl.where((tk == cpos) & (~take_n), float("-inf"), cv)


@triton.jit
def _nvf4_topk_finalize_kernel(vals_ptr, inds_ptr, bb_ptr, out_ptr,
                               R, QCAP: tl.constexpr, TOPK: tl.constexpr):
    pid = tl.program_id(0)
    h = pid // R
    r = pid - h * R
    offs = tl.arange(0, TOPK)
    base = (h * QCAP + r) * TOPK + offs
    vals = tl.load(vals_ptr + base)
    inds = tl.load(inds_ptr + base)
    bb = tl.load(bb_ptr + r)
    local = inds - bb
    masked = tl.where((vals > float("-inf")) & (local >= 0), local, 2147483647)
    srt = tl.sort(masked)
    tl.store(out_ptr + base, tl.where(srt == 2147483647, -1, srt).to(tl.int32))


def _triton_cache_write(cache64, index_key, slot_mapping):
    """Bit-exact write_packed_index_cache byte-scatter in one kernel."""
    T = int(index_key.shape[0])
    if T == 0:
        return
    ik = index_key.reshape(T, _INDEX_HEAD_DIM)
    if not ik.is_contiguous():
        ik = ik.contiguous()
    slot = slot_mapping.reshape(T).to(torch.int32)
    _cache_write_kernel[(T,)](
        ik, slot, cache64, cache64.view(torch.float32), T,
        _INDEX_HEAD_DIM, _PACK_PAGE, _PACK_ROW_BYTES, _ROW_F32, _DATA_F32,
        _FP8_MAX, num_warps=4)


def _cache_as_pages64(index_cache: torch.Tensor) -> torch.Tensor:
    """View the indexer side cache storage as page-64 packed rows.

    The vLLM cache is uint8 with shape ``[num_blocks, 128, 132]`` (== flat
    ``num_blocks*16896`` bytes).  Reinterpret as ``[num_blocks*2, 8448]``: two
    64-token b12x pages per 128-token vLLM block, page-major within each.
    """
    flat = index_cache.reshape(-1)
    num_pages64 = flat.numel() // _PACK_ROW_BYTES
    return flat[: num_pages64 * _PACK_ROW_BYTES].view(num_pages64, _PACK_ROW_BYTES)


def write_packed_index_cache(
    index_cache: torch.Tensor,
    index_key: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter normed/roped single-head index-K into the packed page-64 cache.

    ``index_key``: ``[T, 128]`` (bf16/fp16/fp32).  ``slot_mapping``: ``[T]``
    int, global 128-block slot (block*128 + tok_in_block) for the indexer group.
    Per-token amax/448 scale (b12x ``pack_index_k_cache_reference`` contract).
    """
    if index_key.numel() == 0:
        return
    cache64 = _cache_as_pages64(index_cache)
    if _USE_TRITON_CACHE_WRITE:
        # One fused Triton kernel: per-token amax/scale -> fp8 quant -> packed
        # page-64 scatter (q bytes + f32 scale). Bit-exact replacement for the
        # ~12-launch PyTorch byte-scatter; graph-captured in the decode path.
        _triton_cache_write(cache64, index_key, slot_mapping)
        return
    ik = index_key.reshape(-1, _INDEX_HEAD_DIM).to(torch.float32)
    amax = ik.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / _FP8_MAX, torch.ones_like(amax)).to(torch.float32)
    q = (ik / scale.unsqueeze(1)).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    sb = scale.contiguous().view(torch.uint8).view(-1, 4)
    page = torch.div(slot_mapping, _PACK_PAGE, rounding_mode="floor").to(torch.long)
    slot = (slot_mapping % _PACK_PAGE).to(torch.long)
    dcols = (slot.unsqueeze(1) * _INDEX_HEAD_DIM
             + torch.arange(_INDEX_HEAD_DIM, device=index_key.device))
    cache64[page.unsqueeze(1), dcols] = q
    scols = (_PACK_DATA_BYTES + slot.unsqueeze(1) * 4
             + torch.arange(4, device=index_key.device))
    cache64[page.unsqueeze(1), scols] = sb


@triton.jit(do_not_specialize=["T"])
def _cache_write_nvfp4_kernel(ik_ptr, slot_ptr, cache_ptr, T,
                              HD: tl.constexpr, PAGE: tl.constexpr, ROWB: tl.constexpr,
                              NBLK: tl.constexpr, BLK: tl.constexpr, DATAB: tl.constexpr,
                              E2M1_MAX: tl.constexpr):
    t = tl.program_id(0)
    slot128 = tl.load(slot_ptr + t)
    # slot_mapping == -1 marks padding tokens -- SKIP (see _cache_write_kernel).
    if slot128 >= 0:
        brow = tl.arange(0, NBLK)
        bcol = tl.arange(0, BLK)
        cols2d = brow[:, None] * BLK + bcol[None, :]      # [8,16] block-major == channel order
        ik = tl.load(ik_ptr + t * HD + cols2d).to(tl.float32)   # [8,16]
        amax = tl.max(tl.abs(ik), axis=1)                 # [8] per-16-block amax
        sr = tl.maximum(amax / E2M1_MAX, 1e-6)            # [8]
        s_e4 = sr.to(tl.float8e4nv)                       # [8] rounded e4m3 scale
        v = ik / s_e4.to(tl.float32)[:, None]            # divide by the ROUNDED scale (bit-exact)
        av = tl.abs(v)
        # e2m1 magnitude index via argmin-equivalent midpoint thresholds; strict
        # '>' reproduces torch.argmin first-occurrence tie-break exactly.
        idx = ((av > 0.25).to(tl.int32) + (av > 0.75).to(tl.int32)
               + (av > 1.25).to(tl.int32) + (av > 1.75).to(tl.int32)
               + (av > 2.5).to(tl.int32) + (av > 3.5).to(tl.int32)
               + (av > 5.0).to(tl.int32))
        codes = ((v < 0).to(tl.int32) << 3) | idx        # [8,16] e2m1 nibble 0..15
        c3 = tl.reshape(codes, [NBLK, BLK // 2, 2])      # [8,8,2] -> (even ch, odd ch)
        eo = tl.arange(0, 2)[None, None, :]
        lo = tl.sum(tl.where(eo == 0, c3, 0), axis=2)    # [8,8] even-channel nibble
        hi = tl.sum(tl.where(eo == 1, c3, 0), axis=2)    # [8,8] odd-channel nibble
        packed = tl.reshape((lo | (hi << 4)).to(tl.uint8), [HD // 2])  # [64] even=low, odd=high
        page = slot128 // PAGE
        slot = slot128 % PAGE
        tl.store(cache_ptr + page * ROWB + slot * (HD // 2) + tl.arange(0, HD // 2), packed)
        tl.store(cache_ptr + page * ROWB + DATAB + slot * NBLK + tl.arange(0, NBLK),
                 s_e4.to(tl.uint8, bitcast=True))


def _cache_as_pages64_nv(index_cache: torch.Tensor) -> torch.Tensor:
    """View the nvfp4 indexer cache storage as page-64 packed rows [npages, 4608]
    (two 64-token b12x pages per 128-token vLLM block, page-major within each)."""
    flat = index_cache.reshape(-1)
    npages = flat.numel() // _PACK_ROW_BYTES_NV
    return flat[: npages * _PACK_ROW_BYTES_NV].view(npages, _PACK_ROW_BYTES_NV)


def write_packed_index_cache_nvfp4(index_cache, index_key, slot_mapping) -> None:
    """S5 nvfp4 page-major block-quant writer (per-16-block e2m1 + e4m3). One
    fused Triton kernel, graph-capturable; byte-for-byte == the harness
    pack_index_k_cache_nvfp4 the kernel was gate-C validated against."""
    if index_key.numel() == 0:
        return
    cache_nv = _cache_as_pages64_nv(index_cache)
    ik = index_key.reshape(-1, _INDEX_HEAD_DIM)
    if not ik.is_contiguous():
        ik = ik.contiguous()
    T = int(ik.shape[0])
    slot = slot_mapping.reshape(T).to(torch.int32)
    _cache_write_nvfp4_kernel[(T,)](
        ik, slot, cache_nv, T, _INDEX_HEAD_DIM, _PACK_PAGE, _PACK_ROW_BYTES_NV,
        _NV_NBLK, _NVFP4_BLK, _PACK_DATA_BYTES_NV, _NVFP4_E2M1_MAX, num_warps=4)


def _build_page64_table(block_table_128: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``page64[i, 2c+r] = 2*block_table[i,c] + r`` into persistent ``out``."""
    rows, width = block_table_128.shape
    bt = block_table_128.to(torch.int32)
    view = out[:rows, : width * 2].view(rows, width, 2)
    view[:, :, 0] = bt * 2
    view[:, :, 1] = bt * 2 + 1
    return out[:rows, : width * 2]


class MiniMaxM3IndexerB12xImpl(MiniMaxM3IndexerImpl):
    """b12x MSA indexer: paged decode + contiguous prefill, q2k for both attns."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.index_head_dim == _INDEX_HEAD_DIM, (
            f"b12x indexer requires index_head_dim==128, got {self.index_head_dim}"
        )
        self.topk = min(int(self.topk_blocks), MSA_TOPK_BLOCKS)
        # S5: indexer-K cache quant. "nvfp4" -> per-16-block e2m1+e4m3 (72 B/tok,
        # bf16-expand decode scorer); else fp8 packed (132 B/tok). Single source
        # of truth for the caps.kv_quant + cache view + writer + prefill gather.
        self._index_kv_quant = (
            "nvfp4" if getattr(self, "indexer_kv_dtype", "bf16") == "nvfp4" else "none"
        )
        cfg = get_current_vllm_config()
        # QMAX is the prefill q-row capacity that PINS the one-time JIT compile.
        # It MUST be the server's max_num_batched_tokens -- the single knob we
        # allow to drive compilation -- never a hardcoded constant. (max_k_rows
        # below only sizes the k_quant scratch; the kernel compiles once for the
        # fixed supertile_k=32768 default and processes longer contexts as
        # multiple supertiles, with sizes passed as runtime k_start/k_end.)
        self._max_q = max(int(cfg.scheduler_config.max_num_batched_tokens), 1)
        self._max_seqs = max(int(cfg.scheduler_config.max_num_seqs), 1)
        self._max_blocks = (int(cfg.model_config.max_model_len) + 127) // 128
        # Native-nvf4 VERIFY capacity -- derived from SERVING config (fixed at
        # init; no hardcoded sizes, no runtime resizing).  Verify q rows are
        # bounded by max_num_batched_tokens AND max_num_seqs.  K is streamed in
        # fixed supertiles; max_chunks covers the worst-case concatenation of
        # max_verify_reqs requests each at max_model_len.
        _spec = getattr(cfg, "speculative_config", None)
        if isinstance(_spec, dict):
            _num_spec = int(_spec.get("num_speculative_tokens", 0) or 0)
        else:
            _num_spec = (
                int(getattr(_spec, "num_speculative_tokens", 0) or 0)
                if _spec is not None else 0
            )
        self._verify_q_len = max(_num_spec + 1, 1)
        self._max_verify_reqs = max(
            1,
            min(self._max_seqs, max(self._max_q // self._verify_q_len, 1)),
        )
        self._max_verify_q = self._max_verify_reqs * self._verify_q_len
        self._verify_supertile_k = ((_IDX_SUPERTILE_K + 127) // 128) * 128
        self._verify_max_k_per_req = int(self._max_blocks) * _MSA_BLOCK_TOKENS
        self._verify_max_total_k = self._max_verify_reqs * self._verify_max_k_per_req
        self._verify_max_chunks = max(
            1,
            (self._verify_max_total_k + self._verify_supertile_k - 1)
            // self._verify_supertile_k,
        )
        # Metadata-owner: first-constructed indexer == lowest sparse layer ==
        # first to execute the indexer in a forward (graph-safe owner).
        self._meta_owner = not _META_OWNER_TAKEN[0]
        _META_OWNER_TAKEN[0] = True
        # decode (paged) scratch + persistent metadata buffers, keyed by
        # (q_rows, width128, kv_quant): the full plan key (width/quant changes
        # must build a NEW plan, never reuse a stale batch-keyed one).
        self._dec: dict[tuple, dict] = {}
        # Persistent prefill q2k buffer: the main MSA extend scratch records the
        # q2k data_ptr on first bind and requires it FIXED across calls
        # (graph-safe contract). Reused/rewritten in place every prefill.
        self._q2k_prefill: torch.Tensor | None = None
        self._triton_warm = False
        # Native nvfp4 contiguous scorer (prefill/verify) -- only when the index-K
        # cache is nvfp4 AND not disabled. Replaces the dequant->fp8 gather with a
        # zero-conversion e2m1 byte gather + mxf4nvf4 scorer.
        self._index_native_nvf4 = (self._index_kv_quant == "nvfp4") and _USE_NATIVE_NVF4

    def _prewarm_triton(self, device) -> None:
        """Compile the fused select + cache-write Triton kernels eagerly (profiler
        pass) so neither JIT-compiles inside the decode CUDA-graph capture."""
        if self._triton_warm:
            return
        H = int(self.num_index_heads)
        blk = _next_pow2(self._max_blocks)
        bs = torch.zeros((H, 1, blk), dtype=torch.float32, device=device)
        qpos = torch.zeros((1,), dtype=torch.int32, device=device)
        out = torch.full((H, 1, self.topk), -1, dtype=torch.int32, device=device)
        _triton_msa_select(bs, qpos, self.topk, out=out, block=blk)
        qd = torch.zeros((1, H, _INDEX_HEAD_DIM), dtype=torch.bfloat16, device=device)
        _triton_msa_qquant(qd)
        ik = torch.zeros((1, _INDEX_HEAD_DIM), dtype=torch.bfloat16, device=device)
        cache = torch.zeros((4, _PACK_ROW_BYTES), dtype=torch.uint8, device=device)
        slot = torch.zeros((1,), dtype=torch.int32, device=device)
        _triton_cache_write(cache, ik, slot)
        if self._index_kv_quant == "nvfp4":
            # warm the nvfp4 page-major writer before decode graph capture
            cache_nv = torch.zeros((2, _PACK_ROW_BYTES_NV), dtype=torch.uint8, device=device)
            write_packed_index_cache_nvfp4(cache_nv, ik, slot)
        if self._index_native_nvf4:
            # Warm ALL native-nvf4 verify kernels + the CuTe scorer at the fixed
            # serving capacities, before any CUDA graph capture/inference.
            qcap = int(self._max_verify_q)
            max_reqs = int(self._max_verify_reqs)
            supertile_k = int(self._verify_supertile_k)
            vbuf = _nvf4_verify_buffers(
                qcap, supertile_k, H, self.topk, max_reqs, int(self._max_blocks), device)
            block_req = _next_pow2(max_reqs)
            req_log = max(1, (max_reqs + 1).bit_length())
            q_block = 128
            q_grid = ((qcap + q_block - 1) // q_block,)
            dummy_q = torch.zeros((qcap, H, _INDEX_HEAD_DIM), dtype=torch.bfloat16, device=device)
            vbuf["seq"].zero_()
            vbuf["block_table"].zero_()
            _nvf4_verify_base_kernel[(1,)](
                vbuf["seq"], vbuf["base"], vbuf["cum"], max_reqs, block_req, num_warps=1)
            _nvf4_verify_query_meta_kernel[q_grid](
                vbuf["seq"], vbuf["base"], vbuf["k_start_g"], vbuf["k_end_g"],
                vbuf["block_base"], vbuf["qpos"], qcap, int(self._verify_q_len),
                qcap, q_block, num_warps=4)
            _msa_qquant_nvfp4_kernel[(qcap * H,)](
                dummy_q, vbuf["q_e2m1"], vbuf["q_sfa"], _INDEX_HEAD_DIM, _NV_NBLK,
                _NVFP4_BLK, _NVFP4_E2M1_MAX, num_warps=4)
            _nvf4_topk_init_kernel[(H * qcap,)](
                vbuf["carry_v"][0], vbuf["carry_i"][0], qcap, qcap, self.topk, num_warps=1)
            _nvf4_verify_gather_chunk_kernel[(supertile_k,)](
                cache_nv.reshape(-1), vbuf["block_table"], vbuf["seq"],
                vbuf["base"], vbuf["cum"], vbuf["k_e2m1"], vbuf["k_sfb"],
                0, max_reqs, int(cache_nv.shape[0]), int(self._max_blocks),
                int(vbuf["block_table"].stride(0)), int(vbuf["block_table"].stride(1)),
                req_log, _PACK_ROW_BYTES_NV, _INDEX_HEAD_DIM // 2, _NV_NBLK,
                _PACK_DATA_BYTES_NV, num_warps=2)
            _nvf4_verify_local_bounds_kernel[q_grid](
                vbuf["k_start_g"], vbuf["k_end_g"], vbuf["k_start"], vbuf["k_end"],
                qcap, 0, supertile_k, qcap, q_block, num_warps=4)
            run_contiguous_block_scores_kernel_nvf4(
                vbuf["q_e2m1"], vbuf["q_sfa"], vbuf["weights"],
                vbuf["k_e2m1"], vbuf["k_sfb"], vbuf["k_start"], vbuf["k_end"],
                valid_q_rows=qcap, valid_k_rows=supertile_k,
                num_blocks_out=int(vbuf["chunk_blocks"]), block_scores=vbuf["bs"])
            _nvf4_block_topk_merge_kernel[(H * qcap,)](
                vbuf["bs"], vbuf["carry_v"][0], vbuf["carry_i"][0],
                vbuf["carry_v"][1], vbuf["carry_i"][1], vbuf["qpos"],
                qcap, qcap, int(vbuf["chunk_blocks"]), 0, self.topk,
                _next_pow2(int(vbuf["chunk_blocks"])), _MSA_BLOCK_TOKENS, num_warps=4)
            _nvf4_topk_finalize_kernel[(H * qcap,)](
                vbuf["carry_v"][1], vbuf["carry_i"][1], vbuf["block_base"],
                vbuf["q2k"], qcap, qcap, self.topk, num_warps=1)
        torch.cuda.synchronize()
        self._triton_warm = True

    def _cache_pages(self) -> torch.Tensor:
        """Packed page-64 view of the indexer side cache (nvfp4 4608 B/page when
        kv_quant=nvfp4, else fp8 8448 B/page)."""
        kv = self.index_cache.kv_cache
        return (_cache_as_pages64_nv(kv) if self._index_kv_quant == "nvfp4"
                else _cache_as_pages64(kv))

    # ---- decode (paged, graph-captured) ----
    def _decode_ctx(self, batch: int, width128: int, device) -> dict:
        key = (int(batch), int(width128), str(self._index_kv_quant))
        if key not in self._dec:
            if debug_paged_policy_enabled():
                import sys
                print(f"# b12x_decode_ctx site=indexer.decode key={key}",
                      file=sys.stderr, flush=True)
            width64 = width128 * 2
            caps = B12XIndexerPagedScratchCaps(
                device=device,
                num_q_heads=self.num_index_heads,
                num_idx_heads=self.num_index_heads,
                max_q_rows=batch,
                max_page_table_width=width64,
                topk=MSA_TOPK_BLOCKS,
                page_size=_PACK_PAGE,
                score_mode="msa",
                kv_quant=self._index_kv_quant,   # S5: caps -> binding -> kernel
            )
            plan = plan_indexer_paged_scratch(caps)
            (spec,) = plan.scratch_specs()
            scratch = _shared_idx_storage(
                (batch, width64, tuple(spec.shape), str(spec.dtype)), spec, device)
            # page64 + schedule now live in the shared per-forward metadata
            # (_fwd_meta); only the per-layer scratch + q2k output stay here.
            # Phase 5 fused-indexer scratch: per-row slab + self-resetting
            # arrival state (zero-initialized ONCE; the kernel restores it
            # every launch, so graph replays never need a memset).
            from b12x.attention.indexer.fused_msa_indexer import (
                msa_fused_scratch_shapes,
            )
            slab_shape, state_shape = msa_fused_scratch_shapes(
                batch, self.num_index_heads, width64)
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count
            self._dec[key] = dict(
                plan=plan,
                scratch=scratch,
                # Persistent (graph-safe) fused-select q2k output + the FIXED
                # pow2 block capacity that pins the select kernel's single
                # compile (>= ceil(max_model_len/128)).
                q2k=torch.full((self.num_index_heads, batch, self.topk), -1,
                               dtype=torch.int32, device=device),
                select_block=_next_pow2(self._max_blocks),
                fused_slab=torch.empty(slab_shape, dtype=torch.float32, device=device),
                fused_state=torch.zeros(state_shape, dtype=torch.int32, device=device),
                fused_ctas=max(1, min((width64 + 1) // 2, num_sms // max(1, batch))),
            )
        return self._dec[key]

    def _decode(self, index_query, d, nd) -> torch.Tensor:
        device = index_query.device
        batch = int(d.block_table.shape[0])
        width128 = int(d.block_table.shape[1])
        q_len = int(d.decode_query_len)
        # Verify-grouped one-pass fused indexer: q_len rows share ONE K stream
        # (each 64-token page is loaded once and scored for all q_len*heads
        # tile slots) => verify costs ~= decode (85 vs 129 us/layer @60k nvfp4,
        # 92 vs 182 fp8; tools/baselines/verify_microbench.py).
        fused_verify = (
            q_len > 1
            and _USE_FUSED_INDEXER
            and _USE_FUSED_VERIFY
            and q_len * self.num_index_heads <= 8
        )
        if q_len > 1 and self._index_native_nvf4 and not fused_verify:
            return self._verify_nvf4_native(index_query, d, nd)
        # Spec-decode VERIFY scores EACH of the q_len query positions per
        # request: q_rows = nd = batch*q_len. The decode scratch/plan is reused
        # at q_rows (not batch) -- tile_logits quantizes to 32-row tiles, so
        # 1..32 rows share ONE tile => verify costs the SAME scratch as decode
        # (no KV-budget hit); only nd*topk*4 bytes scale with nd. q_len==1 is
        # the original single-token decode path, unchanged.
        q_rows = nd
        ctx = self._decode_ctx(q_rows, width128, device)
        # Shared per-forward metadata (page64 + schedule + seqlens + qpos): built
        # by the owner only (or every layer when _USE_META_ONCE is off). The
        # block_table/seq_lens are identical across the 57 sparse layers, so the
        # owner's build (which runs first) is valid for all of them.
        meta = _fwd_meta(q_rows, width128 * 2, device)
        if self._meta_owner or not _USE_META_ONCE:
            if q_len == 1:
                _build_page64_table(d.block_table, meta["page64"])
                meta["seqlens"].copy_(d.seq_lens)
            elif fused_verify:
                # Grouped verify: page64 stays PER REQUEST (the kernel walks
                # one table per group); seqlens are PER ROW causal bounds --
                # query j of request i attends to (seq_len_i - q_len) + j + 1
                # tokens (row-major expansion identical to the multirow path).
                _build_page64_table(d.block_table, meta["page64"])
                base = (d.seq_lens.to(torch.int32) - q_len).view(batch, 1)
                jrange = torch.arange(
                    q_len, device=device, dtype=torch.int32).view(1, q_len)
                meta["seqlens"].copy_((base + jrange + 1).reshape(q_rows))
            else:
                # Verify expansion (per-request -> per-query). Query j of request
                # i attends causally to (seq_len_i - q_len) + j + 1 tokens --
                # bit-exact with minimax_m3_index_decode's
                #   query_pos = seq_len - decode_query_len + q_offset,
                #   kv_len = query_pos + 1 -- and shares request i's page table.
                # Row order is request-major (row = i*q_len + j), matching the
                # request-major iq layout. Broadcast writes into the persistent
                # buffers: no O(context) temporaries that could eat the KV budget.
                bt = d.block_table.to(torch.int32)               # [batch, width128]
                pv = meta["page64"][:q_rows, : width128 * 2].view(
                    batch, q_len, width128, 2)
                pv[:, :, :, 0] = (bt * 2).view(batch, 1, width128)
                pv[:, :, :, 1] = (bt * 2 + 1).view(batch, 1, width128)
                base = (d.seq_lens.to(torch.int32) - q_len).view(batch, 1)
                jrange = torch.arange(
                    q_len, device=device, dtype=torch.int32).view(1, q_len)
                meta["seqlens"].copy_((base + jrange + 1).reshape(q_rows))
            if not _USE_FUSED_INDEXER:
                # schedule + qpos metadata only feed the scorer+select chain;
                # the one-pass fused indexer derives both in-kernel.
                build_paged_mqa_schedule_metadata(
                    meta["seqlens"], _PACK_PAGE, out=meta["sched"])
                meta["qpos"].copy_(msa_decode_query_positions(meta["seqlens"]))
        page64 = meta["page64"]
        seqlens = meta["seqlens"]
        iq = index_query[:nd].view(q_rows, self.num_index_heads, _INDEX_HEAD_DIM)
        # Fused Triton query-quantize (one launch) replaces b12x's ~10-launch
        # PyTorch amax/scale/fp8 chain; bit-exact, graph-captured.
        if _USE_FUSED_INDEXER:
            # Phase 5/6 one-pass fused MSA indexer: score -> 128-token block
            # max -> local force -> top-k -> ascending q2k, ONE kernel.
            # Contract-exact vs the scorer+select chain for fp8
            # (tests/test_fused_msa_indexer); nvfp4 scores RAW bf16 q against
            # the expanded e2m1 pages (no q-quant launch, one fewer kernel and
            # strictly higher score precision than the old expand fallback).
            from b12x.attention.indexer.fused_msa_indexer import run_fused_msa_indexer
            from b12x.attention.indexer.kernel import (
                _split_index_k_cache_runtime_views,
            )
            _is_nv = self._index_kv_quant == "nvfp4"
            cache = self._cache_pages()
            views = ctx.get("fused_kviews")
            if views is None or int(views[0].shape[0]) != int(cache.shape[0]):
                # live packed-cache views (fp8: u8 [p,64,128] + strided f32
                # [p,64]; nvfp4: e2m1 [p,64,64] + e4m3 [p,64,8] u8), cached
                # because the cache tensor is persistent.
                views = ctx["fused_kviews"] = \
                    _split_index_k_cache_runtime_views(cache)
            kq, ks = views
            weights = ctx.get("fused_w")
            if weights is None:
                weights = ctx["fused_w"] = torch.full(
                    (q_rows, self.num_index_heads), MSA_SM_SCALE,
                    dtype=torch.float32, device=device)
            if _is_nv:
                q_in = iq.view(torch.uint8)          # raw bf16 bytes
            else:
                q_fp8, q_scale = (_triton_msa_qquant(iq) if _USE_TRITON_QQUANT
                                  else quantize_msa_q_fp8(iq))
                torch.mul(q_scale, MSA_SM_SCALE, out=weights[:q_rows])
                q_in = q_fp8.view(torch.uint8)
            pt_arg = page64[:q_rows]
            ctas_arg = ctx["fused_ctas"]
            ql_arg = 1
            if fused_verify:
                # groups = requests: one page-table row per group, more CTAs
                # per group (the K stream is walked once per REQUEST now)
                ql_arg = q_len
                pt_arg = page64[:batch]
                ctas_arg = ctx.get("fused_ctas_vfy")
                if ctas_arg is None:
                    num_sms = torch.cuda.get_device_properties(
                        device).multi_processor_count
                    ctas_arg = ctx["fused_ctas_vfy"] = max(
                        1, min((width128 * 2 + 1) // 2,
                               (2 * num_sms) // max(1, batch)))
            return run_fused_msa_indexer(
                q_bytes=q_in,
                weights=weights[:q_rows],
                k_quant_bytes=kq,
                k_scales=ks,
                real_page_table=pt_arg,
                seqlens=seqlens[:q_rows],
                num_heads=self.num_index_heads,
                topk=self.topk,
                out_indices=ctx["q2k"],
                ctas_per_group=ctas_arg,
                slab=ctx["fused_slab"],
                state=ctx["fused_state"],
                state_preinitialized=True,
                kv_quant=self._index_kv_quant if _is_nv else "none",
                q_len=ql_arg,
            )
        q_fp8, q_scale = (_triton_msa_qquant(iq) if _USE_TRITON_QQUANT
                          else quantize_msa_q_fp8(iq))
        binding = ctx["plan"].bind_msa(
            scratch=ctx["scratch"],
            real_page_table=page64,
            cache_seqlens_int32=seqlens,
            schedule_metadata=meta["sched"],
            topk=self.topk,
        )
        # Score + per-head PAGE_HEAD_MAX block max-pool: reuse b12x's fused CuTe
        # kernel (the heavy MSA-aware part, score_mode=MSA_BILINEAR). Replace
        # ONLY b12x's PyTorch msa_topk_blocks tail (~25 launch-bound kernels)
        # with one fused Triton select -> bit-exact q2k, ~26us/layer cheaper.
        if not _USE_TRITON_SELECT:
            return msa_q2k_indices_decode(
                q_fp8=q_fp8, q_scale=q_scale,
                index_k_cache=self._cache_pages(),
                binding=binding)
        block_scores = msa_paged_decode_block_scores(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=self._cache_pages(),
            binding=binding,
        )
        return _triton_msa_select(
            block_scores, meta["qpos"], self.topk, out=ctx["q2k"], block=ctx["select_block"])

    def _verify_nvf4_native(self, index_query, d, nd) -> torch.Tensor:
        """Native nvfp4 contiguous VERIFY indexer, supertiled over K blocks.

        Single-token decode remains on the existing paged path.  This path is
        only for spec verify (q_len>1) and never converts K through fp8/bf16:
        paged nvfp4 bytes are gathered into one fixed K supertile, Q is quantized
        to nvfp4, the native mxf4nvf4 block scorer emits chunk block scores, and
        a graph-safe block-topk carry folds chunks into final q2k.
        """
        device = index_query.device
        batch = int(d.block_table.shape[0])
        width128 = int(d.block_table.shape[1])
        q_len = int(d.decode_query_len)
        q_rows = int(nd)
        heads = int(self.num_index_heads)
        qcap = int(self._max_verify_q)
        max_reqs = int(self._max_verify_reqs)
        supertile_k = int(self._verify_supertile_k)
        max_chunks = int(self._verify_max_chunks)
        assert q_len <= int(self._verify_q_len), (
            f"verify q_len {q_len} exceeds configured capacity {self._verify_q_len}"
        )
        assert batch <= max_reqs, f"verify batch {batch} exceeds capacity {max_reqs}"
        assert q_rows <= qcap, f"verify q_rows {q_rows} exceeds capacity {qcap}"
        assert width128 <= int(self._max_blocks), (
            f"verify block-table width {width128} exceeds capacity {self._max_blocks}"
        )
        cache_nv = self._cache_pages()
        npages = int(cache_nv.shape[0])
        b = _nvf4_verify_buffers(
            qcap, supertile_k, heads, self.topk, max_reqs, int(self._max_blocks), device)
        # Stage metadata into fixed int32 buffers. copy_ handles dtype conversion
        # without allocating and is captured as normal device work.
        b["seq"][:batch].copy_(d.seq_lens)
        b["block_table"][:batch, :width128].copy_(d.block_table)
        block_req = _next_pow2(max_reqs)
        req_log = max(1, (max_reqs + 1).bit_length())
        q_block = 128
        q_grid = ((qcap + q_block - 1) // q_block,)

        _nvf4_verify_base_kernel[(1,)](
            b["seq"], b["base"], b["cum"], batch, block_req, num_warps=1)
        _nvf4_verify_query_meta_kernel[q_grid](
            b["seq"], b["base"], b["k_start_g"], b["k_end_g"], b["block_base"], b["qpos"],
            q_rows, q_len, qcap, q_block, num_warps=4)

        iq = index_query[:q_rows].view(q_rows, heads, _INDEX_HEAD_DIM)
        _msa_qquant_nvfp4_kernel[(q_rows * heads,)](
            iq, b["q_e2m1"], b["q_sfa"], _INDEX_HEAD_DIM, _NV_NBLK,
            _NVFP4_BLK, _NVFP4_E2M1_MAX, num_warps=4)

        # Initialize carry[0] to empty top-k for every padded row/head.
        _nvf4_topk_init_kernel[(heads * qcap,)](
            b["carry_v"][0], b["carry_i"][0], qcap, qcap, self.topk, num_warps=1)
        chunk_blocks = int(b["chunk_blocks"])
        block_pow2 = _next_pow2(chunk_blocks)
        for chunk_idx in range(max_chunks):
            chunk_start = int(chunk_idx) * supertile_k
            _nvf4_verify_gather_chunk_kernel[(supertile_k,)](
                cache_nv.reshape(-1), b["block_table"], b["seq"], b["base"], b["cum"],
                b["k_e2m1"], b["k_sfb"], chunk_start, batch, npages,
                int(self._max_blocks), int(b["block_table"].stride(0)),
                int(b["block_table"].stride(1)), req_log, _PACK_ROW_BYTES_NV,
                _INDEX_HEAD_DIM // 2, _NV_NBLK, _PACK_DATA_BYTES_NV, num_warps=2)
            _nvf4_verify_local_bounds_kernel[q_grid](
                b["k_start_g"], b["k_end_g"], b["k_start"], b["k_end"],
                q_rows, chunk_start, supertile_k, qcap, q_block, num_warps=4)
            run_contiguous_block_scores_kernel_nvf4(
                b["q_e2m1"], b["q_sfa"], b["weights"], b["k_e2m1"], b["k_sfb"],
                b["k_start"], b["k_end"], valid_q_rows=qcap,
                valid_k_rows=supertile_k, num_blocks_out=chunk_blocks,
                block_scores=b["bs"])
            in_slot = chunk_idx & 1
            out_slot = 1 - in_slot
            _nvf4_block_topk_merge_kernel[(heads * qcap,)](
                b["bs"], b["carry_v"][in_slot], b["carry_i"][in_slot],
                b["carry_v"][out_slot], b["carry_i"][out_slot], b["qpos"],
                qcap, qcap, chunk_blocks, chunk_start // _MSA_BLOCK_TOKENS,
                self.topk, block_pow2, _MSA_BLOCK_TOKENS, num_warps=4)
        final_slot = max_chunks & 1
        _nvf4_topk_finalize_kernel[(heads * qcap,)](
            b["carry_v"][final_slot], b["carry_i"][final_slot], b["block_base"],
            b["q2k"], qcap, qcap, self.topk, num_warps=1)
        return b["q2k"]

    # ---- prefill (contiguous gather, eager) ----
    def _gather_request_k(self, cache64, block_table_row, cache_len) -> tuple:
        """Gather a request's contiguous index-K (fp8) + scale from packed cache.

        token pos -> 128-block col c=pos//128, sub-page r=(pos%128)//64, so the
        64-page id is block_table_row[c]*2 + r and the in-page slot is pos%64.
        """
        device = cache64.device
        pos = torch.arange(cache_len, device=device)
        c = torch.div(pos, 128, rounding_mode="floor")
        r = torch.div(pos % 128, _PACK_PAGE, rounding_mode="floor")
        page64 = (block_table_row[c].to(torch.long) * 2 + r).to(torch.long)
        slot = (pos % _PACK_PAGE).to(torch.long)
        dcols = slot.unsqueeze(1) * _INDEX_HEAD_DIM + torch.arange(_INDEX_HEAD_DIM, device=device)
        k_quant = cache64[page64.unsqueeze(1), dcols].view(torch.float8_e4m3fn)  # [L,128]
        scols = _PACK_DATA_BYTES + slot.unsqueeze(1) * 4 + torch.arange(4, device=device)
        k_scale = cache64[page64.unsqueeze(1), scols].reshape(cache_len, 4).view(torch.float32).reshape(cache_len)
        return k_quant.contiguous(), k_scale.contiguous()

    def _gather_request_k_nvfp4(self, cache_nv, block_table_row, cache_len) -> tuple:
        """S5 MVP: gather a request's index-K from the paged nvfp4 cache, dequant
        (e2m1 x per-16 e4m3) -> f32, then re-quantize per-token fp8 for the
        unchanged fp8 contiguous prefill scorer. Same page/slot map as the fp8
        gather; only the byte layout + dequant differ."""
        device = cache_nv.device
        pos = torch.arange(cache_len, device=device)
        c = torch.div(pos, 128, rounding_mode="floor")
        r = torch.div(pos % 128, _PACK_PAGE, rounding_mode="floor")
        page64 = (block_table_row[c].to(torch.long) * 2 + r).to(torch.long)
        slot = (pos % _PACK_PAGE).to(torch.long)
        half = _INDEX_HEAD_DIM // 2
        dcols = slot.unsqueeze(1) * half + torch.arange(half, device=device)
        packed = cache_nv[page64.unsqueeze(1), dcols]                 # [L,64] uint8
        codes = torch.empty((cache_len, _INDEX_HEAD_DIM), dtype=torch.uint8, device=device)
        codes[:, 0::2] = packed & 0xF
        codes[:, 1::2] = (packed >> 4) & 0xF
        scols = _PACK_DATA_BYTES_NV + slot.unsqueeze(1) * _NV_NBLK + torch.arange(_NV_NBLK, device=device)
        sc = cache_nv[page64.unsqueeze(1), scols].reshape(cache_len, _NV_NBLK).view(torch.float8_e4m3fn).float()
        vals = _e2m1_decode_torch(codes).reshape(cache_len, _NV_NBLK, _NVFP4_BLK)
        k = (vals * sc.unsqueeze(-1)).reshape(cache_len, _INDEX_HEAD_DIM)        # [L,128] dequant
        amax = k.abs().amax(dim=1)
        kscale = torch.where(amax > 0, amax / _FP8_MAX, torch.ones_like(amax)).float()
        kq = (k / kscale.unsqueeze(1)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        return kq.contiguous(), kscale.contiguous()

    def _prefill(self, index_query, p, nd, nt) -> torch.Tensor:
        device = index_query.device
        cache64 = self._cache_pages()
        cu = p.cu_seqlens_q  # [num_prefills+1], rebased to 0 (query offsets within prefill span)
        seq_lens = p.seq_lens.to(torch.int64)      # total KV length per request
        block_table = p.block_table
        num_prefills = int(seq_lens.shape[0])
        cu_cpu = cu.detach().cpu().tolist()
        seq_cpu = seq_lens.detach().cpu().tolist()

        q_all = index_query[nd:nt].view(-1, self.num_index_heads, _INDEX_HEAD_DIM)
        total_q = q_all.shape[0]
        # concat per-request K; per query-row k_start/k_end into the concat.
        k_quants, k_scales = [], []
        k_start = torch.zeros(total_q, dtype=torch.int32, device=device)
        k_end = torch.zeros(total_q, dtype=torch.int32, device=device)
        base = 0
        for r in range(num_prefills):
            cache_len = int(seq_cpu[r])
            # S5 MVP: nvfp4 gather dequants the paged nvfp4 cache to bf16 then
            # re-quantizes to fp8 for the (unchanged) fp8 contiguous prefill
            # scorer. Correctness-equivalent for selection (gate B); Phase 3 (P1)
            # makes prefill read the nvfp4 paged cache views-direct like decode.
            kq, ks = (self._gather_request_k_nvfp4(cache64, block_table[r], cache_len)
                      if self._index_kv_quant == "nvfp4"
                      else self._gather_request_k(cache64, block_table[r], cache_len))
            # MSA contiguous requires each request's k_start to be 128-aligned,
            # so pad each request's K up to a 128-multiple in the concat.
            padded = ((cache_len + 127) // 128) * 128
            if padded > cache_len:
                kq = torch.cat([kq, kq.new_zeros(padded - cache_len, _INDEX_HEAD_DIM)], 0)
                ks = torch.cat([ks, ks.new_zeros(padded - cache_len)], 0)
            k_quants.append(kq)
            k_scales.append(ks)
            q0, q1 = int(cu_cpu[r]), int(cu_cpu[r + 1])
            qo_len = q1 - q0
            # right-aligned causal: query row i (0..qo_len-1) sees [0, cache_len-qo_len+i+1)
            rows = torch.arange(qo_len, device=device, dtype=torch.int32)
            k_start[q0:q1] = base
            k_end[q0:q1] = base + (cache_len - qo_len) + rows + 1
            base += padded
        kq_cat = torch.cat(k_quants, 0).contiguous()   # [base, 128] fp8
        ks_cat = torch.cat(k_scales, 0).contiguous()   # [base] f32
        k_rows = int(base)

        # COMPILE-ONCE-AT-MAX MSA PREFILL (b12x contiguous scratch, proven warm).
        # Bind the shared max-capacity contiguous scratch and pass FIXED q/k
        # shapes (max_num_batched_tokens x max_model_len); only k_start/k_end
        # vary per request, so b12x's MSA prefill scorer + tiled top-k compile
        # EXACTLY once and then serve any context warm (~1-5ms). The gathered
        # context K goes into the scratch's OWN k_quant/k_scale prefix (strict
        # MSA binding: K must BE the scratch buffers); rows past k_rows are
        # masked by each query row's causal k_end.
        heads = int(self.num_index_heads)
        qmax = int(self._max_q)
        kmax = int(self._max_blocks) * _MSA_BLOCK_TOKENS
        assert total_q <= qmax, f"prefill q_rows {total_q} exceeds capacity {qmax}"
        assert k_rows <= kmax, f"prefill context {k_rows} exceeds capacity {kmax}"
        cctx = _contiguous_ctx(qmax, kmax, heads, device)
        # k_start/k_end at the fixed q capacity (padded rows keep k_end=0 -> no
        # blocks selected).
        ks_q = torch.zeros(qmax, dtype=torch.int32, device=device)
        ke_q = torch.zeros(qmax, dtype=torch.int32, device=device)
        ks_q[:total_q].copy_(k_start)
        ke_q[:total_q].copy_(k_end)
        binding = cctx["plan"].bind_msa(
            scratch=cctx["scratch"], k_start=ks_q, k_end=ke_q, topk=self.topk)
        sv = binding.scratch
        sv.k_quant[:k_rows].copy_(kq_cat)
        sv.k_scale[:k_rows].copy_(ks_cat)
        # Q at the fixed capacity (pad to qmax rows -> one compile), fp8.
        if total_q < qmax:
            q_all = torch.cat(
                [q_all, q_all.new_zeros(qmax - total_q, heads, _INDEX_HEAD_DIM)], 0)
        q_fp8, q_scale = quantize_msa_q_fp8(q_all)
        q2k = msa_q2k_indices_prefill(
            q_fp8=q_fp8, q_scale=q_scale,
            kv_fp8=(sv.k_quant, sv.k_scale), binding=binding)
        # Copy the valid rows into the per-layer persistent buffer: the main MSA
        # extend captures THIS pointer and needs it fixed, decoupled from the
        # shared scratch q2k (overwritten by the next layer's indexer).
        if self._q2k_prefill is None or int(self._q2k_prefill.shape[0]) != heads \
                or int(self._q2k_prefill.shape[1]) < qmax:
            self._q2k_prefill = torch.full(
                (heads, qmax, MSA_TOPK_BLOCKS), -1, dtype=torch.int32, device=device)
        self._q2k_prefill[:, :total_q, :].copy_(q2k[:, :total_q, :])
        return self._q2k_prefill

    def forward(self, index_query: torch.Tensor):
        attn_metadata = get_forward_context().attn_metadata
        # Allocate the fixed-capacity prefill scratch eagerly -- including on the
        # memory profiler's dummy forward (attn_metadata is None there) -- so its
        # ~315MB is counted in peak memory and the KV cache is sized AROUND it,
        # instead of being claimed lazily on the first real prefill (OOM risk).
        _contiguous_ctx(self._max_q, int(self._max_blocks) * _MSA_BLOCK_TOKENS,
                        int(self.num_index_heads), index_query.device)
        # Compile the fused Triton select + cache-write kernels NOW (eager,
        # profiler pass) so they are warm before the decode CUDA-graph capture
        # -- a JIT compile inside capture would fail.
        self._prewarm_triton(index_query.device)
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MiniMaxM3IndexerMetadata)
        nt = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        iq = index_query[:nt].view(-1, self.num_index_heads, _INDEX_HEAD_DIM)

        decode_q2k = None
        prefill_q2k = None
        if index_md.num_decodes > 0:
            decode_q2k = self._decode(iq.view(-1, self.num_index_heads * _INDEX_HEAD_DIM), index_md.decode, nd)
        if index_md.num_prefills > 0:
            prefill_q2k = self._prefill(
                iq.view(-1, self.num_index_heads * _INDEX_HEAD_DIM), index_md.prefill, nd, nt
            )
        return decode_q2k, prefill_q2k
