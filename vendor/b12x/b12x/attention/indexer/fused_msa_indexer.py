"""One-pass fused MSA indexer: score -> 128-token block max -> top-k -> q2k.

Phase 5 of the decode-throughput plan. One kernel replaces the MSA decode
indexer chain (scheduled PAGE_HEAD_MAX scorer + pair-max + local-force + topk
+ sort + pad tail): per-token scores never leave SMEM; per-head 128-token
block maxima accumulate in an SMEM array (blocks fit trivially: <=512 blocks
@ 64k tokens); the final stage force-includes the local causal block,
iteratively selects the per-head top-k blocks, sorts them ascending and
writes the ``-1``-padded ``q2k_indices`` -- the exact MSA contract of
``msa_q2k_indices_decode`` / ``msa_topk_blocks``.

Reuse (kept in lock-step with the source kernels):
- fp8 page load+permute: ``_load_permute_k_page_g2s`` (fused_indexer)
- per-head page-max MMA epilogue: ``_compute_mxfp8_tile_head_token_max``
  (kernel.py -- the same primitive the scheduled PAGE_HEAD_MAX path runs)
- cross-CTA last-arrival merge + self-resetting state: the serial relay arm
  of ``SparseNSAFusedIndexerKernel`` (fused_indexer)

Layout contract: one CTA group per q row; ``ctas_per_group`` CTAs split the
row's pages in contiguous EVEN-aligned slices (so a 128-token block never
straddles CTAs). Each CTA folds its slice's block maxima into SMEM, publishes
them to a per-row global slab, and the last-arriving CTA runs the selection.
``ctas_per_group == 1`` selects straight from SMEM (no slab, no atomics).
"""

from __future__ import annotations

from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Float32, Int32, Int64

from cutlass import Uint32

from b12x.cute.fp4 import (
    atomic_add_global_i32,
    bf16_mma_m16n8k16_f32,
    f16x2_to_f32x2,
    fp4_decode_2,
    fp8_e4m3_to_f32,
    get_ptr_as_int64,
    ld_shared_u32,
    ldmatrix_m8n8x4_b16,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_shared_u32,
    threadfence,
)
from b12x.cute.utils import current_cuda_stream

from b12x.attention._cute import ops as attention_ops
from b12x.attention.indexer.fused_indexer import (
    _launch_fused,
    _load_permute_k_page_g2s,
    _to_kernel_tensor,
)
from b12x.attention.indexer.kernel import (
    _INDEX_HEAD_DIM,
    _PAGE_SIZE,
    _PAGED_Q_HEAD_TILE,
    _PAGED_TOKENS_PER_GROUP,
    _WARP_THREADS,
    _advance_offset_by_column_128b_2,
    _advance_offset_by_row_128b,
    _pack_q_mxfp8_reg,
    _permuted_offset_128b,
    _reduce_quad_max,
    _smem_addr_from_b128_offset,
)
from b12x.cute.fp4 import (
    frag_layout_swizzle_16b_to_8b,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
)

# 512 threads => 2 resident CTAs/SM (SM120 1536 thr/SM caps 1024-thread CTAs
# at ONE resident block): twice the CTA-level parallelism halves each CTA's
# serial page chain, which dominates this latency-bound loop (Phase 7 tuning:
# 1024thr/48ctas 74.8us -> 512thr/96ctas measured below @60k fp8).
_THREADS = 512
_TOKEN_GROUPS = _PAGE_SIZE // _PAGED_TOKENS_PER_GROUP  # 8: one pass covers the page
_SCORE_THREADS = _TOKEN_GROUPS * _WARP_THREADS  # 8 warps run the MMA
_INT32_MAX = 2147483647
_NEG_INF = float("-inf")
MSA_BLOCK_PAGES = 2  # 128-token MSA block = 2 x 64-token index pages


@cute.jit
def _load_expand_k_page_nvfp4_wide(
    k_e2m1_bytes: cute.Tensor,   # [pages, 64, 64] uint8 e2m1 (2 ch/byte)
    k_scale_bytes: cute.Tensor,  # [pages, 64, 8] uint8 e4m3 per-16 scales
    page_id: Int32,
    s_k_bf16_base: Int32,        # bf16 [64,128] row-major k-stage (SMEM addr)
    tx: Int32,
    stride: Int32,
):
    """1024-thread port of kernel.py's _load_and_expand_index_k_page_nvfp4
    (same math: e2m1 pair -> f32 -> x per-16 e4m3 scale -> packed bf16x2)."""
    linear = tx
    total = Int32(_PAGE_SIZE * (_INDEX_HEAD_DIM // 2))
    while linear < total:
        tok = linear // Int32(_INDEX_HEAD_DIM // 2)
        j = linear - tok * Int32(_INDEX_HEAD_DIM // 2)
        byte = Uint32(k_e2m1_bytes[page_id, tok, j])
        sbyte = Uint32(k_scale_bytes[page_id, tok, j // Int32(8)])
        f0, f1 = f16x2_to_f32x2(fp4_decode_2(byte))
        sf = fp8_e4m3_to_f32(sbyte)
        bf = pack_f32x2_to_bfloat2(f0 * sf, f1 * sf)
        st_shared_u32(
            s_k_bf16_base + (tok * Int32(_INDEX_HEAD_DIM) + j * Int32(2)) * Int32(2), bf
        )
        linear += stride


@cute.jit
def _compute_bf16_tile_head_token_max(
    s_q_bf16_base: Int32,   # bf16 [16,128] row-major q-stage (SMEM addr)
    s_w: cute.Tensor,
    num_heads: Int32,       # active tile rows (slots: q_len * heads at verify)
    k_bf16_base: Int32,     # bf16 [64,128] row-major k-stage (SMEM addr)
    token_base: Int32,
    lane: Int32,
    s_valid: cute.Tensor,   # i32 [16] per-slot valid tokens in this page
    s_page_partials: cute.Tensor,
    token_group: Int32,
):
    """bf16 port of _compute_mxfp8_tile_head_token_max: the m16n8k16 MMA loop
    of _compute_bf16_tile_partials with the per-head PAGE-MAX epilogue (no
    per-token k scale -- the nvfp4 block scale is folded during expand)."""
    group_id = lane // Int32(4)
    thread_id_in_group = lane % Int32(4)
    col_pair_base = thread_id_in_group * Int32(2)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    b_token = token_base + group_id
    q0_acc = Float32(0.0)
    q1_acc = Float32(0.0)
    q2_acc = Float32(0.0)
    q3_acc = Float32(0.0)
    for kk in cutlass.range_constexpr(_INDEX_HEAD_DIM // 16):
        k0 = Int32(kk * 16)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
            s_q_bf16_base + (a_row * Int32(_INDEX_HEAD_DIM) + a_col + k0) * Int32(2)
        )
        b0 = ld_shared_u32(
            k_bf16_base + (b_token * Int32(_INDEX_HEAD_DIM) + k0 + col_pair_base) * Int32(2)
        )
        b1 = ld_shared_u32(
            k_bf16_base
            + (b_token * Int32(_INDEX_HEAD_DIM) + k0 + Int32(8) + col_pair_base) * Int32(2)
        )
        q0_acc, q1_acc, q2_acc, q3_acc = bf16_mma_m16n8k16_f32(
            q0_acc, q1_acc, q2_acc, q3_acc, a0, a1, a2, a3, b0, b1
        )

    head0 = group_id
    head1 = head0 + Int32(8)
    w0 = Float32(0.0)
    w1 = Float32(0.0)
    if head0 < num_heads:
        w0 = Float32(s_w[head0])
    if head1 < num_heads:
        w1 = Float32(s_w[head1])
    col0 = token_base + col_pair_base
    col1 = col0 + Int32(1)
    v0 = Int32(0)
    v1 = Int32(0)
    if head0 < num_heads:
        v0 = Int32(s_valid[head0])
    if head1 < num_heads:
        v1 = Int32(s_valid[head1])
    max0 = Float32(_NEG_INF)
    max1 = Float32(_NEG_INF)
    if (head0 < num_heads) and (col0 < v0):
        max0 = attention_ops.fmax(max0, Float32(q0_acc * w0))
    if (head0 < num_heads) and (col1 < v0):
        max0 = attention_ops.fmax(max0, Float32(q1_acc * w0))
    if (head1 < num_heads) and (col0 < v1):
        max1 = attention_ops.fmax(max1, Float32(q2_acc * w1))
    if (head1 < num_heads) and (col1 < v1):
        max1 = attention_ops.fmax(max1, Float32(q3_acc * w1))
    max0 = _reduce_quad_max(max0)
    max1 = _reduce_quad_max(max1)
    if thread_id_in_group == Int32(0):
        if head0 < num_heads:
            s_page_partials[token_group, head0] = attention_ops.fmax(
                Float32(s_page_partials[token_group, head0]), max0
            )
        if head1 < num_heads:
            s_page_partials[token_group, head1] = attention_ops.fmax(
                Float32(s_page_partials[token_group, head1]), max1
            )


@cute.jit
def _compute_mxfp8_tile_slot_token_max(
    s_q_bytes: cute.Tensor,
    s_w: cute.Tensor,
    num_heads: Int32,       # active tile rows (slots: q_len * heads at verify)
    k_perm_base_addr: Int32,
    token_base: Int32,
    lane: Int32,
    s_scale: cute.Tensor,
    s_valid: cute.Tensor,   # i32 [16] per-slot valid tokens in this page
    s_page_partials: cute.Tensor,
    token_group: Int32,
):
    """Per-slot-valid clone of kernel.py's _compute_mxfp8_tile_head_token_max
    (head_tile_base pinned to 0; the scalar valid_slots becomes s_valid[slot]
    so verify slots mask their own causal boundary). Kept in lock-step with
    the source epilogue."""
    group_id = lane // Int32(4)
    thread_id_in_group = lane % Int32(4)
    col_pair_base = thread_id_in_group * Int32(2)
    q0_acc = Float32(0.0)
    q1_acc = Float32(0.0)
    q2_acc = Float32(0.0)
    q3_acc = Float32(0.0)
    k_offset = _permuted_offset_128b(
        token_base + Int32(8) * (lane // Int32(16)) + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        Int32(_INDEX_HEAD_DIM // 16),
    )
    for mma_pair in cutlass.range_constexpr(_INDEX_HEAD_DIM // 32):
        pair_base = Int32(mma_pair * 32) + col_pair_base
        q0 = _pack_q_mxfp8_reg(s_q_bytes, group_id, pair_base)
        q1 = _pack_q_mxfp8_reg(s_q_bytes, group_id + Int32(8), pair_base)
        q2 = _pack_q_mxfp8_reg(s_q_bytes, group_id, pair_base + Int32(16))
        q3 = _pack_q_mxfp8_reg(s_q_bytes, group_id + Int32(8), pair_base + Int32(16))
        b0_k0, _ = ldmatrix_m8n8x4_left_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k1, _ = ldmatrix_m8n8x4_right_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
        k_offset_cur = _advance_offset_by_row_128b(
            k_offset,
            Int32(16),
            Int32(_INDEX_HEAD_DIM // 16),
        )
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            q0_acc, q1_acc, q2_acc, q3_acc,
            q0, q1, q2, q3,
            b0_k0, b0_k1,
            Uint32(0x7F7F7F7F), Uint32(0x7F7F7F7F),
        )
        q0_acc = d0
        q1_acc = d1
        q2_acc = d2
        q3_acc = d3
        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            16 * (_INDEX_HEAD_DIM // 16)
        )

    head0 = group_id
    head1 = head0 + Int32(8)
    w0 = Float32(0.0)
    w1 = Float32(0.0)
    if head0 < num_heads:
        w0 = Float32(s_w[head0])
    if head1 < num_heads:
        w1 = Float32(s_w[head1])
    col0 = token_base + col_pair_base
    col1 = col0 + Int32(1)
    v0 = Int32(0)
    v1 = Int32(0)
    if head0 < num_heads:
        v0 = Int32(s_valid[head0])
    if head1 < num_heads:
        v1 = Int32(s_valid[head1])
    max0 = Float32(-Float32.inf)
    max1 = Float32(-Float32.inf)
    if (head0 < num_heads) and (col0 < v0):
        max0 = attention_ops.fmax(max0, Float32(q0_acc * w0 * s_scale[col0]))
    if (head0 < num_heads) and (col1 < v0):
        max0 = attention_ops.fmax(max0, Float32(q1_acc * w0 * s_scale[col1]))
    if (head1 < num_heads) and (col0 < v1):
        max1 = attention_ops.fmax(max1, Float32(q2_acc * w1 * s_scale[col0]))
    if (head1 < num_heads) and (col1 < v1):
        max1 = attention_ops.fmax(max1, Float32(q3_acc * w1 * s_scale[col1]))
    max0 = _reduce_quad_max(max0)
    max1 = _reduce_quad_max(max1)
    if thread_id_in_group == Int32(0):
        if head0 < num_heads:
            s_page_partials[token_group, head0] = attention_ops.fmax(
                Float32(s_page_partials[token_group, head0]), max0
            )
        if head1 < num_heads:
            s_page_partials[token_group, head1] = attention_ops.fmax(
                Float32(s_page_partials[token_group, head1]), max1
            )


def _msa_shared_storage_cls(max_blocks: int, heads_cap: int, topk: int, kv_quant: str = "none"):
    class SharedStorage:
        pass

    def _i32(n):
        return cute.struct.Align[cute.struct.MemRange[cutlass.Int32, int(n)], 128]

    def _f32(n):
        return cute.struct.Align[cute.struct.MemRange[cutlass.Float32, int(n)], 128]

    # nvfp4 stages q and K as bf16 (2 bytes/elem); fp8 stages raw fp8 bytes +
    # f32 per-token scales. Slots are sized for the selected variant.
    q_stage_bytes = _PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM * (2 if kv_quant == "nvfp4" else 1)
    k_stage_bytes = _PAGE_SIZE * _INDEX_HEAD_DIM * (2 if kv_quant == "nvfp4" else 1)
    SharedStorage.__annotations__ = {
        "q_bytes": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, q_stage_bytes], 16
        ],
        "weights": _f32(_PAGED_Q_HEAD_TILE),
        "k_page_perm": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, k_stage_bytes], 1024
        ],
        "scales": _f32(_PAGE_SIZE),
        # per-(token_group, head) page partial maxima (head padded to the tile)
        "page_partials": _f32(_TOKEN_GROUPS * _PAGED_Q_HEAD_TILE),
        # per-head running 128-token block maxima (whole row fits in SMEM)
        "block_scores": _f32(int(heads_cap) * int(max_blocks)),
        # selection scratch: per-head 32-lane partial argmax + selected ids
        "red_v": _f32(int(heads_cap) * _WARP_THREADS),
        "red_i": _i32(int(heads_cap) * _WARP_THREADS),
        "sel": _i32(int(heads_cap) * int(topk)),
        "relay": _i32(1),
        # per-slot valid token count within the current page (verify rows
        # mask their own causal boundary; decode: all slots identical)
        "valid": _i32(_PAGED_Q_HEAD_TILE),
    }
    return cute.struct(SharedStorage)


class MSAFusedIndexerKernel:
    """One-pass MSA block indexer (fp8 index-K), single kernel per step."""

    def __init__(
        self,
        *,
        num_heads_static: int,
        topk: int,
        ctas_per_group: int,
        max_blocks: int,
        k_quant_page_stride: int,
        kv_quant: str = "none",
        q_len: int = 1,
    ):
        self.kv_quant = str(kv_quant)
        if self.kv_quant not in ("none", "nvfp4"):
            raise ValueError(f"kv_quant must be 'none' (fp8) or 'nvfp4', got {kv_quant}")
        self.num_heads_static = int(num_heads_static)
        if not (1 <= self.num_heads_static <= _PAGED_Q_HEAD_TILE):
            raise ValueError(
                f"MSA fused indexer supports 1..{_PAGED_Q_HEAD_TILE} heads, got {num_heads_static}"
            )
        # VERIFY grouping: q_len consecutive rows (one request's spec-decode
        # verify queries) share one CTA group and one K stream. Each of the
        # q_len*heads tile slots carries its own weight, causal seqlen and
        # topk selection. q_len=1 is the original decode contract, bit-exact.
        self.q_len_static = int(q_len)
        if self.q_len_static < 1:
            raise ValueError("q_len must be >= 1")
        self.num_slots_static = self.q_len_static * self.num_heads_static
        # slot cap bounds the parallel select stage (one warp-lane-group per
        # slot) and the MMA q tile (16 rows)
        if self.num_slots_static > 8:
            raise ValueError(
                "MSA fused indexer supports q_len*heads <= 8 slots "
                f"(got {self.q_len_static}x{self.num_heads_static})"
            )
        self.topk = int(topk)
        if self.topk <= 0 or self.topk > 64:
            raise ValueError(f"MSA fused indexer supports topk 1..64, got {topk}")
        self.ctas_per_group = max(1, int(ctas_per_group))
        self.max_blocks = int(max_blocks)
        if self.max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        self.k_quant_page_stride = int(k_quant_page_stride)

    def _shared_storage(self):
        return _msa_shared_storage_cls(
            self.max_blocks, self.num_slots_static, self.topk, self.kv_quant
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,       # u8 [rows, heads, 128] (fp8 e4m3 bytes)
        weights: cute.Tensor,       # f32 [rows, heads] = q_scale * MSA_SM_SCALE
        k_quant_bytes: cute.Tensor, # u8 [pages, 64, 128] (dim0 stride may be packed)
        k_scales: cute.Tensor,      # f32 [pages, 64]
        real_page_table: cute.Tensor,  # i32 [rows, max_pages]
        seqlens: cute.Tensor,       # i32 [rows] causal-adjusted per row
        out_indices: cute.Tensor,   # i32 [heads, rows, topk]
        slab: cute.Tensor,          # f32 [rows * heads * max_blocks] (ctas_pg>1)
        state: cute.Tensor,         # i32 [rows], zero-initialized (self-resets)
        stream: cuda.CUstream,
    ):
        SharedStorage = self._shared_storage()
        self.kernel(
            q_bytes, weights, k_quant_bytes, k_scales, real_page_table,
            seqlens, out_indices, slab, state,
        ).launch(
            grid=((q_bytes.shape[0] // self.q_len_static) * self.ctas_per_group, 1, 1),
            block=[_THREADS, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens: cute.Tensor,
        out_indices: cute.Tensor,
        slab: cute.Tensor,
        state: cute.Tensor,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        bid = Int32(bid)
        ctas_pg = Int32(self.ctas_per_group)
        group_id = bid // ctas_pg
        cta_in_group = bid - group_id * ctas_pg
        q_idx = group_id
        lane = tx % Int32(_WARP_THREADS)
        num_heads = Int32(self.num_heads_static)
        q_len = Int32(self.q_len_static)
        num_slots = Int32(self.num_slots_static)
        base_row = group_id * q_len
        max_blocks = Int32(self.max_blocks)
        topk = Int32(self.topk)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage())
        s_q = storage.q_bytes.get_tensor(
            cute.make_layout((_PAGED_Q_HEAD_TILE, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1))
        )
        q_stage_base_addr = shared_ptr_to_u32(storage.q_bytes.data_ptr())
        s_w = storage.weights.get_tensor(cute.make_layout((_PAGED_Q_HEAD_TILE,), stride=(1,)))
        k_page_perm_base_addr = shared_ptr_to_u32(storage.k_page_perm.data_ptr())
        s_scale = storage.scales.get_tensor(cute.make_layout((_PAGE_SIZE,), stride=(1,)))
        s_page_partials = storage.page_partials.get_tensor(
            cute.make_layout(
                (_TOKEN_GROUPS, _PAGED_Q_HEAD_TILE), stride=(_PAGED_Q_HEAD_TILE, 1)
            )
        )
        s_block = storage.block_scores.get_tensor(
            cute.make_layout(
                (self.num_slots_static, self.max_blocks), stride=(self.max_blocks, 1)
            )
        )
        s_red_v = storage.red_v.get_tensor(
            cute.make_layout((self.num_slots_static, _WARP_THREADS), stride=(_WARP_THREADS, 1))
        )
        s_red_i = storage.red_i.get_tensor(
            cute.make_layout((self.num_slots_static, _WARP_THREADS), stride=(_WARP_THREADS, 1))
        )
        s_sel = storage.sel.get_tensor(
            cute.make_layout((self.num_slots_static, self.topk), stride=(self.topk, 1))
        )
        s_relay = storage.relay.get_tensor(cute.make_layout((1,), stride=(1,)))
        s_valid = storage.valid.get_tensor(
            cute.make_layout((_PAGED_Q_HEAD_TILE,), stride=(1,))
        )

        # group scan range = max slot seqlen (verify rows are causal-ascending
        # but scan the max to be order-independent)
        seq_len = Int32(0)
        j_scan = Int32(0)
        while j_scan < q_len:
            sl_j = Int32(seqlens[base_row + j_scan])
            if sl_j > seq_len:
                seq_len = sl_j
            j_scan += Int32(1)
        total_pages = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)
        # blocks live in [0, num_blocks); ranges partition on EVEN page seams
        num_blocks = (total_pages + Int32(MSA_BLOCK_PAGES - 1)) // Int32(MSA_BLOCK_PAGES)
        pages_per_cta = (total_pages + ctas_pg - Int32(1)) // ctas_pg
        # round UP to even so 128-token blocks never straddle CTA slices
        pages_per_cta = ((pages_per_cta + Int32(1)) // Int32(2)) * Int32(2)
        page_start = cta_in_group * pages_per_cta
        page_end = page_start + pages_per_cta
        if page_end > total_pages:
            page_end = total_pages
        if page_start > total_pages:
            page_start = total_pages

        # ---- stage q + head weights ----
        if cutlass.const_expr(self.kv_quant == "nvfp4"):
            # q arrives as raw bf16 bytes [rows, heads, 256]; stage row-major
            # bf16 [16,128] (pad heads with zero bytes = bf16 +0.0). The u8
            # staging tensor is only used through its base address here.
            sq_nv = storage.q_bytes.get_tensor(
                cute.make_layout(
                    (_PAGED_Q_HEAD_TILE, _INDEX_HEAD_DIM * 2),
                    stride=(_INDEX_HEAD_DIM * 2, 1),
                )
            )
            q_linear = tx
            total_q_bytes = Int32(_PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM * 2)
            row_bytes = Int32(_INDEX_HEAD_DIM * 2)
            while q_linear < total_q_bytes:
                slot_idx = q_linear // row_bytes
                col_idx = q_linear - slot_idx * row_bytes
                j_idx = slot_idx // num_heads
                h_idx = slot_idx - j_idx * num_heads
                sq_nv[slot_idx, col_idx] = (
                    q_bytes[base_row + j_idx, h_idx, col_idx]
                    if slot_idx < num_slots
                    else cutlass.Uint8(0)
                )
                q_linear += Int32(_THREADS)
        else:
            q_linear = tx
            total_q_bytes = Int32(_PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM)
            while q_linear < total_q_bytes:
                slot_idx = q_linear // Int32(_INDEX_HEAD_DIM)
                col_idx = q_linear - slot_idx * Int32(_INDEX_HEAD_DIM)
                j_idx = slot_idx // num_heads
                h_idx = slot_idx - j_idx * num_heads
                s_q[slot_idx, col_idx] = (
                    q_bytes[base_row + j_idx, h_idx, col_idx]
                    if slot_idx < num_slots
                    else cutlass.Uint8(0)
                )
                q_linear += Int32(_THREADS)
        w_linear = tx
        while w_linear < Int32(_PAGED_Q_HEAD_TILE):
            j_w = w_linear // num_heads
            h_w = w_linear - j_w * num_heads
            s_w[w_linear] = (
                Float32(weights[base_row + j_w, h_w])
                if w_linear < num_slots
                else Float32(0.0)
            )
            w_linear += Int32(_THREADS)
        # ---- init block scores to -inf (whole SMEM array) ----
        b_linear = tx
        total_blocks_smem = num_slots * max_blocks
        while b_linear < total_blocks_smem:
            h_i = b_linear // max_blocks
            b_i = b_linear - h_i * max_blocks
            s_block[h_i, b_i] = Float32(_NEG_INF)
            b_linear += Int32(_THREADS)
        cute.arch.sync_threads()

        # ---- page loop: score one 64-token page, fold per-head page max ----
        page_col = page_start
        while page_col < page_end:
            page_base = page_col * Int32(_PAGE_SIZE)
            valid_any = seq_len - page_base
            if valid_any > Int32(_PAGE_SIZE):
                valid_any = Int32(_PAGE_SIZE)
            page_id = Int32(-1)
            if page_col < Int32(real_page_table.shape[1]):
                page_id = Int32(real_page_table[q_idx, page_col])
            if (page_id >= Int32(0)) & (valid_any > Int32(0)):
                if cutlass.const_expr(self.kv_quant == "nvfp4"):
                    _load_expand_k_page_nvfp4_wide(
                        k_quant_bytes, k_scales, page_id,
                        k_page_perm_base_addr, tx, Int32(_THREADS),
                    )
                else:
                    _load_permute_k_page_g2s(
                        k_quant_bytes, page_id, Int64(self.k_quant_page_stride),
                        k_page_perm_base_addr, tx, Int32(_THREADS),
                    )
                    scale_idx = tx
                    while scale_idx < Int32(_PAGE_SIZE):
                        s_scale[scale_idx] = Float32(k_scales[page_id, scale_idx])
                        scale_idx += Int32(_THREADS)
                # init page partials + per-slot valid token counts for THIS page
                p_linear = tx
                while p_linear < Int32(_TOKEN_GROUPS * _PAGED_Q_HEAD_TILE):
                    g_i = p_linear // Int32(_PAGED_Q_HEAD_TILE)
                    h_i = p_linear - g_i * Int32(_PAGED_Q_HEAD_TILE)
                    s_page_partials[g_i, h_i] = Float32(_NEG_INF)
                    p_linear += Int32(_THREADS)
                if tx < Int32(_PAGED_Q_HEAD_TILE):
                    v_slot = Int32(0)
                    if tx < num_slots:
                        j_v = tx // num_heads
                        v_slot = Int32(seqlens[base_row + j_v]) - page_base
                        if v_slot > Int32(_PAGE_SIZE):
                            v_slot = Int32(_PAGE_SIZE)
                        if v_slot < Int32(0):
                            v_slot = Int32(0)
                    s_valid[tx] = v_slot
                cute.arch.sync_threads()
                if tx < Int32(_SCORE_THREADS):
                    token_group = tx // Int32(_WARP_THREADS)
                    token_base = token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                    if cutlass.const_expr(self.kv_quant == "nvfp4"):
                        _compute_bf16_tile_head_token_max(
                            q_stage_base_addr, s_w, num_slots,
                            k_page_perm_base_addr,
                            token_base,
                            lane,
                            s_valid,
                            s_page_partials,
                            token_group,
                        )
                    else:
                        _compute_mxfp8_tile_slot_token_max(
                            s_q, s_w, num_slots,
                            k_page_perm_base_addr,
                            token_base,
                            lane,
                            s_scale,
                            s_valid,
                            s_page_partials,
                            token_group,
                        )
                cute.arch.sync_threads()
                # fold the page max into the 128-token block max (one thread/slot)
                if tx < num_slots:
                    pm = Float32(_NEG_INF)
                    g_it = Int32(0)
                    while g_it < Int32(_TOKEN_GROUPS):
                        pm = attention_ops.fmax(pm, Float32(s_page_partials[g_it, tx]))
                        g_it += Int32(1)
                    blk = page_col // Int32(MSA_BLOCK_PAGES)
                    if blk < max_blocks:
                        s_block[tx, blk] = attention_ops.fmax(
                            Float32(s_block[tx, blk]), pm
                        )
                cute.arch.sync_threads()
            page_col += Int32(1)

        # ---- cross-CTA publish + last-arrival select ----
        run_select = Int32(1)
        if cutlass.const_expr(self.ctas_per_group > 1):
            run_select = Int32(0)
            blk_start = page_start // Int32(MSA_BLOCK_PAGES)
            blk_end = (page_end + Int32(MSA_BLOCK_PAGES - 1)) // Int32(MSA_BLOCK_PAGES)
            slab_row = q_idx * num_slots * max_blocks
            i = Int32(tx)
            span = blk_end - blk_start
            total_span = num_slots * span
            while i < total_span:
                h_i = i // span
                b_i = blk_start + (i - h_i * span)
                slab[slab_row + h_i * max_blocks + b_i] = Float32(s_block[h_i, b_i])
                i += Int32(_THREADS)
            cute.arch.sync_threads()
            if tx == Int32(0):
                threadfence()  # release the slab writes
                s_relay[0] = atomic_add_global_i32(
                    get_ptr_as_int64(state, q_idx), Int32(1)
                )
            cute.arch.sync_threads()
            if Int32(s_relay[0]) == (ctas_pg - Int32(1)):
                run_select = Int32(1)
                threadfence()  # acquire every producer's slab writes
                # reload the FULL block-score row from the slab
                j = Int32(tx)
                total_load = num_slots * num_blocks
                while j < total_load:
                    h_j = j // num_blocks
                    b_j = j - h_j * num_blocks
                    s_block[h_j, b_j] = Float32(slab[slab_row + h_j * max_blocks + b_j])
                    j += Int32(_THREADS)
                cute.arch.sync_threads()

        if run_select != Int32(0):
            # ---- force local causal block (each slot's OWN causal position) ----
            if tx < num_slots:
                j_f = tx // num_heads
                sl_f = Int32(seqlens[base_row + j_f])
                if sl_f > Int32(0):
                    local_blk = (sl_f - Int32(1)) // Int32(_PAGE_SIZE * MSA_BLOCK_PAGES)
                    if local_blk < num_blocks:
                        s_block[tx, local_blk] = Float32(Float32.inf)
            cute.arch.sync_threads()

            # ---- iterative per-slot argmax top-k (slots run in parallel) ----
            head_slot = tx // Int32(_WARP_THREADS)
            in_select = (head_slot < num_slots) & (tx < num_slots * Int32(_WARP_THREADS))
            k_round = Int32(0)
            while k_round < topk:
                if in_select:
                    best_v = Float32(_NEG_INF)
                    best_i = Int32(_INT32_MAX)
                    b_scan = lane
                    while b_scan < num_blocks:
                        v = Float32(s_block[head_slot, b_scan])
                        if v > best_v:
                            best_v = v
                            best_i = b_scan
                        b_scan += Int32(_WARP_THREADS)
                    s_red_v[head_slot, lane] = best_v
                    s_red_i[head_slot, lane] = best_i
                cute.arch.sync_threads()
                if (tx < num_slots):
                    fbest_v = Float32(_NEG_INF)
                    fbest_i = Int32(_INT32_MAX)
                    l_it = Int32(0)
                    while l_it < Int32(_WARP_THREADS):
                        rv = Float32(s_red_v[tx, l_it])
                        ri = Int32(s_red_i[tx, l_it])
                        # strict > keeps the LOWEST block id on ties (lanes scan
                        # ascending ids), matching a deterministic contract
                        if rv > fbest_v:
                            fbest_v = rv
                            fbest_i = ri
                        l_it += Int32(1)
                    if fbest_v > Float32(_NEG_INF):
                        s_sel[tx, k_round] = fbest_i
                        s_block[tx, fbest_i] = Float32(_NEG_INF)
                    else:
                        s_sel[tx, k_round] = Int32(_INT32_MAX)
                cute.arch.sync_threads()
                k_round += Int32(1)

            # ---- ascending sort (invalid INT32_MAX sinks to the tail) + write ----
            if tx < num_slots:
                s_it = Int32(1)
                while s_it < topk:
                    key = Int32(s_sel[tx, s_it])
                    p_it = s_it - Int32(1)
                    moved = Int32(1)
                    while (p_it >= Int32(0)) & (moved != Int32(0)):
                        cur = Int32(s_sel[tx, p_it])
                        if cur > key:
                            s_sel[tx, p_it + Int32(1)] = cur
                            s_sel[tx, p_it] = key
                            p_it -= Int32(1)
                        else:
                            moved = Int32(0)
                    s_it += Int32(1)
            cute.arch.sync_threads()
            o_linear = tx
            total_out = num_slots * topk
            while o_linear < total_out:
                s_o = o_linear // topk
                k_o = o_linear - s_o * topk
                j_o = s_o // num_heads
                h_o = s_o - j_o * num_heads
                v_o = Int32(s_sel[s_o, k_o])
                out_indices[h_o, base_row + j_o, k_o] = (
                    Int32(-1) if v_o == Int32(_INT32_MAX) else v_o
                )
                o_linear += Int32(_THREADS)

            # ---- self-reset the arrival counter (graph-replay safe) ----
            if cutlass.const_expr(self.ctas_per_group > 1):
                cute.arch.sync_threads()
                if tx == Int32(0):
                    state[q_idx] = Int32(0)


@lru_cache(maxsize=32)
def _build_fused_msa_kernel(
    num_heads: int,
    topk: int,
    ctas_per_group: int,
    max_blocks: int,
    k_quant_page_stride: int,
    kv_quant: str = "none",
    q_len: int = 1,
):
    return MSAFusedIndexerKernel(
        num_heads_static=num_heads,
        topk=topk,
        ctas_per_group=ctas_per_group,
        max_blocks=max_blocks,
        k_quant_page_stride=k_quant_page_stride,
        kv_quant=kv_quant,
        q_len=q_len,
    )


def msa_fused_scratch_shapes(
    max_rows: int, num_heads: int, max_pages: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(slab f32 shape, state i32 shape) for caller-owned graph-safe scratch."""
    max_blocks = (int(max_pages) + MSA_BLOCK_PAGES - 1) // MSA_BLOCK_PAGES
    return (
        (int(max_rows) * int(num_heads) * max_blocks,),
        (int(max_rows),),
    )


def run_fused_msa_indexer(
    *,
    q_bytes: torch.Tensor,          # u8 view: fp8 [rows,heads,128] / bf16 [rows,heads,256]
    weights: torch.Tensor,          # f32 [rows, heads] (fp8: q_scale*SM; nvfp4: SM)
    k_quant_bytes: torch.Tensor,    # u8 [pages,64,128] fp8 / [pages,64,64] e2m1
    k_scales: torch.Tensor,         # f32 [pages,64] fp8 / u8 [pages,64,8] e4m3
    real_page_table: torch.Tensor,  # i32 [rows, max_pages]
    seqlens: torch.Tensor,          # i32 [rows]
    num_heads: int,
    topk: int,
    out_indices: torch.Tensor | None = None,  # i32 [heads, rows, topk]
    ctas_per_group: int | None = None,
    slab: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    state_preinitialized: bool = False,
    kv_quant: str = "none",
    q_len: int = 1,
) -> torch.Tensor:
    """One-pass MSA q2k selection. Returns q2k_indices [heads, rows, topk] i32.

    ``slab``/``state`` are optional caller-owned scratch (size via
    ``msa_fused_scratch_shapes``) for graph capture; ``state`` must be
    zero-initialized once (the kernel self-resets it every launch).
    kv_quant="nvfp4": q_bytes is a uint8 view of RAW BF16 q (no q-quant) and
    k_quant_bytes/k_scales are the packed e2m1/e4m3 page views.
    q_len>1 (spec-decode VERIFY): rows are request-major groups of q_len
    consecutive query rows sharing one K stream; real_page_table is PER
    GROUP [rows//q_len, max_pages]; seqlens stays PER ROW (each row's own
    causal bound). Requires q_len*num_heads <= 8.
    """
    rows = int(q_bytes.shape[0])
    q_len = int(q_len)
    if q_len < 1 or rows % q_len != 0:
        raise ValueError(f"rows ({rows}) must be a multiple of q_len ({q_len})")
    groups = rows // q_len
    if int(real_page_table.shape[0]) < groups:
        raise ValueError("real_page_table must have one row per group")
    dev = q_bytes.device
    max_pages = int(real_page_table.shape[1])
    max_blocks = (max_pages + MSA_BLOCK_PAGES - 1) // MSA_BLOCK_PAGES
    if ctas_per_group is None:
        num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        # one CTA wave; never slice below one 2-page block per CTA
        # 2 CTAs per SM at 512 threads: one wave = 2*num_sms CTAs
        ctas_per_group = max(1, min((max_pages + 1) // 2, (2 * num_sms) // max(1, groups)))
    ctas_per_group = max(1, int(ctas_per_group))
    if out_indices is None:
        out_indices = torch.empty((num_heads, rows, topk), dtype=torch.int32, device=dev)
    if ctas_per_group > 1:
        slab_shape, state_shape = msa_fused_scratch_shapes(rows, num_heads, max_pages)
        if slab is None:
            slab = torch.empty(slab_shape, dtype=torch.float32, device=dev)
        if state is None:
            state = torch.zeros(state_shape, dtype=torch.int32, device=dev)
            state_preinitialized = True
        if slab.numel() < slab_shape[0] or state.numel() < state_shape[0]:
            raise ValueError("MSA fused scratch too small (size via msa_fused_scratch_shapes)")
        if not bool(state_preinitialized):
            state[:rows].zero_()
        slab_t = slab[: slab_shape[0]]
        state_t = state[: state_shape[0]]
    else:
        # constexpr-dead in the single-CTA kernel; reuse tiny live tensors
        slab_t = weights.reshape(-1)
        state_t = seqlens
    k_quant_page_stride = int(k_quant_bytes.stride(0))
    kernel = _build_fused_msa_kernel(
        int(num_heads), int(topk), int(ctas_per_group), int(max_blocks),
        k_quant_page_stride, str(kv_quant), int(q_len),
    )
    args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(weights, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8, assumed_align=4),
        _to_kernel_tensor(
            k_scales,
            cutlass.Uint8 if k_scales.dtype == torch.uint8 else cutlass.Float32,
            assumed_align=4,
        ),
        _to_kernel_tensor(real_page_table, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(seqlens, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_indices, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(slab_t, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(state_t, cutlass.Int32, assumed_align=4),
        current_cuda_stream(),
    )
    key_tensors = [
        ("q", q_bytes), ("w", weights), ("kq", k_quant_bytes), ("ks", k_scales),
        ("pt", real_page_table), ("sl", seqlens), ("oi", out_indices),
        ("slab", slab_t), ("st", state_t),
    ]
    _launch_fused(
        kernel, args, key_tensors,
        ("msa", int(num_heads), int(topk), int(ctas_per_group), int(max_blocks),
         k_quant_page_stride, str(kv_quant), int(q_len)),
    )
    return out_indices
