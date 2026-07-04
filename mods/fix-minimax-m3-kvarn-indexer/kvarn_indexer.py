# SPDX-License-Identifier: Apache-2.0
"""KVarN 4-bit index-K cache for MiniMax-M3 (Phase 2B).

The indexer scans the FULL context every decode step, so unlike the main
sparse KV (2A: workspace dequant of topk-selected tiles) the dequant must be
in-kernel. This module carries K-only KVarN tile records
([k_packed D*G/2 | s_col D | zp D | s_row G] fp16 scales = 8960 B/tile =
70 B/token at D=128) plus scorer kernel variants:

Both scorer kernels branch per block: rotated fp16 pool (current-chunk /
tail / sink tiles, staged by _insert_kv BEFORE the indexer runs and packed
only in a LATER builder step) if block_to_slot >= 0, else record dequant.
Forced init/local blocks skip the K read entirely (score overwritten with
1e30/1e29 by the stock semantics). A records-only decode kernel was
considered and rejected: a multi-token (spec) decode step can cross a tile
boundary, making the just-completed tile non-local one builder step before
it is packed, and sinks stay pool-resident by design — the b2s load is
noise next to the 8960-byte record read.

index_q is rotated by the same Hadamard H before scoring ((qH)·(kH) = q·k).
Bookkeeping reuses kvarn_sparse.KVarNSparseGroup (separate instance: the
indexer group has its own block-id space and metadata builder).
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl

_STATE = None  # KVarNSparseGroup for the indexer kv-cache group


def rec_layout(head_dim: int, group: int = 128) -> dict[str, int]:
    k_packed = head_dim * group // 2
    return dict(
        K_PACKED_OFF=0,
        K_SCOL_OFF=k_packed,
        K_ZP_OFF=k_packed + head_dim * 2,
        K_SROW_OFF=k_packed + head_dim * 4,
        REC=k_packed + head_dim * 4 + group * 2,
    )


# ── store: K-only rotated fp16 pool scatter (capture-safe) ──────────────────


@triton.jit
def _k_scatter_store_kernel(
    K_in_ptr,  # [N, D] fp16 (rotated)
    Slot_mapping_ptr,  # [N] int (slot < 0 => pad)
    Block_to_slot_ptr,  # [num_blocks] int32
    Pool_K_ptr,  # [POOL, G, D] fp16
    stride_in_n,
    stride_pool_b,
    stride_pool_t,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    i = tl.program_id(0)
    sm = tl.load(Slot_mapping_ptr + i)
    if sm < 0:
        return
    block_id = sm // GROUP
    pos = (sm % GROUP).to(tl.int64)
    if (block_id < 0) | (block_id >= NUM_BLOCKS_LOOKUP):
        return
    slot = tl.load(Block_to_slot_ptr + block_id)
    if slot < 0:
        return
    d = tl.arange(0, D)
    row = tl.load(K_in_ptr + i * stride_in_n + d)
    tl.store(Pool_K_ptr + slot * stride_pool_b + pos * stride_pool_t + d, row)


# ── scorer variants ──────────────────────────────────────────────────────────


@triton.jit
def _deq_k_tile(
    rec_ptr,  # one record base (uint8)
    off_k,  # positions [BK]
    off_d,  # dims [D]
    K_PACKED_OFF: tl.constexpr,
    K_SCOL_OFF: tl.constexpr,
    K_ZP_OFF: tl.constexpr,
    K_SROW_OFF: tl.constexpr,
    G: tl.constexpr,
):
    """Dequant K tile as [BK, D] fp32 (pos-major, matches decode kernel)."""
    b = tl.load(
        rec_ptr + K_PACKED_OFF
        + off_d[None, :] * (G // 2)
        + (off_k[:, None] // 2)
    )
    nib = (b >> ((off_k[:, None] % 2) * 4)) & 0xF
    s_col = tl.load(
        (rec_ptr + K_SCOL_OFF).to(tl.pointer_type(tl.float16)) + off_d
    ).to(tl.float32)
    zp = tl.load(
        (rec_ptr + K_ZP_OFF).to(tl.pointer_type(tl.float16)) + off_d
    ).to(tl.float32)
    s_row = tl.load(
        (rec_ptr + K_SROW_OFF).to(tl.pointer_type(tl.float16)) + off_k
    ).to(tl.float32)
    return (nib.to(tl.float32) * s_col[None, :] + zp[None, :]) * s_row[:, None]


@triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
def _kvarn_decode_index_score_kernel(
    q_ptr,  # idx_q ROTATED: [total_q, H, D]
    rec_cache_ptr,  # uint8 records: [num_blocks, REC(padded page)]
    b2s_ptr,  # [num_blocks] int32 block -> pool slot (-1 = packed)
    pool_ptr,  # [POOL, G, D] fp16 (rotated)
    score_ptr,  # [H, total_q, max_block] fp32
    block_table_ptr,
    seq_lens,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    sm_scale,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_rec_blk,
    stride_pool_b,
    stride_pool_t,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    num_kv_chunks,
    K_PACKED_OFF: tl.constexpr,
    K_SCOL_OFF: tl.constexpr,
    K_ZP_OFF: tl.constexpr,
    K_SROW_OFF: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Stock _decode_index_score_kernel with (a) pool/record branch per block,
    (b) forced init/local blocks SKIP the K load entirely."""
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + req_id * stride_bt_b
    local_start = tl.maximum(0, num_blocks - local_blocks)
    q = tl.load(
        q_ptr
        + pid_b * stride_q_n
        + tl.arange(0, num_idx_heads) * stride_q_h
        + off_d[:, None] * stride_q_d,
    )  # [D, H]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        is_init = blk < init_blocks
        is_local = blk >= local_start
        if is_init | is_local:
            score = tl.zeros((num_idx_heads,), dtype=tl.float32) + tl.where(
                is_local, 1e29, 1e30
            )
        else:
            page = tl.load(bt_row + blk).to(tl.int64)
            slot = tl.load(b2s_ptr + page)
            if slot >= 0:
                # pool-resident (just-completed / sink tile): fp16 [N, D]
                k = tl.load(
                    pool_ptr
                    + slot.to(tl.int64) * stride_pool_b
                    + off_k[:, None] * stride_pool_t
                    + off_d[None, :],
                ).to(q.dtype)
            else:
                k = _deq_k_tile(
                    rec_cache_ptr + page * stride_rec_blk,
                    off_k, off_d,
                    K_PACKED_OFF=K_PACKED_OFF, K_SCOL_OFF=K_SCOL_OFF,
                    K_ZP_OFF=K_ZP_OFF, K_SROW_OFF=K_SROW_OFF,
                    G=BLOCK_SIZE_K,
                ).to(q.dtype)  # [N, D]
            pos = blk * BLOCK_SIZE_K + off_k
            pos_mask = pos < kv_len
            kq = tl.dot(k, q) * sm_scale_log2e  # [N, H]
            kq = tl.where(pos_mask[:, None], kq, float("-inf"))
            score = tl.max(kq, axis=0)
        tl.store(
            score_ptr
            + tl.arange(0, num_idx_heads) * stride_s_h
            + pid_b * stride_s_n
            + blk * stride_s_k,
            score,
        )


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _kvarn_index_block_score_kernel(
    q_ptr,  # idx_q ROTATED: [total_q, H, D]
    rec_cache_ptr,  # uint8 records [num_blocks, REC(padded)]
    b2s_ptr,  # [num_blocks] int32 block -> pool slot (-1 = packed)
    pool_ptr,  # [POOL, G, D] fp16 (rotated)
    score_ptr,
    block_table_ptr,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    num_idx_heads,
    head_dim: tl.constexpr,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_rec_blk,
    stride_pool_b,
    stride_pool_t,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    K_PACKED_OFF: tl.constexpr,
    K_SCOL_OFF: tl.constexpr,
    K_ZP_OFF: tl.constexpr,
    K_SROW_OFF: tl.constexpr,
):
    """Stock _index_block_score_kernel with a per-block pool/record branch."""
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        slot = tl.load(b2s_ptr + page)
        if slot >= 0:
            # pool-resident (current chunk / tail): fp16 [G, D] -> [D, K]
            k = tl.load(
                pool_ptr
                + slot.to(tl.int64) * stride_pool_b
                + off_k[None, :] * stride_pool_t
                + off_d[:, None],
            ).to(q.dtype)
        else:
            k = tl.trans(
                _deq_k_tile(
                    rec_cache_ptr + page * stride_rec_blk,
                    off_k, off_d,
                    K_PACKED_OFF=K_PACKED_OFF, K_SCOL_OFF=K_SCOL_OFF,
                    K_ZP_OFF=K_ZP_OFF, K_SROW_OFF=K_SROW_OFF,
                    G=BLOCK_SIZE_K,
                )
            ).to(q.dtype)  # [D, K]
        qk = tl.dot(q, k) * sm_scale_log2e
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        score = tl.max(qk, axis=1)
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)


# ── per-layer state (plugs into kvarn_sparse.KVarNSparseGroup) ──────────────


class KVarNIndexerLayer:
    """K-only rotated fp16 pool + uint8 record cache for one indexer layer."""

    def __init__(self, layer_name: str, head_dim: int):
        self.name = layer_name
        self.d = head_dim
        self.layout = rec_layout(head_dim)
        self.kv_cache: torch.Tensor | None = None  # uint8 [nb, >=REC]
        self.pool_k: torch.Tensor | None = None  # fp16 [POOL, G, D]
        self._rot_k: torch.Tensor | None = None
        self._H16: torch.Tensor | None = None

    def ensure(self, kv_cache: torch.Tensor, group) -> bool:
        if kv_cache is None or kv_cache.numel() == 0:
            return False
        if self.kv_cache is not None and (
                self.kv_cache.data_ptr() != kv_cache.data_ptr()
                or self.kv_cache.shape != kv_cache.shape):
            self.kv_cache = kv_cache  # placeholder -> real bind
        if self.kv_cache is None:
            assert kv_cache.dim() == 2 and (
                kv_cache.shape[1] >= self.layout["REC"]), (
                f"kvarn indexer cache shape {tuple(kv_cache.shape)}; "
                f"want [nb, >= {self.layout['REC']}]")
            self.kv_cache = kv_cache
            dev = kv_cache.device
            g = 128
            self.pool_k = torch.zeros(group.pool_slots, g, self.d,
                                      dtype=torch.float16, device=dev)
            self._rot_k = torch.empty(group.max_store_tokens, self.d,
                                      dtype=torch.float16, device=dev)
            from vllm.v1.attention.backends.kvarn_attn import _build_hadamard
            self._H16 = _build_hadamard(self.d, dev).to(torch.float16)
            group.register_layer(self, kv_cache.shape[0], dev)
        return True

    def store(self, index_key: torch.Tensor, slot_mapping: torch.Tensor,
              group) -> None:
        n = slot_mapping.shape[0]
        if n == 0 or group.block_to_slot is None:
            return
        k = index_key[:n].view(n, self.d).to(torch.float16)
        k_rot = self._rot_k[:n]
        torch.matmul(k, self._H16, out=k_rot)
        _k_scatter_store_kernel[(n,)](
            k_rot, slot_mapping[:n], group.block_to_slot, self.pool_k,
            k_rot.stride(0),
            self.pool_k.stride(0), self.pool_k.stride(1),
            GROUP=128, D=self.d, NUM_BLOCKS_LOOKUP=group.num_blocks,
            num_warps=2, num_stages=2,
        )

    def flush_pages(self, pages: list[int], slots: list[int]) -> None:
        if not pages:
            return
        nb = int(self.kv_cache.shape[0])
        bad = [p for p in pages if not (0 <= p < nb)]
        assert not bad, (
            f"kvarn indexer flush OOB: layer {self.name} pages {bad[:8]} "
            f"vs num_blocks {nb}")
        from vllm.v1.attention.ops.kvarn_store import (
            kvarn_store_tile_k_batch_from_sinkhorn,
        )
        from vllm.v1.attention.ops.triton_kvarn_sinkhorn import (
            kvarn_sinkhorn_triton,
        )
        d = self.d
        idx = torch.tensor(slots, dtype=torch.long, device=self.pool_k.device)
        tiles = self.pool_k[idx].permute(0, 2, 1).contiguous().float()  # [n,D,G]
        bal, sc, sr = kvarn_sinkhorn_triton(tiles, iterations=8)
        K_out = kvarn_store_tile_k_batch_from_sinkhorn(bal, sc, sr, bits=4)
        n = len(pages)
        lo = self.layout
        rec = torch.empty(n, lo["REC"], dtype=torch.uint8,
                          device=self.pool_k.device)

        def _put(off, t):
            b = t.reshape(n, -1).view(torch.uint8)
            rec[:, off:off + b.shape[1]] = b

        _put(lo["K_PACKED_OFF"], K_out["q_packed_uint8"])
        _put(lo["K_SCOL_OFF"], K_out["s_col_K"].to(torch.float16))
        _put(lo["K_ZP_OFF"], K_out["zp_K"].to(torch.float16))
        _put(lo["K_SROW_OFF"], K_out["s_row_K"].to(torch.float16))
        pidx = torch.tensor(pages, dtype=torch.long, device=rec.device)
        self.kv_cache[pidx, : lo["REC"]] = rec


# ── wrappers mirroring minimax_m3_index_decode / minimax_m3_index_score ─────


def layer_group_b2s(layer: KVarNIndexerLayer, block_table):
    return _STATE.block_to_slot


def kvarn_index_decode_score(idx_q_rot, layer: KVarNIndexerLayer, block_table,
                             seq_lens, max_seq_len, init_blocks, local_blocks,
                             sm_scale, decode_query_len, score,
                             num_kv_chunks, use_pdl, pdl_launch):
    total_q, num_idx_heads, head_dim = idx_q_rot.shape
    lo = layer.layout
    grid = (total_q, num_kv_chunks)
    _kvarn_decode_index_score_kernel[grid](
        idx_q_rot,
        layer.kv_cache,
        layer_group_b2s(layer, block_table),
        layer.pool_k,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        init_blocks,
        local_blocks,
        sm_scale,
        decode_query_len,
        idx_q_rot.stride(0), idx_q_rot.stride(1), idx_q_rot.stride(2),
        layer.kv_cache.stride(0),
        layer.pool_k.stride(0), layer.pool_k.stride(1),
        score.stride(0), score.stride(1), score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=128,
        num_kv_chunks=num_kv_chunks,
        K_PACKED_OFF=lo["K_PACKED_OFF"], K_SCOL_OFF=lo["K_SCOL_OFF"],
        K_ZP_OFF=lo["K_ZP_OFF"], K_SROW_OFF=lo["K_SROW_OFF"],
        USE_PDL=use_pdl,
        **pdl_launch,
    )


def kvarn_index_prefill_score(idx_q_rot, layer: KVarNIndexerLayer, group,
                              block_table, cu_seqlens_q, seq_lens, prefix_lens,
                              max_query_len, sm_scale, score):
    total_q, num_idx_heads, head_dim = idx_q_rot.shape
    lo = layer.layout
    batch = cu_seqlens_q.shape[0] - 1
    BLOCK_SIZE_Q = 64
    grid = (triton.cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads)
    _kvarn_index_block_score_kernel[grid](
        idx_q_rot,
        layer.kv_cache,
        group.block_to_slot,
        layer.pool_k,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        head_dim,
        sm_scale,
        idx_q_rot.stride(0), idx_q_rot.stride(1), idx_q_rot.stride(2),
        layer.kv_cache.stride(0),
        layer.pool_k.stride(0), layer.pool_k.stride(1),
        score.stride(0), score.stride(1), score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=128,
        K_PACKED_OFF=lo["K_PACKED_OFF"], K_SCOL_OFF=lo["K_SCOL_OFF"],
        K_ZP_OFF=lo["K_ZP_OFF"], K_SROW_OFF=lo["K_SROW_OFF"],
    )


def get_group(vllm_config):
    """Indexer-group coordinator (separate block-id space from the main KV)."""
    global _STATE
    if _STATE is None:
        from vllm.models.minimax_m3.common.kvarn_sparse import KVarNSparseGroup
        _STATE = KVarNSparseGroup(vllm_config)
    return _STATE
