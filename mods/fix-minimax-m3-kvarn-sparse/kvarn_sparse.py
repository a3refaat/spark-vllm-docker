# SPDX-License-Identifier: Apache-2.0
"""KVarN tile KV for MiniMax-M3 sparse MSA layers (Phase 2A).

Design (see docs/b12x-kvarn-integration-plan.md): the block-sparse Triton
attend kernels are UNCHANGED. Incoming K/V is Hadamard-rotated and staged in a
small fp16 pool (graph-safe scatter); full 128-token tiles are Sinkhorn+RTN
packed into the uint8 tile cache by the metadata builder BETWEEN graph
replays. At read time a Triton kernel dequantizes the indexer-selected tiles
(and copies pool-resident tiles) into a bounded bf16 workspace shaped exactly
like the stock paged cache, and scatters a parallel block table so
``block_table[req, blk]`` redirects to the workspace page. topk indices,
position math and causal masks are untouched. Queries are rotated (q @ H) and
the output un-rotated (o @ H): scores are exact under the orthonormal H, and
V stays in the rotated frame end-to-end.

Correctness notes:
  * Duplicate selections (two heads / tokens picking the same block) dequant
    into different workspace pages with IDENTICAL content; the parallel
    block-table scatter race is benign.
  * All 57 sparse layers share one KV-cache group => one physical block-id
    space => ONE shared block->slot map + fill tracker; pools are per layer.
  * Fill is recomputed ABSOLUTELY from (num_computed, seq_len) each build
    (idempotent under preemption/resume).
  * Sink blocks (first block of a live request) stay fp16-resident and are
    flushed only on reclaim. Partial tiles of finished requests are dropped
    (vLLM never prefix-caches partial blocks); full ones are flushed.
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl

# ── config / layout ──────────────────────────────────────────────────────────

_STATE: "KVarNSparseGroup | None" = None


def _cfg(cache_dtype: str, head_dim: int):
    from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig

    return KVarNConfig.from_cache_dtype(cache_dtype, head_dim)


def is_kvarn_dtype(cache_dtype) -> bool:
    return isinstance(cache_dtype, str) and cache_dtype.startswith("kvarn_") and \
        not cache_dtype.startswith("kvarn_mla")


# ── Triton: dequant one (block, head) record into a workspace page ──────────


@triton.jit
def _dequant_record_to_ws(
    rec_ptr,  # base of one record (uint8), cfg offsets as constexprs
    ws_ptr,  # base of ws[ws_id, :, :, h, :] page for this head
    stride_ws_kv,
    stride_ws_pos,
    stride_ws_d,
    K_PACKED_OFF: tl.constexpr,
    K_SCOL_OFF: tl.constexpr,
    K_ZP_OFF: tl.constexpr,
    K_SROW_OFF: tl.constexpr,
    V_PACKED_OFF: tl.constexpr,
    V_SCOL_OFF: tl.constexpr,
    V_SROW_OFF: tl.constexpr,
    V_ZP_OFF: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    KBITS: tl.constexpr,
    VBITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    offs_g = tl.arange(0, G)
    offs_d = tl.arange(0, D)

    # K: packed [D, G // (8//KBITS)]; value g at byte g//pack, shift (g%pack)*KBITS
    kpack = 8 // KBITS
    kmask = (1 << KBITS) - 1
    k_bytes = tl.load(
        rec_ptr + K_PACKED_OFF
        + offs_d[:, None] * (G // kpack)
        + (offs_g[None, :] // kpack)
    )
    k_q = (k_bytes >> ((offs_g[None, :] % kpack) * KBITS)) & kmask
    s_col_k = tl.load(
        (rec_ptr + K_SCOL_OFF).to(tl.pointer_type(tl.float16)) + offs_d
    ).to(tl.float32)
    zp_k = tl.load(
        (rec_ptr + K_ZP_OFF).to(tl.pointer_type(tl.float16)) + offs_d
    ).to(tl.float32)
    s_row_k = tl.load(
        (rec_ptr + K_SROW_OFF).to(tl.pointer_type(tl.float16)) + offs_g
    ).to(tl.float32)
    # x[d, g] = (q * s_col'[d] + zp'[d]) * s_row[g]  (rotated frame)
    k_val = (k_q.to(tl.float32) * s_col_k[:, None] + zp_k[:, None]) * s_row_k[None, :]
    # store transposed into ws K page: [pos g, d]
    tl.store(
        ws_ptr + 0 * stride_ws_kv
        + offs_g[None, :] * stride_ws_pos
        + offs_d[:, None] * stride_ws_d,
        k_val.to(OUT_DTYPE),
    )

    # V: packed [G, D // (8//VBITS)]; value d at byte d//pack, shift (d%pack)*VBITS
    vpack = 8 // VBITS
    vmask = (1 << VBITS) - 1
    v_bytes = tl.load(
        rec_ptr + V_PACKED_OFF
        + offs_g[:, None] * (D // vpack)
        + (offs_d[None, :] // vpack)
    )
    v_q = (v_bytes >> ((offs_d[None, :] % vpack) * VBITS)) & vmask
    s_col_v = tl.load(
        (rec_ptr + V_SCOL_OFF).to(tl.pointer_type(tl.float16)) + offs_d
    ).to(tl.float32)
    s_row_v = tl.load(
        (rec_ptr + V_SROW_OFF).to(tl.pointer_type(tl.float16)) + offs_g
    ).to(tl.float32)
    zp_v = tl.load(
        (rec_ptr + V_ZP_OFF).to(tl.pointer_type(tl.float16)) + offs_g
    ).to(tl.float32)
    # x[g, d] = (q * s_row'[g] + zp'[g]) * s_col[d]  (rotated frame)
    v_val = (v_q.to(tl.float32) * s_row_v[:, None] + zp_v[:, None]) * s_col_v[None, :]
    tl.store(
        ws_ptr + 1 * stride_ws_kv
        + offs_g[:, None] * stride_ws_pos
        + offs_d[None, :] * stride_ws_d,
        v_val.to(OUT_DTYPE),
    )


@triton.jit
def _copy_pool_to_ws(
    poolk_ptr,  # pool K base for (slot, h): [G, D] stride (pool_t, 1)
    poolv_ptr,
    ws_ptr,
    stride_pool_t,
    stride_ws_kv,
    stride_ws_pos,
    stride_ws_d,
    G: tl.constexpr,
    D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    offs_g = tl.arange(0, G)
    offs_d = tl.arange(0, D)
    k = tl.load(poolk_ptr + offs_g[:, None] * stride_pool_t + offs_d[None, :])
    v = tl.load(poolv_ptr + offs_g[:, None] * stride_pool_t + offs_d[None, :])
    dst = offs_g[:, None] * stride_ws_pos + offs_d[None, :] * stride_ws_d
    tl.store(ws_ptr + 0 * stride_ws_kv + dst, k.to(OUT_DTYPE))
    tl.store(ws_ptr + 1 * stride_ws_kv + dst, v.to(OUT_DTYPE))


@triton.jit
def _ws_build_topk_kernel(
    topk_ptr,  # [HK, total_q, topk] int
    bt_ptr,  # [num_reqs, width] int32 (real)
    b2s_ptr,  # [num_blocks] int32 block->pool slot (-1 = flushed tile)
    cache_ptr,  # uint8 [num_blocks, HK, REC]
    poolk_ptr,  # fp16 [POOL, G, HK, D]
    poolv_ptr,
    ws_ptr,  # [WS_PAGES, 2, G, HK, D] bf16
    btws_ptr,  # [num_reqs, width] int32 (parallel/workspace table)
    total_q,
    q_len,
    stride_th,
    stride_tn,
    stride_bt_b,
    stride_cache_blk,
    stride_cache_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_ws_b,
    stride_ws_kv,
    stride_ws_pos,
    stride_ws_h,
    stride_ws_d,
    TOPK: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    KBITS: tl.constexpr,
    VBITS: tl.constexpr,
    K_PACKED_OFF: tl.constexpr,
    K_SCOL_OFF: tl.constexpr,
    K_ZP_OFF: tl.constexpr,
    K_SROW_OFF: tl.constexpr,
    V_PACKED_OFF: tl.constexpr,
    V_SCOL_OFF: tl.constexpr,
    V_SROW_OFF: tl.constexpr,
    V_ZP_OFF: tl.constexpr,
):
    """grid = (HK * total_q * TOPK,). Program s: selection (kh, tok, j) ->
    logical block blk -> physical page -> dequant/copy ALL heads into
    ws[s] and scatter btws[req, blk] = s (duplicate writes are benign:
    every claimed page carries identical, all-head content)."""
    s = tl.program_id(0)
    j = s % TOPK
    tok = (s // TOPK) % total_q
    kh_sel = s // (TOPK * total_q)
    if tok >= total_q:
        return
    blk = tl.load(topk_ptr + kh_sel * stride_th + tok * stride_tn + j)
    if blk < 0:
        return
    req = tok // q_len
    page = tl.load(bt_ptr + req * stride_bt_b + blk).to(tl.int64)
    slot = tl.load(b2s_ptr + page)
    for h in tl.static_range(HK):
        ws_page = ws_ptr + s * stride_ws_b + h * stride_ws_h
        if slot >= 0:
            _copy_pool_to_ws(
                poolk_ptr + slot * stride_pool_b + h * stride_pool_h,
                poolv_ptr + slot * stride_pool_b + h * stride_pool_h,
                ws_page,
                stride_pool_t,
                stride_ws_kv, stride_ws_pos, stride_ws_d,
                G=G, D=D, OUT_DTYPE=tl.bfloat16,
            )
        else:
            _dequant_record_to_ws(
                cache_ptr + page * stride_cache_blk + h * stride_cache_h,
                ws_page,
                stride_ws_kv, stride_ws_pos, stride_ws_d,
                K_PACKED_OFF=K_PACKED_OFF, K_SCOL_OFF=K_SCOL_OFF,
                K_ZP_OFF=K_ZP_OFF, K_SROW_OFF=K_SROW_OFF,
                V_PACKED_OFF=V_PACKED_OFF, V_SCOL_OFF=V_SCOL_OFF,
                V_SROW_OFF=V_SROW_OFF, V_ZP_OFF=V_ZP_OFF,
                G=G, D=D, KBITS=KBITS, VBITS=VBITS, OUT_DTYPE=tl.bfloat16,
            )
    tl.store(btws_ptr + req * stride_bt_b + blk, s)


@triton.jit
def _ws_build_pages_kernel(
    pages_ptr,  # [N] int32 physical pages to materialize; ws_id = index
    b2s_ptr,
    cache_ptr,
    poolk_ptr,
    poolv_ptr,
    ws_ptr,
    stride_cache_blk,
    stride_cache_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_ws_b,
    stride_ws_kv,
    stride_ws_pos,
    stride_ws_h,
    stride_ws_d,
    HK: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    KBITS: tl.constexpr,
    VBITS: tl.constexpr,
    K_PACKED_OFF: tl.constexpr,
    K_SCOL_OFF: tl.constexpr,
    K_ZP_OFF: tl.constexpr,
    K_SROW_OFF: tl.constexpr,
    V_PACKED_OFF: tl.constexpr,
    V_SCOL_OFF: tl.constexpr,
    V_SROW_OFF: tl.constexpr,
    V_ZP_OFF: tl.constexpr,
):
    """Prefill context gather: grid (N,), ws[i] <- dequant/copy pages[i]."""
    i = tl.program_id(0)
    page = tl.load(pages_ptr + i).to(tl.int64)
    if page < 0:
        return
    slot = tl.load(b2s_ptr + page)
    for h in tl.static_range(HK):
        ws_page = ws_ptr + i * stride_ws_b + h * stride_ws_h
        if slot >= 0:
            _copy_pool_to_ws(
                poolk_ptr + slot * stride_pool_b + h * stride_pool_h,
                poolv_ptr + slot * stride_pool_b + h * stride_pool_h,
                ws_page,
                stride_pool_t,
                stride_ws_kv, stride_ws_pos, stride_ws_d,
                G=G, D=D, OUT_DTYPE=tl.bfloat16,
            )
        else:
            _dequant_record_to_ws(
                cache_ptr + page * stride_cache_blk + h * stride_cache_h,
                ws_page,
                stride_ws_kv, stride_ws_pos, stride_ws_d,
                K_PACKED_OFF=K_PACKED_OFF, K_SCOL_OFF=K_SCOL_OFF,
                K_ZP_OFF=K_ZP_OFF, K_SROW_OFF=K_SROW_OFF,
                V_PACKED_OFF=V_PACKED_OFF, V_SCOL_OFF=V_SCOL_OFF,
                V_SROW_OFF=V_SROW_OFF, V_ZP_OFF=V_ZP_OFF,
                G=G, D=D, KBITS=KBITS, VBITS=VBITS, OUT_DTYPE=tl.bfloat16,
            )


# ── per-layer state + shared group coordinator ───────────────────────────────


class KVarNSparseLayer:
    """Per-layer rotated fp16 pool + uint8 tile cache handle."""

    def __init__(self, layer_name: str, num_kv_heads: int, head_dim: int,
                 cache_dtype: str):
        self.name = layer_name
        self.hk = num_kv_heads
        self.d = head_dim
        self.cfg = _cfg(cache_dtype, head_dim)
        self.kv_cache: torch.Tensor | None = None  # uint8 [nb, hk, REC]
        self.pool_k: torch.Tensor | None = None  # fp16 [POOL, G, hk, d]
        self.pool_v: torch.Tensor | None = None
        self._rot_k: torch.Tensor | None = None
        self._rot_v: torch.Tensor | None = None
        self._H16: torch.Tensor | None = None

    def ensure(self, kv_cache: torch.Tensor, group: "KVarNSparseGroup") -> bool:
        if kv_cache is None or kv_cache.numel() == 0:
            return False  # profiling run: caches unbound
        if self.kv_cache is not None and (
                self.kv_cache.data_ptr() != kv_cache.data_ptr()
                or self.kv_cache.shape != kv_cache.shape):
            # vLLM binds a small placeholder cache during backend init and the
            # REAL cache later (bind_kv_cache); track the current tensor.
            # Eager-only rebind risk is nil: decode graphs are captured after
            # the real bind, and flush/ws launches read self.kv_cache fresh.
            self.kv_cache = kv_cache
        if self.kv_cache is None:
            import os
            if os.environ.get("KVARN_SPARSE_DEBUG", "0") == "1":
                print(f"[kvarn-sparse] {self.name}: kv_cache shape="
                      f"{tuple(kv_cache.shape)} strides={kv_cache.stride()} "
                      f"dtype={kv_cache.dtype}", flush=True)
            assert kv_cache.dim() == 3 and kv_cache.shape[1] == self.hk, (
                f"kvarn sparse cache bound with unexpected shape "
                f"{tuple(kv_cache.shape)} (want [nb, {self.hk}, rec])")
            self.kv_cache = kv_cache
            g = self.cfg.group
            dev = kv_cache.device
            self.pool_k = torch.zeros(
                group.pool_slots, g, self.hk, self.d,
                dtype=torch.float16, device=dev)
            self.pool_v = torch.zeros_like(self.pool_k)
            n = group.max_store_tokens
            self._rot_k = torch.empty(n, self.hk, self.d,
                                      dtype=torch.float16, device=dev)
            self._rot_v = torch.empty_like(self._rot_k)
            from vllm.v1.attention.backends.kvarn_attn import _build_hadamard
            self._H16 = _build_hadamard(self.d, dev).to(torch.float16)
            group.register_layer(self, kv_cache.shape[0], dev)
        return True

    def store(self, key: torch.Tensor, value: torch.Tensor,
              slot_mapping: torch.Tensor, group: "KVarNSparseGroup") -> None:
        """Rotate + scatter into the pool. Pure tensor ops (capture-safe).

        NOTE: NUM_BLOCKS_LOOKUP is read per launch (eager store; decode-path
        stores happen inside the graph but slot ids there are bounded by the
        capture-time map, which only ever grows)."""
        n = slot_mapping.shape[0]
        if n == 0 or group.block_to_slot is None:
            return
        k = key[:n].view(n, self.hk, self.d).to(torch.float16)
        v = value[:n].view(n, self.hk, self.d).to(torch.float16)
        k_rot = self._rot_k[:n]
        v_rot = self._rot_v[:n]
        torch.matmul(k, self._H16, out=k_rot)
        torch.matmul(v, self._H16, out=v_rot)
        from vllm.v1.attention.ops.triton_kvarn_decode import (
            _kvarn_scatter_store_kernel,
        )
        _kvarn_scatter_store_kernel[(n, self.hk)](
            k_rot, v_rot,
            slot_mapping[:n], group.block_to_slot,
            self.pool_k, self.pool_v,
            k_rot.stride(0), k_rot.stride(1),
            self.pool_k.stride(0), self.pool_k.stride(1), self.pool_k.stride(2),
            GROUP=self.cfg.group, D=self.d,
            NUM_BLOCKS_LOOKUP=group.num_blocks,
            num_warps=2, num_stages=2,
        )

    def flush_pages(self, pages: list[int], slots: list[int]) -> None:
        """Sinkhorn+RTN pack pool slots -> uint8 records at `pages` (eager)."""
        if not pages:
            return
        nb = int(self.kv_cache.shape[0])
        bad = [p for p in pages if not (0 <= p < nb)]
        assert not bad, (
            f"kvarn flush OOB: layer {self.name} pages {bad[:8]} "
            f"vs cache num_blocks {nb} (all pages {pages[:16]}...)")
        from vllm.v1.attention.backends.kvarn_attn import _sinkhorn_pack_kv
        cfg = self.cfg
        g, d, hk = cfg.group, self.d, self.hk
        idx = torch.tensor(slots, dtype=torch.long, device=self.pool_k.device)
        # [n, G, hk, D] -> per (tile, head): K [n*hk, D, G], V [n*hk, G, D]
        pk = self.pool_k[idx].permute(0, 2, 3, 1).reshape(-1, d, g).float()
        pv = self.pool_v[idx].permute(0, 2, 1, 3).reshape(-1, g, d).float()
        K_out, V_out = _sinkhorn_pack_kv(pk, pv, cfg)
        n = len(pages)
        rec = torch.empty(n * hk, cfg.tile_bytes_aligned,
                          dtype=torch.uint8, device=self.pool_k.device)

        def _put(off, t):
            b = t.reshape(n * hk, -1).view(torch.uint8)
            rec[:, off:off + b.shape[1]] = b

        _put(cfg.k_packed_offset, K_out["q_packed_uint8"])
        _put(cfg.k_s_col_offset, K_out["s_col_K"].to(torch.float16))
        _put(cfg.k_zp_offset, K_out["zp_K"].to(torch.float16))
        _put(cfg.k_s_row_offset, K_out["s_row_K"].to(torch.float16))
        _put(cfg.v_packed_offset, V_out["q_packed_uint8"])
        _put(cfg.v_s_col_offset, V_out["s_col_V"].to(torch.float16))
        _put(cfg.v_s_row_offset, V_out["s_row_V"].to(torch.float16))
        _put(cfg.v_zp_offset, V_out["zp_V"].to(torch.float16))
        pidx = torch.tensor(pages, dtype=torch.long, device=rec.device)
        # Advanced indexing (not .view): the cache may be an as_strided page-
        # padded tensor (page_size_padded from unify_kv_cache_spec_page_size).
        self.kv_cache[pidx] = rec.view(n, hk, -1)


class KVarNSparseGroup:
    """Shared coordinator: one block-id space for all 57 sparse layers.

    Host bookkeeping (slot alloc, absolute fill, sink marking, flush/reclaim)
    runs ONLY from the metadata builder between graph replays. GPU state
    mutated here: block_to_slot (int32 [nb]).
    """

    def __init__(self, vllm_config):
        sched = vllm_config.scheduler_config
        self._cache_config = vllm_config.cache_config
        self.max_seqs = max(int(sched.max_num_seqs), 1)
        self.max_store_tokens = max(int(sched.max_num_batched_tokens), 1)
        spec = getattr(vllm_config, "speculative_config", None)
        nspec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
        self.q_len_max = 1 + nspec
        group = 128
        prefill_blocks = (self.max_store_tokens + group - 1) // group
        # 2x prefill blocks: flush lags one step behind the store (a tile is
        # only packed once its data is known-resident), so two full chunks of
        # tiles can be pool-resident at once. Per live seq: sink + current
        # partial + a scheduled-crossed-but-uncommitted tile (spec decode
        # verify can write past a tile boundary that only commits later).
        self.pool_slots = (3 * self.max_seqs + self.q_len_max
                           + 2 * prefill_blocks + 8)
        self.num_blocks = 0
        self.block_to_slot: torch.Tensor | None = None
        self.layers: list[KVarNSparseLayer] = []
        self._free: list[int] = []
        self._fill: dict[int, int] = {}  # page -> tokens SCHEDULED (advisory)
        self._cfull: set[int] = set()  # committed-full pool pages (sinks)
        self._slot_of: dict[int, int] = {}
        self._sinks: set[int] = set()
        self._max_model_len = int(vllm_config.model_config.max_model_len)
        # decode ws sized for the largest captured decode batch
        self._ws_dec: torch.Tensor | None = None
        self._ws_pre: torch.Tensor | None = None
        self._bt_ws: torch.Tensor | None = None
        self._topk = 16
        self._dev = None
        self._seqlens_cpu = None  # stashed per build for the prefill ws path

    # -- registration ---------------------------------------------------------

    def _init_maps(self, num_blocks: int, dev) -> None:
        self.num_blocks = num_blocks
        self.block_to_slot = torch.full(
            (num_blocks,), -1, dtype=torch.int32, device=dev)
        self._free = list(range(self.pool_slots - 1, -1, -1))
        self._dev = dev

    def _ensure_map_size(self, min_blocks: int) -> None:
        """The captured decode graph bakes the map POINTER into the scatter-
        store kernel, so the map must NEVER be reallocated after capture.
        It is therefore allocated once, generously (int32: 1M ids = 4 MB);
        overflow past that is a hard error, not silent growth."""
        assert min_blocks <= self.num_blocks, (
            f"kvarn sparse block map too small: page id {min_blocks - 1} >= "
            f"{self.num_blocks}; raise the fixed allocation in _init_maps")

    def register_layer(self, layer: KVarNSparseLayer, num_blocks: int, dev):
        if self.block_to_slot is None:
            # First toucher wins: serving usually reaches builder_step first
            # (oversized map); offline tests reach here first (exact size).
            self._init_maps(max(num_blocks, 1024), dev)
        self.layers.append(layer)

    def ws_buffers(self, hk: int, d: int, group: int, topk: int, width: int):
        if self._ws_dec is None:
            self._topk = topk
            dec_pages = hk * self.max_seqs * self.q_len_max * topk
            self._ws_dec = torch.zeros(dec_pages, 2, group, hk, d,
                                       dtype=torch.bfloat16, device=self._dev)
            pre_pages = (self._max_model_len + group - 1) // group + 4
            self._ws_pre = torch.zeros(pre_pages, 2, group, hk, d,
                                       dtype=torch.bfloat16, device=self._dev)
            self._bt_ws = torch.zeros(self.max_seqs + 4, width,
                                      dtype=torch.int32, device=self._dev)
        assert self._bt_ws.shape[1] == width, "block table width changed"
        return self._ws_dec, self._ws_pre, self._bt_ws

    def grow_prefill_ws(self, pages: int, hk: int, d: int, group: int):
        """Eager-path only: several concurrent prefills can exceed
        max_model_len worth of context blocks."""
        if pages > self._ws_pre.shape[0]:
            self._ws_pre = torch.zeros(pages, 2, group, hk, d,
                                       dtype=torch.bfloat16, device=self._dev)
        return self._ws_pre

    # -- builder hook (eager, between replays) --------------------------------

    def builder_step(self, common_attn_metadata) -> None:
        """Eager host bookkeeping, runs BEFORE this step's forwards.

        Ordering matters: at entry ``self._fill`` reflects tokens scheduled by
        the PREVIOUS step, whose data the previous forwards already stored --
        those are the only tiles safe to pack. So: (1) flush/reclaim on the
        old fills, (2) merge this step's scheduled fills, (3) allocate slots
        (flush freed capacity first).
        """
        cm = common_attn_metadata
        num_reqs = int(cm.num_reqs)
        if num_reqs <= 0:
            return
        if self.block_to_slot is None:
            nb = int(getattr(self._cache_config, "num_gpu_blocks", 0) or 0)
            self._init_maps(max(4 * nb, 1 << 20),
                            cm.block_table_tensor.device)
        seq_lens_cpu = cm.seq_lens_cpu
        self._seqlens_cpu = seq_lens_cpu
        computed_cpu = cm.num_computed_tokens_cpu
        g = 128
        bt = cm.block_table_tensor
        nblks = [
            max((int(seq_lens_cpu[r]) + g - 1) // g, 1) for r in range(num_reqs)
        ]
        bt_cpu = bt[:num_reqs, : max(nblks)].cpu()  # one small D2H per build
        max_page = int(bt_cpu.max())
        self._ensure_map_size(max_page + 1)

        live_pages: set[int] = set()
        new_fill: dict[int, int] = {}  # scheduled THIS step (stored later)
        sinks_now: list[int] = []
        committed_flush: list[int] = []
        committed_seen: set[int] = set()
        for r in range(num_reqs):
            row = bt_cpu[r]
            n_r = nblks[r]
            live_pages.update(int(x) for x in row[:n_r].tolist())
            c0 = int(computed_cpu[r])
            s1 = int(seq_lens_cpu[r])
            # COMMIT-gated flush (spec-decode-safe): positions < c0 are
            # accepted AND already stored by prior forwards. Scheduled
            # positions >= c0 may still be REJECTED and rewritten in the
            # pool next step; packing to int4 is permanent, so a tile is
            # only packable once fully below c0 (mirrors the dense
            # kvarn_attn builder's committed-boundary rule). Walk BACKWARD
            # from the committed boundary while tiles still hold pool
            # slots: flushes are prompt + in-order, so the first slotless
            # tile ends the walk. Sinks stay pool-resident (tile 0 only).
            for b in range(c0 // g - 1, -1, -1):
                p = int(row[b])
                if (p not in self._slot_of or p in committed_seen
                        or p in self._sinks):
                    break
                committed_seen.add(p)
                committed_flush.append(p)
            if c0 >= g:
                p0 = int(row[0])
                if p0 in self._sinks:
                    self._cfull.add(p0)  # committed-full sink: reclaimable
            if s1 <= c0:
                continue
            if c0 == 0:
                sinks_now.append(int(row[0]))
            for b in range(c0 // g, (s1 - 1) // g + 1):
                new_fill[int(row[b])] = min(s1 - b * g, g)

        # 1) reclaim: tracked pages no longer live. Only committed-full
        #    pages may be packed (a dead request's last tiles can hold
        #    rejected spec tokens); non-sinks flush promptly below, so a
        #    stale pool page is either a committed sink (flush: a prefix-
        #    cache hit may read it) or uncommitted (discard: never hashed).
        stale = [p for p in self._slot_of if p not in live_pages]
        self._flush([p for p in stale if p in self._cfull])
        for p in stale:
            self._release(p)

        #    flush live fully-COMMITTED non-sink tiles (never merely
        #    scheduled-full: spec tokens can be rejected and rewritten)
        self._flush(committed_flush)

        # 2) merge this step's scheduled fills + sink marks
        self._fill.update(new_fill)
        self._sinks.update(sinks_now)

        # 3) allocate slots for newly written pages
        todo = [p for p in new_fill if p not in self._slot_of]
        if todo:
            if len(todo) > len(self._free):
                # pressure: evict committed-full pages not being written this
                # step (committed sinks as last resort). NEVER evict a live
                # partial/uncommitted tile (pool rewrites would be lost).
                for p in list(self._slot_of):
                    if len(todo) <= len(self._free):
                        break
                    if p in new_fill:
                        continue
                    if p in self._cfull:
                        self._flush([p])
                    elif p not in live_pages:
                        self._release(p)
            assert len(todo) <= len(self._free), (
                f"kvarn sparse pool exhausted: need {len(todo)}, "
                f"free {len(self._free)} of {self.pool_slots}")
            pages_t, slots_t = [], []
            for p in todo:
                s = self._free.pop()
                self._slot_of[p] = s
                pages_t.append(p)
                slots_t.append(s)
            idx = torch.tensor(pages_t, dtype=torch.long, device=self._dev)
            val = torch.tensor(slots_t, dtype=torch.int32, device=self._dev)
            self.block_to_slot[idx] = val

    def _flush(self, pages: list[int]) -> None:
        pages = [p for p in pages if p in self._slot_of]
        if not pages:
            return
        slots = [self._slot_of[p] for p in pages]
        for layer in self.layers:
            layer.flush_pages(pages, slots)
        for p in pages:
            self._release(p)

    def _release(self, page: int) -> None:
        s = self._slot_of.pop(page, None)
        if s is not None:
            self._free.append(s)
            self.block_to_slot[page] = -1
        self._fill.pop(page, None)
        self._sinks.discard(page)
        self._cfull.discard(page)


# ── module API used by the patched model / impl ─────────────────────────────


def get_group(vllm_config) -> KVarNSparseGroup:
    global _STATE
    if _STATE is None:
        _STATE = KVarNSparseGroup(vllm_config)
    return _STATE


def kernel_consts(cfg):
    return dict(
        KBITS=cfg.key_bits, VBITS=cfg.value_bits,
        K_PACKED_OFF=cfg.k_packed_offset, K_SCOL_OFF=cfg.k_s_col_offset,
        K_ZP_OFF=cfg.k_zp_offset, K_SROW_OFF=cfg.k_s_row_offset,
        V_PACKED_OFF=cfg.v_packed_offset, V_SCOL_OFF=cfg.v_s_col_offset,
        V_SROW_OFF=cfg.v_s_row_offset, V_ZP_OFF=cfg.v_zp_offset,
    )


def build_ws_decode(state: KVarNSparseLayer, group: KVarNSparseGroup,
                    topk_idx: torch.Tensor, block_table: torch.Tensor,
                    total_q: int, q_len: int):
    """Decode/verify workspace. Capture-safe (fixed shapes per graph)."""
    cfg = state.cfg
    hk, d, g = state.hk, state.d, cfg.group
    topk = topk_idx.shape[-1]
    ws, _, bt_ws = group.ws_buffers(hk, d, g, topk, block_table.shape[1])
    grid = (hk * total_q * topk,)
    _ws_build_topk_kernel[grid](
        topk_idx, block_table, group.block_to_slot,
        state.kv_cache, state.pool_k, state.pool_v,
        ws, bt_ws,
        total_q, q_len,
        topk_idx.stride(0), topk_idx.stride(1),
        block_table.stride(0),
        state.kv_cache.stride(0), state.kv_cache.stride(1),
        state.pool_k.stride(0), state.pool_k.stride(1), state.pool_k.stride(2),
        ws.stride(0), ws.stride(1), ws.stride(2), ws.stride(3), ws.stride(4),
        TOPK=topk, HK=hk, G=g, D=d,
        **kernel_consts(cfg),
        num_warps=4, num_stages=2,
    )
    return ws, bt_ws[: block_table.shape[0]]


def build_ws_prefill(state: KVarNSparseLayer, group: KVarNSparseGroup,
                     block_table: torch.Tensor, num_prefills: int):
    """Prefill context workspace (eager path): materialize ALL context blocks
    of the prefilling requests; bt_ws rows map logical -> ws pages.
    Prefill rows are the LAST ``num_prefills`` of the (decode-first) batch,
    matching ``p.block_table = block_table[num_decodes:]``, so the stashed
    CPU seq_lens tail lines up with rows here."""
    cfg = state.cfg
    hk, d, g = state.hk, state.d, cfg.group
    _, ws, _ = group.ws_buffers(hk, d, g, group._topk, block_table.shape[1])
    dev = block_table.device
    seq_lens_cpu = group._seqlens_cpu[-num_prefills:]
    counts = [max((int(seq_lens_cpu[r]) + g - 1) // g, 1)
              for r in range(num_prefills)]
    total = sum(counts)
    ws = group.grow_prefill_ws(total, hk, d, g)
    pages = torch.full((total,), -1, dtype=torch.int32, device=dev)
    bt_ws = torch.zeros(num_prefills, block_table.shape[1],
                        dtype=torch.int32, device=dev)
    off = 0
    for r, n_r in enumerate(counts):
        pages[off:off + n_r] = block_table[r, :n_r]
        bt_ws[r, :n_r] = torch.arange(off, off + n_r, dtype=torch.int32,
                                      device=dev)
        off += n_r
    _ws_build_pages_kernel[(total,)](
        pages, group.block_to_slot,
        state.kv_cache, state.pool_k, state.pool_v, ws,
        state.kv_cache.stride(0), state.kv_cache.stride(1),
        state.pool_k.stride(0), state.pool_k.stride(1), state.pool_k.stride(2),
        ws.stride(0), ws.stride(1), ws.stride(2), ws.stride(3), ws.stride(4),
        HK=hk, G=g, D=d,
        **kernel_consts(cfg),
        num_warps=4, num_stages=2,
    )
    return ws, bt_ws


def rotate_heads(x: torch.Tensor, H16: torch.Tensor) -> torch.Tensor:
    """[T, H, D] @ H (per head, shared rotation), preserving dtype."""
    return torch.matmul(x.to(torch.float16), H16).to(x.dtype)


# ── Phase 2C: fused decode (dequant + q-rotation inside the attend split) ───
# Eliminates the decode workspace round-trip (~410 MB/step @topk16/57L) and
# the q-rotate launch. Each split program dequants ONLY its own (chunk,
# kv-head) record slices — no duplicate all-head dequant like the ws builder.
# The stock merge kernel is reused; output un-rotation remains one matmul.


@triton.jit
def _deq_k_rec(rec_ptr, off_n, off_d,
               K_PACKED_OFF: tl.constexpr, K_SCOL_OFF: tl.constexpr,
               K_ZP_OFF: tl.constexpr, K_SROW_OFF: tl.constexpr,
               G: tl.constexpr, KBITS: tl.constexpr):
    """K tile as [D, N] (dim-major, matches the attend kernel's k layout)."""
    kpack = 8 // KBITS
    kmask = (1 << KBITS) - 1
    b = tl.load(rec_ptr + K_PACKED_OFF
                + off_d[:, None] * (G // kpack) + (off_n[None, :] // kpack))
    qv = (b >> ((off_n[None, :] % kpack) * KBITS)) & kmask
    s_col = tl.load((rec_ptr + K_SCOL_OFF).to(tl.pointer_type(tl.float16))
                    + off_d).to(tl.float32)
    zp = tl.load((rec_ptr + K_ZP_OFF).to(tl.pointer_type(tl.float16))
                 + off_d).to(tl.float32)
    s_row = tl.load((rec_ptr + K_SROW_OFF).to(tl.pointer_type(tl.float16))
                    + off_n).to(tl.float32)
    return (qv.to(tl.float32) * s_col[:, None] + zp[:, None]) * s_row[None, :]


@triton.jit
def _deq_v_rec(rec_ptr, off_n, off_d,
               V_PACKED_OFF: tl.constexpr, V_SCOL_OFF: tl.constexpr,
               V_SROW_OFF: tl.constexpr, V_ZP_OFF: tl.constexpr,
               G: tl.constexpr, VBITS: tl.constexpr, D: tl.constexpr):
    """V tile as [N, D] (pos-major, matches the attend kernel's v layout)."""
    vpack = 8 // VBITS
    vmask = (1 << VBITS) - 1
    b = tl.load(rec_ptr + V_PACKED_OFF
                + off_n[:, None] * (D // vpack) + (off_d[None, :] // vpack))
    qv = (b >> ((off_d[None, :] % vpack) * VBITS)) & vmask
    s_col = tl.load((rec_ptr + V_SCOL_OFF).to(tl.pointer_type(tl.float16))
                    + off_d).to(tl.float32)
    s_row = tl.load((rec_ptr + V_SROW_OFF).to(tl.pointer_type(tl.float16))
                    + off_n).to(tl.float32)
    zp = tl.load((rec_ptr + V_ZP_OFF).to(tl.pointer_type(tl.float16))
                 + off_n).to(tl.float32)
    return (qv.to(tl.float32) * s_row[:, None] + zp[:, None]) * s_col[None, :]


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _kvarn_fused_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim] UNROTATED
    h_ptr,  # Hadamard H: [D, D] fp16
    rec_cache_ptr,  # uint8 [num_blocks, hk, REC]
    b2s_ptr,
    poolk_ptr,  # fp16 [POOL, G, hk, D]
    poolv_ptr,
    t_ptr,
    o_ptr,
    lse_ptr,
    block_table_ptr,
    seq_lens,
    total_q,
    gqa_group_size,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn, stride_qh, stride_qd,
    stride_rec_blk, stride_rec_h,
    stride_pool_b, stride_pool_t, stride_pool_h,
    stride_th, stride_tn, stride_tk,
    stride_o_c, stride_o_b, stride_o_h, stride_o_d,
    stride_l_c, stride_l_b, stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    D: tl.constexpr,  # head_dim == 128
    KBITS: tl.constexpr, VBITS: tl.constexpr,
    K_PACKED_OFF: tl.constexpr, K_SCOL_OFF: tl.constexpr,
    K_ZP_OFF: tl.constexpr, K_SROW_OFF: tl.constexpr,
    V_PACKED_OFF: tl.constexpr, V_SCOL_OFF: tl.constexpr,
    V_SROW_OFF: tl.constexpr, V_ZP_OFF: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    topk_idx = tl.load(idx_base + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, D)
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, D),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, D),
        order=(1, 0),
    )
    q0 = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # fold q-rotation: q = q0 @ H (fp16 operands, fp32 accumulate — matches
    # the eager rotate_heads matmul)
    H = tl.load(h_ptr + off_d[:, None] * D + off_d[None, :])
    q = tl.dot(q0.to(tl.float16), H).to(q0.dtype)

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        slot = tl.load(b2s_ptr + page)
        if slot >= 0:
            pool_base = slot.to(tl.int64) * stride_pool_b + pid_kh * stride_pool_h
            k = tl.load(
                poolk_ptr + pool_base
                + off_n[None, :] * stride_pool_t + off_d[:, None]
            ).to(q.dtype)
            v = tl.load(
                poolv_ptr + pool_base
                + off_n[:, None] * stride_pool_t + off_d[None, :]
            ).to(q.dtype)
        else:
            rec = rec_cache_ptr + page * stride_rec_blk + pid_kh * stride_rec_h
            k = _deq_k_rec(rec, off_n, off_d,
                           K_PACKED_OFF=K_PACKED_OFF, K_SCOL_OFF=K_SCOL_OFF,
                           K_ZP_OFF=K_ZP_OFF, K_SROW_OFF=K_SROW_OFF,
                           G=BLOCK_SIZE_K, KBITS=KBITS).to(q.dtype)
            v = _deq_v_rec(rec, off_n, off_d,
                           V_PACKED_OFF=V_PACKED_OFF, V_SCOL_OFF=V_SCOL_OFF,
                           V_SROW_OFF=V_SROW_OFF, V_ZP_OFF=V_ZP_OFF,
                           G=BLOCK_SIZE_K, VBITS=VBITS, D=D).to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, D),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


# ── Phase 2D: fused prefill (dequant + q-rot + out-unrot inside the attend) ─
# Kills the prefill workspace entirely: build_ws_prefill wrote the WHOLE
# dequantized context per layer per chunk (O(ctx²·layers) traffic; ws_pre
# grows ~1 MB/K-ctx of max_model_len, charged against KV). The attend itself
# is topk-sparse, so in-kernel dequant makes prefill traffic topk-bounded
# like decode. BLOCK_SIZE_QH = next_pow2(gqa) >= 16, so unlike the decode
# merge path BOTH rotations fold into the kernel (no extra launches).


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _kvarn_fused_prefill_kernel(
    q_ptr,  # [total_q, num_heads, head_dim] UNROTATED
    h_ptr,  # Hadamard H: [D, D] fp16 (symmetric orthonormal)
    rec_cache_ptr,  # uint8 [num_blocks, hk, REC]
    b2s_ptr,
    poolk_ptr,  # fp16 [POOL, G, hk, D]
    poolv_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim] (UNROTATED result)
    block_table_ptr,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    gqa_group_size,
    max_topk,
    sm_scale,
    stride_qn, stride_qh, stride_qd,
    stride_rec_blk, stride_rec_h,
    stride_pool_b, stride_pool_t, stride_pool_h,
    stride_th, stride_tn, stride_tk,
    stride_on, stride_oh, stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,  # == 1 (mirrors stock launch)
    BLOCK_SIZE_K: tl.constexpr,  # == 128
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    D: tl.constexpr,  # head_dim == 128 exactly (dequant helpers assume it)
    KBITS: tl.constexpr, VBITS: tl.constexpr,
    K_PACKED_OFF: tl.constexpr, K_SCOL_OFF: tl.constexpr,
    K_ZP_OFF: tl.constexpr, K_SROW_OFF: tl.constexpr,
    V_PACKED_OFF: tl.constexpr, V_SCOL_OFF: tl.constexpr,
    V_SROW_OFF: tl.constexpr, V_ZP_OFF: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= q_len:  # block_size_q == 1: one program per query row
        return
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, D)

    t_ptr_j = t_ptr + (q_start + pid_q) * stride_tn + pid_kh * stride_th
    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, D),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q, 0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, D),
        order=(2, 1, 0),
    )
    q0 = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q0 = tl.reshape(q0, BLOCK_SIZE_QH, D)
    # fold q-rotation (fp16 operands, fp32 accumulate == eager rotate_heads)
    H = tl.load(h_ptr + off_d[:, None] * D + off_d[None, :])
    q = tl.dot(q0.to(tl.float16), H).to(q0.dtype)
    # causal offset of this query row vs kv position (stock semantics)
    off_q = (
        tl.arange(0, BLOCK_SIZE_Q)[:, None]
        + pid_q * BLOCK_SIZE_Q
        + prefix_len
        - tl.arange(0, BLOCK_SIZE_K)[None, :]
    )
    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_QH, D), dtype=tl.float32)
    for _ in range(real_topk):
        blk = tl.load(t_ptr_j).to(tl.int32)
        t_ptr_j = t_ptr_j + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < seq_len
        slot = tl.load(b2s_ptr + page)
        if slot >= 0:
            pool_base = slot.to(tl.int64) * stride_pool_b + pid_kh * stride_pool_h
            k = tl.load(
                poolk_ptr + pool_base
                + off_n[None, :] * stride_pool_t + off_d[:, None]
            ).to(q.dtype)
            v = tl.load(
                poolv_ptr + pool_base
                + off_n[:, None] * stride_pool_t + off_d[None, :]
            ).to(q.dtype)
        else:
            rec = rec_cache_ptr + page * stride_rec_blk + pid_kh * stride_rec_h
            k = _deq_k_rec(rec, off_n, off_d,
                           K_PACKED_OFF=K_PACKED_OFF, K_SCOL_OFF=K_SCOL_OFF,
                           K_ZP_OFF=K_ZP_OFF, K_SROW_OFF=K_SROW_OFF,
                           G=BLOCK_SIZE_K, KBITS=KBITS).to(q.dtype)
            v = _deq_v_rec(rec, off_n, off_d,
                           V_PACKED_OFF=V_PACKED_OFF, V_SCOL_OFF=V_SCOL_OFF,
                           V_SROW_OFF=V_SROW_OFF, V_ZP_OFF=V_ZP_OFF,
                           G=BLOCK_SIZE_K, VBITS=VBITS, D=D).to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))
        qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
        qk += tl.dot(q, k) * sm_scale_log2e
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
    # fold output un-rotation (H symmetric orthonormal: unrotate == @H)
    acc_o = tl.dot(acc_o.to(tl.float16), H)
    acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, D),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(pid_q * BLOCK_SIZE_Q, 0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, D),
        order=(2, 1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


def kvarn_sparse_attn_prefill_fused(
    q: torch.Tensor,  # [total_q, num_heads, head_dim] UNROTATED
    state: KVarNSparseLayer,
    group: KVarNSparseGroup,
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim] (un-rotated result)
) -> None:
    """Drop-in for minimax_m3_sparse_attn on the kvarn tile cache: rotated-
    frame attention with q-rotation AND output un-rotation fused in-kernel
    (BLOCK_SIZE_QH >= 16 satisfies the tl.dot M constraint). No workspace."""
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        _sparse_attn_num_stages_kwarg,
    )

    total_q, num_heads, head_dim = q.shape
    assert head_dim == state.d == 128, "fused kvarn prefill assumes head_dim 128"
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    cfg = state.cfg
    grid = (max_query_len, num_kv_heads, batch)
    _kvarn_fused_prefill_kernel[grid](
        q, state._H16,
        state.kv_cache, group.block_to_slot, state.pool_k, state.pool_v,
        topk_idx, output, block_table,
        cu_seqlens_q, seq_lens, prefix_lens,
        gqa_group_size, topk, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        state.kv_cache.stride(0), state.kv_cache.stride(1),
        state.pool_k.stride(0), state.pool_k.stride(1), state.pool_k.stride(2),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=cfg.group,
        D=head_dim,
        **kernel_consts(cfg),
        **_sparse_attn_num_stages_kwarg(),
    )


def kvarn_sparse_attn_decode_fused(
    q: torch.Tensor,  # [total_q, num_heads, head_dim] UNROTATED
    state: KVarNSparseLayer,
    group: KVarNSparseGroup,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim] (un-rotated result)
    decode_query_len: int,
) -> None:
    """Drop-in for minimax_m3_sparse_attn_decode on the kvarn tile cache.

    Rotated-frame attention with q-rotation fused into the split kernel and
    output un-rotation as one matmul after the stock merge.
    """
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        _merge_topk_attn_out_kernel,
        _sparse_attn_num_stages_kwarg,
    )
    from vllm.platforms import current_platform

    total_q, num_heads, head_dim = q.shape
    assert head_dim == state.d == 128, "fused kvarn decode assumes head_dim 128"
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_pdl = current_platform.is_arch_support_pdl()
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, total_q * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, head_dim, dtype=q.dtype, device=q.device
    )
    lse_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device
    )
    cfg = state.cfg
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _kvarn_fused_decode_kernel[grid](
        q, state._H16,
        state.kv_cache, group.block_to_slot, state.pool_k, state.pool_v,
        topk_idx, o_partial, lse_partial, block_table, seq_lens,
        total_q, gqa_group_size, max_topk, sm_scale, decode_query_len,
        q.stride(0), q.stride(1), q.stride(2),
        state.kv_cache.stride(0), state.kv_cache.stride(1),
        state.pool_k.stride(0), state.pool_k.stride(1), state.pool_k.stride(2),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=cfg.group,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        D=head_dim,
        **kernel_consts(cfg),
        USE_PDL=use_pdl,
        **_sparse_attn_num_stages_kwarg(),
        **pdl_launch,
    )
    o_rot = torch.empty_like(output)
    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial, lse_partial, o_rot, head_dim,
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        o_rot.stride(0), o_rot.stride(1), o_rot.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
    output[:] = rotate_heads(o_rot, state._H16)
