"""b12x CuTe MSA main attention for MiniMax-M3 on SM120/SM121 (GB10).

Replaces the Triton block-sparse fallback with b12x's CuTe MSA kernels for BOTH
phases:
  * decode  -> b12x paged MSA decode (CUDA-graph captured; page_size 128)
  * prefill -> b12x MSA extend / union-tile attention (eager)

The q2k block lists come from the b12x MSA *indexer* (companion mod
``b12x_indexer``); they are already in b12x's contract
(``[num_kv_heads, q_rows, 16]`` int32, ascending, ``-1`` padded at the end), so
they feed straight into both the decode and extend kernels -- no Triton indexer,
no reorder.

Main KV cache: vLLM ``[blocks, 2, 128, kv_heads, 128]`` -> ``unbind(1)`` ->
b12x k/v ``[blocks, 128, kv_heads, 128]``. fp8 e4m3 KV uses k/v_descale =
``layer._k_scale``/``layer._v_scale`` (the per-tensor scale the cache was
written with -> correct by construction). nvfp4 KV uses the uint8 packed cache
(e2m1 + per-16 e4m3 block scales, kv_quant="nvfp4") -> b12x reads the block
scales from the cache itself (no per-tensor descale).
"""
from __future__ import annotations

import os
import torch

from vllm.forward_context import get_forward_context
from vllm.config import get_current_vllm_config
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseImpl,
)
from b12x.attention.paged.tuning.policy import (
    attn_meta_once_enabled,
    debug_paged_policy_enabled,
)

_USE_ATTN_META_ONCE = attn_meta_once_enabled()
# Phase 4 meta-once ownership: the FIRST-constructed MSA impl (= first MSA
# layer in model order, which also executes first every step) owns the
# per-step decode replay-metadata copy + runtime chunk update; the other 56
# layers share the same scratch storage + identical plans and skip it.
_MSA_META_OWNER_TAKEN = [False]
from b12x.integration.attention import (
    B12XPagedAttentionScratchCaps,
    plan_paged_attention_scratch,
    paged_attention_forward,
    create_paged_plan,
)


# Shared scratch storage across ALL sparse layers: b12x scratch is transient,
# caller-owned workspace -- attention runs sequentially layer-by-layer, so one
# storage buffer per (mode, shape) is reused by every layer instead of
# allocating 57 separate arenas (the dominant memory consumer is the per-layer
# tmp/lse partial-row + page-table buffers, which scale with max_model_len).
# The PLAN stays per-layer (it owns the q2k data_ptr + decode-graph metadata
# cache); only the big uint8 storage is shared. Decode storage sharing is
# graph-safe: per-layer [metadata-write, kernel] are sequential in the captured
# graph and the decode metadata is identical across sparse layers (same group).
_DEC_STORE: dict = {}
_EXT_STORE: dict = {}
_VFY_STORE: dict = {}


def _shared_storage(store: dict, key, plan, device):
    if key not in store:
        store[key] = tuple(
            torch.zeros(s, dtype=d, device=device)
            for s, d in plan.shapes_and_dtypes())
    return store[key]


class MiniMaxM3SparseB12xImpl(MiniMaxM3SparseImpl):
    """b12x MSA attention (decode paged + prefill extend). No Triton."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # nvfp4 main KV: uint8 packed cache (e2m1 + per-16 e4m3 block scales).
        # The base sets use_fp8_kv=False for nvfp4 (run.sh patch), so the cache
        # stays uint8 (NOT viewed as fp8) and _descales returns None -- b12x
        # reads the per-16 block scales from the cache itself. kv_quant routes
        # the planner/kernel to the native nvfp4 load+expand path.
        self.use_nvfp4 = self.kv_cache_dtype == "nvfp4"
        self._kv_quant = "nvfp4" if self.use_nvfp4 else "none"
        self._attn_meta_owner = not _MSA_META_OWNER_TAKEN[0]
        _MSA_META_OWNER_TAKEN[0] = True
        cfg = get_current_vllm_config()
        # The extend plan's max_total_q PINS the prefill compile: it MUST be the
        # server's max_num_batched_tokens (the only knob we let drive the JIT),
        # never a hardcoded constant. Fail loudly if the config is unavailable
        # rather than silently pinning the wrong capacity.
        self._max_q = max(int(cfg.scheduler_config.max_num_batched_tokens), 1)
        self._max_seqs = max(int(cfg.scheduler_config.max_num_seqs), 1)
        # fp8 descale buffers must cover the spec-decode VERIFY batch: the
        # multirow verify plan treats every verify ROW as its own request
        # (rows = max_num_seqs * (1 + num_speculative_tokens)). Sizing by
        # max_num_seqs alone under-allocates; the [:rows] slice then silently
        # returns fewer rows and the kernel reads descales OUT OF BOUNDS for
        # rows >= max_num_seqs (garbage scales -> garbage verify attention).
        # fp8-only: nvfp4/bf16 pass no descale arrays.
        _spec = getattr(cfg, "speculative_config", None)
        if isinstance(_spec, dict):
            _num_spec = int(_spec.get("num_speculative_tokens", 0) or 0)
        else:
            _num_spec = (
                int(getattr(_spec, "num_speculative_tokens", 0) or 0)
                if _spec is not None else 0
            )
        self._max_desc_rows = self._max_seqs * max(_num_spec + 1, 1)
        _bs = int(cfg.cache_config.block_size)
        self._max_pt_width = (int(cfg.model_config.max_model_len) + _bs - 1) // _bs
        # (batch, width, ncb, qdt, kvdt) -> (plan, scratch, cu). Full plan key:
        # width/capacity/dtype changes must build a NEW plan (batch-only keying
        # would silently reuse a stale plan/replay state).
        self._dec: dict[tuple, tuple] = {}
        # (batch, q_len, width, ncb, qdt, kvdt) -> (plan, scratch, cu) native verify
        self._vfy: dict[tuple, tuple] = {}
        # Native mode="verify" block-sparse spec verify (default). Set
        # B12X_MSA_NATIVE_VERIFY=0 at launch to fall back to the multi-row
        # decode path (no rebuild) if needed.
        self._native_verify = os.environ.get("B12X_MSA_NATIVE_VERIFY", "1") != "0"
        # Debug: run BOTH native + multi-row verify, log per-qo-tile divergence,
        # and emit the known-good multi-row output. Requires --enforce-eager
        # (host-side compare/print can't run inside a captured graph). Localizes
        # which layer / qo-tile / context the native verify diverges at.
        self._verify_debug = os.environ.get("B12X_MSA_VERIFY_DEBUG", "0") != "0"
        self._vfy_dbg_n = 0
        self._vfy_meta: dict[tuple, dict] = {} # (batch,q_len,width) -> verify metadata buffers
        self._ext_plan = None                 # fixed-capacity extend plan (one compile)
        self._kd: torch.Tensor | None = None
        self._vd: torch.Tensor | None = None

    # ---- fp8 descale buffers (persistent; built once, capture-safe) ----
    def _descales(self, layer, batch: int, device):
        if not self.use_fp8_kv:
            return None, None
        if self._kd is None:
            ks = getattr(layer, "_k_scale", None)
            vs = getattr(layer, "_v_scale", None)
            ks = ks.detach().float().reshape(-1)[0] if isinstance(ks, torch.Tensor) \
                else torch.ones((), device=device)
            vs = vs.detach().float().reshape(-1)[0] if isinstance(vs, torch.Tensor) \
                else torch.ones((), device=device)
            n = max(self._max_desc_rows, 1)
            self._kd = ks.to(device).reshape(1, 1).expand(n, self.num_kv_heads).contiguous()
            self._vd = vs.to(device).reshape(1, 1).expand(n, self.num_kv_heads).contiguous()
        if batch > int(self._kd.shape[0]):
            raise RuntimeError(
                f"b12x fp8 descale buffer too small: need {batch} rows, have "
                f"{int(self._kd.shape[0])} (max_seqs*verify_q_len misconfigured)")
        return self._kd[:batch], self._vd[:batch]

    # ---- decode scratch/plan (graph-captured), per decode batch ----
    def _decode_ctx(self, batch, width, ncb, qdt, kvdt, device):
        # torch.dtype is hashable -- keep raw dtypes in the hot per-call key
        # (str() would burn host time every layer every step).
        key = (int(batch), int(width), int(ncb), qdt, kvdt)
        if key not in self._dec:
            if debug_paged_policy_enabled():
                import sys
                print(f"# b12x_decode_ctx site=msa.decode key={key}",
                      file=sys.stderr, flush=True)
            caps = B12XPagedAttentionScratchCaps(
                device=device, mode="decode", dtype=qdt, kv_dtype=kvdt,
                num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size, head_dim_vo=self.head_size, page_size=128,
                max_total_q=batch, max_batch=batch, max_page_table_width=width,
                max_work_items=batch * 512, max_partial_rows=batch * 512,
                num_cache_pages=ncb, use_cuda_graph=True, msa_block_sparse=True,
                kv_quant=self._kv_quant)
            plan = plan_paged_attention_scratch(caps)
            # max_cache_page_count = the WORST-CASE pages a SINGLE request can
            # reference, which is bounded by the page-table width (a request's
            # cache_seqlen <= max_model_len = width*page_size), NOT the total
            # cache block count ncb. Passing ncb makes the replay-state worst
            # case demand ncb pages from a width-wide page table -> planner
            # "page_table width is smaller than required by cache_seqlens" once
            # ncb > width (a full cache + a verify ctx created during serving).
            plan.prepare_decode_graph_replay_state(
                batch=batch, max_page_table_width=width,
                max_cache_page_count=min(int(ncb), int(width)))
            scratch = _shared_storage(
                _DEC_STORE, (batch, width, str(qdt), str(kvdt)), plan, device)
            cu = torch.arange(batch + 1, dtype=torch.int32, device=device)
            self._dec[key] = (plan, scratch, cu)
        return self._dec[key]

    def _decode(self, layer, q, out, k_cache, v_cache, d, q2k, ncb):
        device = q.device
        batch = int(d.seq_lens.shape[0])
        plan, scratch, cu = self._decode_ctx(
            batch, int(d.block_table.shape[1]), ncb, q.dtype, k_cache.dtype, device)
        kd, vd = self._descales(layer, batch, device)
        binding = plan.bind(
            scratch=scratch, q=q, k_cache=k_cache, v_cache=v_cache, output=out,
            page_table=d.block_table, cache_seqlens=d.seq_lens, cu_seqlens_q=cu,
            q2k_indices=q2k, k_descale=kd, v_descale=vd,
            skip_replay_metadata_update=(
                _USE_ATTN_META_ONCE and not self._attn_meta_owner))
        paged_attention_forward(binding=binding)

    # ---- spec-decode verify (q_len>1) = MULTI-ROW block-sparse DECODE.
    # b12x MSA block-sparse supports decode/extend only (not mode="verify"), so
    # the nd = batch*q_len verify query positions are scored as nd independent
    # captured decode rows: each row carries its own causal cache_seqlen
    # (seq_len_i - q_len + j + 1) and its own selected blocks (the indexer's
    # per-query q2k [num_kv_heads, nd, topk]). Each verify query's causal
    # attention is fully determined by its (q2k, cache_seqlen, q-vector) -- the
    # later same-request verify tokens sit beyond cache_seqlen and are masked --
    # so this is identical to a fused verify and reuses the supported +
    # cudagraph-captured decode path (_decode_ctx(nd) gives cu=arange(nd+1)).
    def _verify_meta(self, d, batch, q_len, nd, width, device):
        key = (int(batch), int(q_len), int(width))
        m = self._vfy_meta.get(key)
        if m is None:
            # Persistent (graph-safe) per-query metadata buffers: rebuilt in
            # place every step by captured ops reading the persistent decode
            # block_table/seq_lens. Tiny: nd*width int32 + nd int32.
            m = self._vfy_meta[key] = dict(
                pt=torch.zeros((nd, width), dtype=torch.int32, device=device),
                seqlens=torch.zeros((nd,), dtype=torch.int32, device=device),
                jr=torch.arange(q_len, device=device, dtype=torch.int32).view(1, q_len))
        # request i's page table replicated across its q_len query positions
        m["pt"].view(batch, q_len, width)[:] = \
            d.block_table.to(torch.int32).view(batch, 1, width)
        # per-query causal length: seq_len_i - q_len + j + 1 (bit-exact w/ native)
        base = (d.seq_lens.to(torch.int32) - q_len).view(batch, 1)
        m["seqlens"].copy_((base + m["jr"] + 1).reshape(nd))
        return m["pt"], m["seqlens"]

    def _verify(self, layer, q, out, k_cache, v_cache, d, q2k, ncb):
        if self._verify_debug:
            return self._verify_compare(layer, q, out, k_cache, v_cache, d, q2k, ncb)
        if self._native_verify:
            self._verify_native(layer, q, out, k_cache, v_cache, d, q2k, ncb)
        else:
            self._verify_multirow(layer, q, out, k_cache, v_cache, d, q2k, ncb)

    def _verify_compare(self, layer, q, out, k_cache, v_cache, d, q2k, ncb):
        on = torch.empty_like(out)
        self._verify_native(layer, q, on, k_cache, v_cache, d, q2k, ncb)
        self._verify_multirow(layer, q, out, k_cache, v_cache, d, q2k, ncb)  # known-good -> emit
        batch = int(d.seq_lens.shape[0]); q_len = int(d.decode_query_len)
        L = int(d.seq_lens.max().item())
        cs = []
        for j in range(q_len):
            rows = list(range(j, batch * q_len, q_len))  # request-major: r*q_len+j
            cj = torch.nn.functional.cosine_similarity(
                on[rows].flatten().float(), out[rows].flatten().float(), dim=0).item()
            cs.append(cj)
        if min(cs) < 0.999 or self._vfy_dbg_n < 12:
            ln = getattr(layer, "layer_name", getattr(layer, "prefix", "?"))
            sparse = "SPARSE" if L > 16 * 128 else "dense"
            print(f"[VFY-DBG n={self._vfy_dbg_n}] {ln} L={L}({sparse}) b={batch} qlen={q_len} "
                  f"q2k{tuple(q2k.shape)} nat-vs-multirow per-qo-cos="
                  f"{['%.4f' % c for c in cs]}", flush=True)
            self._vfy_dbg_n += 1

    # ---- native block-sparse VERIFY (mode="verify"): q_len queries per request
    # grouped via cu_seqlens_q; each qo-tile scores against its OWN q2k row
    # (kernel msa_q_row_idx = q_start + qo_tile_idx). page_table/cache_seqlens
    # are PER REQUEST (no per-query replication). Block-sparse KV is bounded by
    # topk, so the captured plan stays valid as context grows (the verify graph
    # state has no decode chunk-LUT). Validated captured+mutated in
    # tools/nvfp4-kv/msa_verify_test.py.
    def _verify_ctx(self, batch, q_len, width, ncb, qdt, kvdt, device):
        key = (int(batch), int(q_len), int(width), int(ncb), qdt, kvdt)
        if key not in self._vfy:
            if debug_paged_policy_enabled():
                import sys
                print(f"# b12x_decode_ctx site=msa.verify key={key}",
                      file=sys.stderr, flush=True)
            nd = batch * q_len
            caps = B12XPagedAttentionScratchCaps(
                device=device, mode="verify", dtype=qdt, kv_dtype=kvdt,
                num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size, head_dim_vo=self.head_size, page_size=128,
                max_total_q=nd, max_batch=batch, max_page_table_width=width,
                max_work_items=nd * 64, max_partial_rows=nd * 64,
                num_cache_pages=ncb, use_cuda_graph=True, msa_block_sparse=True,
                kv_quant=self._kv_quant)
            plan = plan_paged_attention_scratch(caps)
            plan.prepare_verify_graph_replay_state(
                batch=batch, q_len=q_len, max_page_table_width=width,
                max_cache_page_count=min(int(ncb), int(width)))
            scratch = _shared_storage(
                _VFY_STORE, (batch, q_len, width, str(qdt), str(kvdt)), plan, device)
            cu = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * int(q_len)
            self._vfy[key] = (plan, scratch, cu)
        return self._vfy[key]

    def _verify_native(self, layer, q, out, k_cache, v_cache, d, q2k, ncb):
        device = q.device
        batch = int(d.seq_lens.shape[0])
        q_len = int(d.decode_query_len)
        width = int(d.block_table.shape[1])
        plan, scratch, cu = self._verify_ctx(
            batch, q_len, width, ncb, q.dtype, k_cache.dtype, device)
        kd, vd = self._descales(layer, batch, device)
        binding = plan.bind(
            scratch=scratch, q=q, k_cache=k_cache, v_cache=v_cache, output=out,
            page_table=d.block_table, cache_seqlens=d.seq_lens, cu_seqlens_q=cu,
            q2k_indices=q2k, k_descale=kd, v_descale=vd)
        paged_attention_forward(binding=binding)

    def _verify_multirow(self, layer, q, out, k_cache, v_cache, d, q2k, ncb):
        device = q.device
        batch = int(d.seq_lens.shape[0])
        q_len = int(d.decode_query_len)
        nd = batch * q_len
        width = int(d.block_table.shape[1])
        plan, scratch, cu = self._decode_ctx(
            nd, width, ncb, q.dtype, k_cache.dtype, device)
        pt_v, seqlens_v = self._verify_meta(d, batch, q_len, nd, width, device)
        kd, vd = self._descales(layer, nd, device)
        binding = plan.bind(
            scratch=scratch, q=q, k_cache=k_cache, v_cache=v_cache, output=out,
            page_table=pt_v, cache_seqlens=seqlens_v, cu_seqlens_q=cu,
            q2k_indices=q2k, k_descale=kd, v_descale=vd)
        paged_attention_forward(binding=binding)

    # ---- prefill extend (union-tile), eager. FIXED capacity sized to the
    # server maxes (max_num_batched_tokens / max_num_seqs / max_model_len) so
    # the kernel JIT-compiles ONCE: the worklist length, page-table width and
    # total_q capacity (which are in the b12x compile key) are constant across
    # prefills, and prepare() re-plans the actual shapes into the fixed buffers.
    def _ext_ctx(self, q, k_cache, device):
        if self._ext_plan is None:
            gqa = max(self.num_heads // self.num_kv_heads, 1)
            caps = B12XPagedAttentionScratchCaps(
                device=device, mode="extend", dtype=q.dtype, kv_dtype=k_cache.dtype,
                num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size, head_dim_vo=self.head_size, page_size=128,
                max_total_q=self._max_q, max_batch=self._max_seqs,
                max_page_table_width=self._max_pt_width,
                max_work_items=(self._max_q * gqa + 15) // 16, max_partial_rows=0,
                num_cache_pages=int(k_cache.shape[0]), msa_block_sparse=True,
                kv_quant=self._kv_quant)
            self._ext_plan = plan_paged_attention_scratch(caps)
        key = ("ext", self._max_q, self._max_seqs, self._max_pt_width,
               str(q.dtype), str(k_cache.dtype), int(k_cache.shape[0]))
        scratch = _shared_storage(_EXT_STORE, key, self._ext_plan, device)
        return self._ext_plan, scratch

    def _extend(self, layer, q, out, k_cache, v_cache, p, q2k, ncb):
        device = q.device
        sp, scratch = self._ext_ctx(q, k_cache, device)
        cu = p.cu_seqlens_q.to(torch.int32)
        seqlens = p.seq_lens.to(torch.int32)
        kd, vd = self._descales(layer, int(p.block_table.shape[0]), device)
        binding = sp.bind(
            scratch=scratch, q=q, k_cache=k_cache, v_cache=v_cache, output=out,
            page_table=p.block_table, cache_seqlens=seqlens, cu_seqlens_q=cu,
            q2k_indices=q2k, k_descale=kd, v_descale=vd)
        paged_attention_forward(binding=binding)

    def forward(self, layer, query, kv_cache, topk_idx, output):
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return output  # profiling run; caches unbound
        main_md = md[layer.layer_name]
        decode_q2k, prefill_q2k = topk_idx
        nd = main_md.num_decode_tokens
        nt = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:nt].view(-1, self.num_heads, hd)
        out = output[:nt].view(-1, self.num_heads, hd)
        kvc = kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache
        k_cache, v_cache = kvc.unbind(1)  # [blocks, 128, kv_heads, hd]
        ncb = int(k_cache.shape[0])

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert decode_q2k is not None
            if d.decode_query_len != 1:
                # spec-decode verify batch (q_len = 1 + num_speculative_tokens)
                self._verify(layer, q[:nd], out[:nd], k_cache, v_cache, d, decode_q2k, ncb)
            else:
                self._decode(layer, q[:nd], out[:nd], k_cache, v_cache, d, decode_q2k, ncb)
        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert prefill_q2k is not None
            self._extend(layer, q[nd:], out[nd:], k_cache, v_cache, p, prefill_q2k, ncb)
        return output
