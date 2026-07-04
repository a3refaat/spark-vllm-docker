"""b12x CuTe dense full-attention backend for MiniMax-M3 (SM120/121 / GB10).

A *proper* vLLM v1 ``AttentionBackend`` registered under
``--attention-backend b12x`` (registry enum member ``B12X``). The dense M3
layers (0-2) use the stock vLLM ``Attention`` module, which resolves this
backend; the impl delegates to b12x's paged attention in DENSE mode
(``msa_block_sparse=False``, no q2k indices):

  * decode  -> b12x paged decode (CUDA-graph captured; page_size 128)
  * prefill -> b12x extend

No Triton. The attention math is entirely b12x's dense path -- this module is
only the vLLM backend/impl plumbing + the b12x PLAN/BIND/RUN wiring.

We reuse the M3 sparse metadata + builder (``MiniMaxM3SparseMetadata`` /
``MiniMaxM3SparseMetadataBuilder``): the decode/prefill split, block tables,
cu_seqlens and slot_mapping it produces are generic and identical to what the
b12x dense path needs. Only ``is_sparse()`` (False here) and the attend kernel
(full vs block-sparse) differ.

Main KV cache: vLLM ``[blocks, 2, 128, kv_heads, 128]`` -> ``unbind(1)`` ->
b12x k/v ``[blocks, 128, kv_heads, 128]``. fp8 e4m3/e5m2 KV uses
k/v_descale = ``layer._k_scale``/``layer._v_scale`` -- the same per-tensor
scale ``reshape_and_cache_flash`` divides by on write -> exact round trip.
"""
from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import is_quantized_kv_cache
from b12x.attention.paged.tuning.policy import (
    attn_meta_once_enabled,
    debug_paged_policy_enabled,
    dense_decode_split_kv_enabled,
)

_USE_ATTN_META_ONCE = attn_meta_once_enabled()
# Phase 4 meta-once ownership (see msa_attn.py): first dense layer owns the
# per-step replay-metadata copy + update; the other dense layers skip it.
_DENSE_META_OWNER_TAKEN = [False]
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
)

from b12x.integration.attention import (
    B12XPagedAttentionScratchCaps,
    create_paged_plan,
    paged_attention_forward,
    plan_paged_attention_scratch,
)


# Shared scratch storage across the dense layers: b12x scratch is transient,
# caller-owned workspace reused by every dense layer (one storage buffer per
# (mode, shape)) instead of a separate arena per layer. The PLAN stays
# per-layer; only the big uint8 storage is shared. (See b12x_msa_attn.py.)
_DEC_STORE: dict = {}
_EXT_STORE: dict = {}


def _shared_storage(store: dict, key, plan, device):
    if key not in store:
        store[key] = tuple(
            torch.zeros(s, dtype=d, device=device)
            for s, d in plan.shapes_and_dtypes())
    return store[key]


@triton.jit(do_not_specialize=["n_tokens", "s_blk", "s_pos", "s_h", "s_t", "s_sh"])
def _nvfp4_write_triton(
    src_ptr, cache_ptr, slot_ptr, n_tokens,
    s_blk, s_pos, s_h, s_t, s_sh,
    NH: tl.constexpr, HD: tl.constexpr, BLOCK_SIZE: tl.constexpr, NB: tl.constexpr,
):
    """One program per (token, kv_head): block-quant 128 channels -> 72 packed
    bytes (e2m1[0:64] | e4m3[64:72]). Single fused kernel replaces ~40 ATen ops
    per side -- critical for graph-decode GPU time (the unfused path costs ~19 ms
    vs ~1 ms/decode-step). Byte-exact to the ATen threshold-sum (fused_write_test).
    """
    pid = tl.program_id(0)
    t = pid // NH
    h = pid % NH
    if t >= n_tokens:
        return
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return                       # padding slot (unused/padded batch entry)
    blk = slot // BLOCK_SIZE
    pos = slot % BLOCK_SIZE
    bb = tl.arange(0, NB)[:, None]
    jj = tl.arange(0, 8)[None, :]
    base = t * s_t + h * s_sh
    ch_e = bb * 16 + 2 * jj
    xe = tl.load(src_ptr + base + ch_e).to(tl.float32)        # [NB,8] even chans
    xo = tl.load(src_ptr + base + ch_e + 1).to(tl.float32)    # [NB,8] odd chans
    amax = tl.maximum(tl.max(tl.abs(xe), axis=1), tl.max(tl.abs(xo), axis=1))
    se = tl.maximum(amax / 6.0, 1e-6).to(tl.float8e4nv)        # [NB] e4m3
    inv = (1.0 / se.to(tl.float32))[:, None]                  # [NB,1]
    qe = xe * inv
    me = tl.minimum(tl.abs(qe), 6.0)
    ce = ((me >= 0.25).to(tl.uint8) + (me >= 0.75).to(tl.uint8)
          + (me >= 1.25).to(tl.uint8) + (me >= 1.75).to(tl.uint8)
          + (me >= 2.5).to(tl.uint8) + (me >= 3.5).to(tl.uint8)
          + (me >= 5.0).to(tl.uint8))
    ne = ((qe < 0).to(tl.uint8) << 3) | ce
    qo = xo * inv
    mo = tl.minimum(tl.abs(qo), 6.0)
    co = ((mo >= 0.25).to(tl.uint8) + (mo >= 0.75).to(tl.uint8)
          + (mo >= 1.25).to(tl.uint8) + (mo >= 1.75).to(tl.uint8)
          + (mo >= 2.5).to(tl.uint8) + (mo >= 3.5).to(tl.uint8)
          + (mo >= 5.0).to(tl.uint8))
    no = ((qo < 0).to(tl.uint8) << 3) | co
    data = ne | (no << 4)                                     # [NB,8] packed
    out = blk * s_blk + pos * s_pos + h * s_h
    tl.store(cache_ptr + out + bb * 8 + jj, data)
    sb = tl.arange(0, NB)
    tl.store(cache_ptr + out + 64 + sb, se.to(tl.uint8, bitcast=True))


def nvfp4_block_quant_write(src, cache, slot_mapping, num_kv_heads, head_dim):
    """Block-quantize one KV side -> packed nvfp4 cache (shared dense+sparse).

    ``src``   : [T, num_kv_heads*head_dim] (or [T, nh, hd]) bf16/fp16 K or V.
    ``cache`` : [blocks, block_size, nh, hd//2 + hd//16] uint8 (one side from
                ``kv_cache.unbind(1)``); per head: e2m1[0:hd//2] | e4m3[hd//2:].
    Per-16-channel e4m3 block scale = amax/6 (matches b12x read dequant); e2m1
    magnitude by round-to-nearest threshold-sum (== the 3 low nibble bits, which
    are monotonic: 0,.5,1,1.5,2,3,4,6). Even channel -> low nibble. ONE fused
    Triton kernel (compile-once: constexprs fixed, grid is the only runtime arg).
    Capture-safe: a single CUDA kernel, no host LUT / .item() / scaled_fp4_quant.
    """
    nh, hd = num_kv_heads, head_dim
    src = src.reshape(-1, nh, hd).contiguous()
    T = src.shape[0]
    _nvfp4_write_triton[(T * nh,)](
        src, cache, slot_mapping, T,
        cache.stride(0), cache.stride(1), cache.stride(2),
        src.stride(0), src.stride(1),
        NH=nh, HD=hd, BLOCK_SIZE=cache.shape[1], NB=hd // 16,
    )


class B12XAttentionBackend(AttentionBackend):
    """b12x dense full-attention backend (page_size == 128)."""

    # bf16/fp16 queries; bf16 or fp8 (e4m3/e5m2) KV cache (b12x descales fp8).
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ]

    @staticmethod
    def get_name() -> str:
        return "B12X"

    @staticmethod
    def get_impl_cls() -> type["B12XAttentionImpl"]:
        return B12XAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseMetadataBuilder"]:
        # Generic decode/prefill split + block tables + cu_seqlens + slot_mapping.
        return MiniMaxM3SparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Page size == KV block size; b12x main attention uses page_size 128.
        return [128]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "nvfp4":
            from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim
            head_size = nvfp4_kv_cache_full_dim(head_size)  # hd//2 + hd//16
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError  # no cross-layer KV blocks in M3
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        if cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        raise ValueError(f"Unknown cache layout format {cache_layout}.")


class B12XAttentionImpl(AttentionImpl):
    """Dense full attention via b12x paged decode + extend (no Triton)."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("b12x dense: alibi not supported")
        if sliding_window is not None:
            raise NotImplementedError("b12x dense: sliding window not supported")
        if logits_soft_cap:
            raise NotImplementedError("b12x dense: logits soft cap not supported")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("b12x dense: only DECODER attention")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        # nvfp4: keep the uint8 packed cache (NOT viewed as fp8); b12x reads the
        # per-16 e4m3 block scales from the cache, so no per-tensor descale.
        self.use_nvfp4 = kv_cache_dtype == "nvfp4"
        self._kv_quant = "nvfp4" if self.use_nvfp4 else "none"
        self.use_fp8_kv = is_quantized_kv_cache(kv_cache_dtype) and not self.use_nvfp4
        self.kv_cache_fp8_dtype = (
            torch.float8_e5m2 if "e5m2" in kv_cache_dtype else torch.float8_e4m3fn
        )

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
        # (batch, width, ncb, qdt, kvdt) -> (plan, scratch, cu). The key
        # carries EVERY input that pins the compiled plan/replay state, not
        # just batch: a page-table width / cache-capacity / dtype change after
        # first use must build a NEW plan, never reuse a stale one.
        self._dec: dict[tuple, tuple] = {}
        self._vfy_meta: dict[tuple, dict] = {}  # (batch,q_len,width) -> verify metadata
        self._ext_plan = None              # fixed-capacity extend plan (one compile)
        self._kd: torch.Tensor | None = None
        self._vd: torch.Tensor | None = None

    # ---- fp8 descale buffers (persistent; built once, capture-safe) ----
    def _descales(self, layer, batch: int, device):
        if not self.use_fp8_kv:
            return None, None
        if self._kd is None:
            ks = getattr(layer, "_k_scale", None)
            vs = getattr(layer, "_v_scale", None)
            ks = (
                ks.detach().float().reshape(-1)[0]
                if isinstance(ks, torch.Tensor)
                else torch.ones((), device=device)
            )
            vs = (
                vs.detach().float().reshape(-1)[0]
                if isinstance(vs, torch.Tensor)
                else torch.ones((), device=device)
            )
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
        split = dense_decode_split_kv_enabled()
        key = (int(batch), int(width), int(ncb), qdt, kvdt, split)
        if key not in self._dec:
            if debug_paged_policy_enabled():
                import sys
                print(f"# b12x_decode_ctx site=dense.decode key={key} "
                      f"split_kv={split}",
                      file=sys.stderr, flush=True)
            caps = B12XPagedAttentionScratchCaps(
                device=device, mode="decode", dtype=qdt, kv_dtype=kvdt,
                num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size, head_dim_vo=self.head_size, page_size=128,
                max_total_q=batch, max_batch=batch, max_page_table_width=width,
                max_work_items=batch * 512, max_partial_rows=batch * 512,
                num_cache_pages=ncb, use_cuda_graph=True, msa_block_sparse=False,
                kv_quant=self._kv_quant)
            plan = plan_paged_attention_scratch(caps)
            # max_cache_page_count = worst-case pages a SINGLE request can
            # reference, bounded by the page-table width (cache_seqlen <=
            # max_model_len = width*page_size), NOT the total cache block count
            # ncb. Passing ncb makes the replay-state worst case demand ncb
            # pages from a width-wide page table -> "page_table width is smaller
            # than required by cache_seqlens" once ncb > width.
            plan.prepare_decode_graph_replay_state(
                batch=batch, max_page_table_width=width,
                max_cache_page_count=min(int(ncb), int(width)),
                force_split_kv=split)
            scratch = _shared_storage(
                _DEC_STORE, (batch, width, str(qdt), str(kvdt)), plan, device)
            cu = torch.arange(batch + 1, dtype=torch.int32, device=device)
            self._dec[key] = (plan, scratch, cu)
        return self._dec[key]

    def _decode(self, layer, q, out, k_cache, v_cache, d, ncb):
        device = q.device
        batch = int(d.seq_lens.shape[0])
        plan, scratch, cu = self._decode_ctx(
            batch, int(d.block_table.shape[1]), ncb, q.dtype, k_cache.dtype, device)
        kd, vd = self._descales(layer, batch, device)
        if not hasattr(self, "_attn_meta_owner"):
            self._attn_meta_owner = not _DENSE_META_OWNER_TAKEN[0]
            _DENSE_META_OWNER_TAKEN[0] = True
        binding = plan.bind(
            scratch=scratch, q=q, k_cache=k_cache, v_cache=v_cache, output=out,
            page_table=d.block_table, cache_seqlens=d.seq_lens, cu_seqlens_q=cu,
            q2k_indices=None, k_descale=kd, v_descale=vd,
            skip_replay_metadata_update=(
                _USE_ATTN_META_ONCE and not self._attn_meta_owner))
        paged_attention_forward(binding=binding)

    # ---- spec-decode verify (q_len>1) = MULTI-ROW DECODE. The nd = batch*q_len
    # verify query positions are scored as nd independent captured decode rows:
    # each carries its own causal cache_seqlen (seq_len_i - q_len + j + 1), so
    # the later same-request verify tokens sit beyond cache_seqlen and are
    # masked. Identical result to a fused verify; reuses the supported +
    # cudagraph-captured decode path (_decode_ctx(nd) -> cu=arange(nd+1)). Same
    # mechanism as the MSA/indexer verify paths.
    def _verify_meta(self, d, batch, q_len, nd, width, device):
        key = (int(batch), int(q_len), int(width))
        m = self._vfy_meta.get(key)
        if m is None:
            # Persistent (graph-safe) per-query metadata, rebuilt in place every
            # step by captured ops reading the persistent decode block_table/
            # seq_lens. Tiny: nd*width int32 + nd int32.
            m = self._vfy_meta[key] = dict(
                pt=torch.zeros((nd, width), dtype=torch.int32, device=device),
                seqlens=torch.zeros((nd,), dtype=torch.int32, device=device),
                jr=torch.arange(q_len, device=device, dtype=torch.int32).view(1, q_len))
        m["pt"].view(batch, q_len, width)[:] = \
            d.block_table.to(torch.int32).view(batch, 1, width)
        base = (d.seq_lens.to(torch.int32) - q_len).view(batch, 1)
        m["seqlens"].copy_((base + m["jr"] + 1).reshape(nd))
        return m["pt"], m["seqlens"]

    def _verify(self, layer, q, out, k_cache, v_cache, d, ncb):
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
            q2k_indices=None, k_descale=kd, v_descale=vd)
        paged_attention_forward(binding=binding)

    # ---- prefill extend, eager. FIXED capacity sized to the server maxes so
    # the kernel JIT-compiles ONCE (worklist length / page-table width / total_q
    # capacity are in the b12x compile key); prepare() re-plans actual shapes. ----
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
                num_cache_pages=int(k_cache.shape[0]), msa_block_sparse=False,
                kv_quant=self._kv_quant)
            self._ext_plan = plan_paged_attention_scratch(caps)
        key = ("ext", self._max_q, self._max_seqs, self._max_pt_width,
               str(q.dtype), str(k_cache.dtype), int(k_cache.shape[0]))
        scratch = _shared_storage(_EXT_STORE, key, self._ext_plan, device)
        return self._ext_plan, scratch

    def _extend(self, layer, q, out, k_cache, v_cache, p, ncb):
        device = q.device
        sp, scratch = self._ext_ctx(q, k_cache, device)
        cu = p.cu_seqlens_q.to(torch.int32)
        seqlens = p.seq_lens.to(torch.int32)
        kd, vd = self._descales(layer, int(p.block_table.shape[0]), device)
        binding = sp.bind(
            scratch=scratch, q=q, k_cache=k_cache, v_cache=v_cache, output=out,
            page_table=p.block_table, cache_seqlens=seqlens, cu_seqlens_q=cu,
            q2k_indices=None, k_descale=kd, v_descale=vd)
        paged_attention_forward(binding=binding)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MiniMaxM3SparseMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)  # profiling run; caches unbound
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("b12x dense: fused output quant unsupported")

        md = attn_metadata
        nt = md.num_actual_tokens
        hd = self.head_size

        # KV write (this backend owns the cache update): the CUDA op takes the
        # cache in its native storage + the kv_cache_dtype string and quantizes,
        # DIVIDING by layer._k/v_scale before the fp8 cast (raw unbind, NOT the
        # fp8 view -- matches the sparse layer's _insert_kv contract). b12x then
        # descales by the same scale on read -> exact round trip.
        kc_w, vc_w = kv_cache.unbind(1)
        if self.use_nvfp4:
            # Native nvfp4 packed write (reshape_and_cache_flash is fp8-only);
            # b12x reads the per-16 e4m3 block scales straight from the cache.
            nvfp4_block_quant_write(
                key[:nt], kc_w, md.slot_mapping, self.num_kv_heads, hd)
            nvfp4_block_quant_write(
                value[:nt], vc_w, md.slot_mapping, self.num_kv_heads, hd)
        else:
            ops.reshape_and_cache_flash(
                key[:nt].view(-1, self.num_kv_heads, hd),
                value[:nt].view(-1, self.num_kv_heads, hd),
                kc_w,
                vc_w,
                md.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        # b12x read views: [blocks, 128, kv_heads, hd] (fp8-typed for the kernel).
        kvc = kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache
        k_cache, v_cache = kvc.unbind(1)
        ncb = int(k_cache.shape[0])

        q = query[:nt].view(-1, self.num_heads, hd)
        out = output[:nt].view(-1, self.num_heads, hd)
        nd = md.num_decode_tokens

        if md.num_decodes > 0:
            d = md.decode
            if d.decode_query_len != 1:
                # spec-decode verify batch (q_len = 1 + num_speculative_tokens)
                self._verify(layer, q[:nd], out[:nd], k_cache, v_cache, d, ncb)
            else:
                self._decode(layer, q[:nd], out[:nd], k_cache, v_cache, d, ncb)
        if md.num_prefills > 0:
            p = md.prefill
            self._extend(layer, q[nd:], out[nd:], k_cache, v_cache, p, ncb)
        return output
