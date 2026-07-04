"""Serving-config Triton prewarm for the MiniMax-M3 b12x stack.

The stock vLLM warmup does not exercise several MiniMax/b12x/EAGLE Triton
helpers with production metadata shapes.  If they first execute after the JIT
monitor is enabled, vLLM reports "Triton kernel JIT compilation during
inference" and the first real request can stall for minutes.

This module is intentionally small and side-effect free until called from the
GPU worker warmup path.  It launches the relevant kernels once with dummy tensors
whose constexprs match the serving config, thereby populating Triton's persistent
cache before CUDA graph capture / inference.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ServingShape:
    max_num_batched_tokens: int
    max_num_seqs: int
    block_size: int
    max_model_len: int
    page_table_width: int
    kv_cache_dtype: str
    indexer_kv_dtype: str
    spec_tokens: int

    @property
    def verify_q_len(self) -> int:
        return max(int(self.spec_tokens) + 1, 1)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _next_pow2(x: int) -> int:
    x = max(int(x), 1)
    return 1 << (x - 1).bit_length()


def _getattr_chain(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    cur = obj
    for name in names:
        cur = getattr(cur, name, None)
        if cur is None:
            return default
    return cur


def _resolve_serving_shape(worker: Any) -> _ServingShape:
    cfg = getattr(worker, "vllm_config", None)
    scheduler = getattr(cfg, "scheduler_config", None)
    cache = getattr(cfg, "cache_config", None)
    model = getattr(cfg, "model_config", None)
    speculative = getattr(cfg, "speculative_config", None)

    max_tokens = _safe_int(
        getattr(scheduler, "max_num_batched_tokens", None), 2048
    )
    max_seqs = _safe_int(getattr(scheduler, "max_num_seqs", None), 1)
    block_size = _safe_int(getattr(cache, "block_size", None), 128)
    max_model_len = _safe_int(getattr(model, "max_model_len", None), 65536)
    page_table_width = max((max_model_len + block_size - 1) // block_size, 1)

    kv_cache_dtype = str(
        getattr(cache, "cache_dtype", None)
        or getattr(cache, "kv_cache_dtype", None)
        or "auto"
    )
    # vLLM stores attention-config as a dict on model_config in recent builds.
    attn_cfg = getattr(model, "attention_config", None) or {}
    if not isinstance(attn_cfg, dict):
        attn_cfg = {}
    indexer_kv_dtype = str(attn_cfg.get("indexer_kv_dtype", "fp8"))
    spec_tokens = _safe_int(
        getattr(speculative, "num_speculative_tokens", None), 0
    )
    return _ServingShape(
        max_num_batched_tokens=max(max_tokens, 1),
        max_num_seqs=max(max_seqs, 1),
        block_size=max(block_size, 1),
        max_model_len=max(max_model_len, 1),
        page_table_width=page_table_width,
        kv_cache_dtype=kv_cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype,
        spec_tokens=max(spec_tokens, 0),
    )


def _resolve_device(worker: Any) -> torch.device:
    runner = getattr(worker, "model_runner", None)
    device = getattr(runner, "device", None)
    if device is not None:
        return torch.device(device)
    return torch.device("cuda", torch.cuda.current_device())


def _prewarm_slot_mapping(device: torch.device, shape: _ServingShape) -> None:
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.block_table import _compute_slot_mapping_kernel

    max_tokens = int(shape.max_num_batched_tokens)
    max_reqs = int(shape.max_num_seqs)
    width = int(shape.page_table_width)
    qsl = torch.zeros((max_reqs + 1,), dtype=torch.int32, device=device)
    # Use one token per request and let the kernel pad the rest.  num_tokens and
    # max_num_tokens are do_not_specialize in vLLM; width/block size are the
    # serving-config-sensitive parts.
    qsl[1:] = torch.arange(1, max_reqs + 1, dtype=torch.int32, device=device)
    positions = torch.arange(max_tokens, dtype=torch.int64, device=device)
    block_table = torch.zeros((max_reqs, width), dtype=torch.int32, device=device)
    slot_mapping = torch.empty((max_tokens,), dtype=torch.int64, device=device)

    # The MiniMax recipes do not use context parallelism; keep the fallback
    # conservative rather than importing distributed groups during warmup.
    total_cp_world_size = 1
    total_cp_rank = 0
    cp_kv_cache_interleave_size = 1
    _compute_slot_mapping_kernel[(max_reqs + 1,)](
        max_reqs,
        max_tokens,
        qsl,
        positions,
        block_table,
        block_table.stride(0),
        int(shape.block_size),
        slot_mapping,
        TOTAL_CP_WORLD_SIZE=total_cp_world_size,
        TOTAL_CP_RANK=total_cp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE=1024,
    )


def _prewarm_nvfp4_main_kv_write(device: torch.device, shape: _ServingShape) -> None:
    if "nvfp4" not in shape.kv_cache_dtype:
        return
    from b12x.vllm.minimax_m3.backend import nvfp4_block_quant_write

    # Target dense layers have 4 KV heads.  EAGLE3's one-layer draft rides the
    # same b12x dense backend with 64 KV heads.  Warm both specializations; the
    # function itself is cheap when Triton cache is already populated.
    for num_kv_heads in (4, 64):
        head_dim = 128
        packed_dim = head_dim // 2 + head_dim // 16
        src = torch.zeros((1, num_kv_heads, head_dim), dtype=torch.bfloat16, device=device)
        cache = torch.zeros(
            (1, int(shape.block_size), num_kv_heads, packed_dim),
            dtype=torch.uint8,
            device=device,
        )
        slot = torch.zeros((1,), dtype=torch.int64, device=device)
        nvfp4_block_quant_write(src, cache, slot, num_kv_heads, head_dim)


def _prewarm_indexer_writes(device: torch.device, shape: _ServingShape) -> None:
    from b12x.vllm.minimax_m3.indexer import (
        write_packed_index_cache,
        write_packed_index_cache_nvfp4,
    )

    index_key = torch.zeros((1, 128), dtype=torch.bfloat16, device=device)
    slot = torch.zeros((1,), dtype=torch.int64, device=device)
    # fp8 page-major writer is used by the fp8-index recipes and is cheap to warm
    # unconditionally.
    cache_fp8 = torch.zeros((2, 8448), dtype=torch.uint8, device=device)
    write_packed_index_cache(cache_fp8, index_key, slot)
    if shape.indexer_kv_dtype == "nvfp4":
        cache_nv = torch.zeros((2, 4608), dtype=torch.uint8, device=device)
        write_packed_index_cache_nvfp4(cache_nv, index_key, slot)


def _prewarm_msa_prefill_union_metadata(device: torch.device, shape: _ServingShape) -> None:
    from b12x.attention.paged.graph_replay import build_msa_prefill_union_metadata

    kv_heads = 4
    topk = 16
    max_q = int(shape.max_num_batched_tokens)
    gqa = 16
    work_capacity = max((max_q * gqa + 15) // 16, 1)
    q2k = torch.full((kv_heads, max_q, topk), -1, dtype=torch.int32, device=device)
    cache_seqlens = torch.full(
        (shape.max_num_seqs,), shape.max_model_len, dtype=torch.int32, device=device
    )
    cu = torch.zeros((shape.max_num_seqs + 1,), dtype=torch.int32, device=device)
    cu[1:] = max_q
    request_indices = torch.zeros((work_capacity,), dtype=torch.int32, device=device)
    qo_tile_indices = torch.zeros_like(request_indices)
    block_valid_mask = torch.zeros((work_capacity,), dtype=torch.int32, device=device)
    union_blocks = torch.zeros((work_capacity, kv_heads, 128), dtype=torch.int32, device=device)
    union_masks = torch.zeros_like(union_blocks)
    union_counts = torch.zeros((work_capacity, kv_heads), dtype=torch.int32, device=device)
    build_msa_prefill_union_metadata(
        q2k_indices=q2k,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu,
        request_indices=request_indices,
        qo_tile_indices=qo_tile_indices,
        block_valid_mask=block_valid_mask,
        union_blocks=union_blocks,
        union_masks=union_masks,
        union_counts=union_counts,
    )


def _prewarm_msa_verify_metadata(device: torch.device, shape: _ServingShape) -> None:
    if shape.spec_tokens <= 0:
        return
    from b12x.attention.paged.graph_replay import update_msa_verify_graph_chunk_metadata

    batch = int(shape.max_num_seqs)
    q_len = int(shape.verify_q_len)
    total_q = batch * q_len
    # Match MiniMaxM3SparseB12xImpl._verify_ctx capacities.
    work_capacity = max(total_q * 64, 1)
    cache_seqlens = torch.full((batch,), shape.max_model_len, dtype=torch.int32, device=device)
    cu = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * q_len
    request_indices = torch.zeros((work_capacity,), dtype=torch.int32, device=device)
    qo_tile_indices = torch.zeros_like(request_indices)
    kv_tile_indices = torch.zeros_like(request_indices)
    merge_indptr = torch.zeros((total_q + 1,), dtype=torch.int32, device=device)
    o_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device=device)
    block_valid_mask = torch.zeros((work_capacity,), dtype=torch.int32, device=device)
    kv_chunk_size_ptr = torch.zeros((1,), dtype=torch.int32, device=device)
    kv_window_start_tokens = torch.zeros((batch,), dtype=torch.int32, device=device)
    total_num_rows_ptr = torch.zeros((1,), dtype=torch.int32, device=device)
    update_msa_verify_graph_chunk_metadata(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu,
        request_indices=request_indices,
        qo_tile_indices=qo_tile_indices,
        kv_tile_indices=kv_tile_indices,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        block_valid_mask=block_valid_mask,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        kv_window_start_tokens=kv_window_start_tokens,
        total_num_rows_ptr=total_num_rows_ptr,
        kv_chunk_size=128,
        page_size=128,
        max_q_rows_per_req=q_len,
    )


def _prewarm_eagle_kernels(device: torch.device, shape: _ServingShape) -> None:
    if shape.spec_tokens <= 0:
        return
    from vllm.v1.spec_decode.llm_base_proposer import (
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
    )
    from vllm.v1.spec_decode.utils import eagle_step_slot_mapping_metadata_kernel

    batch = int(shape.max_num_seqs)
    sampled_cols = int(shape.spec_tokens) + 1
    block_tokens = _next_pow2(sampled_cols)
    sampled = torch.full((batch, sampled_cols), -1, dtype=torch.int32, device=device)
    discard = torch.zeros((batch,), dtype=torch.bool, device=device)
    backup = torch.zeros((batch,), dtype=torch.int32, device=device)
    next_tokens = torch.empty((batch,), dtype=torch.int32, device=device)
    valid_count = torch.empty((batch,), dtype=torch.int32, device=device)
    eagle_prepare_next_token_padded_kernel[(batch,)](
        sampled,
        discard,
        backup,
        next_tokens,
        valid_count,
        32000,
        sampled_cols,
        batch,
        sampled.stride(0),
        BLOCK_SIZE_TOKENS=block_tokens,
    )

    cu_draft = torch.full((batch,), int(shape.spec_tokens), dtype=torch.int32, device=device)
    qsl = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * sampled_cols
    token_indices = torch.empty((batch,), dtype=torch.int32, device=device)
    rejected = torch.empty((batch,), dtype=torch.int32, device=device)
    eagle_prepare_inputs_padded_kernel[(batch,)](
        cu_draft,
        valid_count,
        qsl,
        token_indices,
        rejected,
        batch,
    )

    width = int(shape.page_table_width)
    positions = torch.zeros((batch,), dtype=torch.int64, device=device)
    block_table = torch.zeros((batch, width), dtype=torch.int32, device=device)
    seq_lens = torch.ones((batch,), dtype=torch.int32, device=device)
    out_pos = torch.empty_like(positions)
    out_slot = torch.empty((batch,), dtype=torch.int64, device=device)
    eagle_step_slot_mapping_metadata_kernel[(batch,)](
        positions,
        block_table,
        block_table.stride(0),
        seq_lens,
        out_pos,
        out_slot,
        block_size=int(shape.block_size),
        max_model_len=int(shape.max_model_len),
        n_blocks_per_req=width,
        PAD_ID=-1,
        batch_size=batch,
    )


def prewarm_triton_kernels_for_worker(worker: Any) -> None:
    """Precompile MiniMax-M3 b12x Triton kernels for this GPU worker.

    Called from the vLLM worker warmup path before CUDA graph capture and before
    jit_monitor activation.  Set ``B12X_TRITON_PREWARM=0`` to disable.  Set
    ``B12X_TRITON_PREWARM_STRICT=1`` to make failures fatal instead of falling
    back to lazy JIT.
    """
    if os.environ.get("B12X_TRITON_PREWARM", "1") == "0":
        logger.info("b12x Triton prewarm disabled by B12X_TRITON_PREWARM=0")
        return
    if not torch.cuda.is_available():
        return

    shape = _resolve_serving_shape(worker)
    device = _resolve_device(worker)
    logger.info("b12x Triton prewarm starting: %s", shape)
    try:
        _prewarm_slot_mapping(device, shape)
        _prewarm_nvfp4_main_kv_write(device, shape)
        _prewarm_indexer_writes(device, shape)
        _prewarm_msa_prefill_union_metadata(device, shape)
        _prewarm_msa_verify_metadata(device, shape)
        _prewarm_eagle_kernels(device, shape)
        torch.cuda.synchronize(device)
    except Exception:
        logger.exception("b12x Triton prewarm failed")
        if os.environ.get("B12X_TRITON_PREWARM_STRICT", "0") == "1":
            raise
    else:
        logger.info("b12x Triton prewarm complete")


__all__ = ["prewarm_triton_kernels_for_worker"]
