"""Unified decode-policy description for paged attention and indexer plans.

Phase 1 groundwork: one dataclass that DESCRIBES the resolved plan policy --
family, mode, quantization, split/chunk/graph knobs, metadata strategy and
backend variant -- plus a stable ``cache_key()`` for callers that cache plans.

The policy is purely descriptive today: it records what the planner/scratch
layer already decided, so attaching or printing one never changes behavior.
Later phases (dense split-KV, one-pass indexer, native nvfp4 backends) extend
the same fields instead of growing ad-hoc env parsing and per-call-site keys.

Debugging reuses the existing ``B12X_DEBUG_PAGED_POLICY=1`` flag (the planner
decode-graph policy print): when set, every plan build / glue plan-cache miss
also emits one ``# b12x_resolved_policy ...`` line to stderr.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields

VALID_FAMILIES = ("dense", "msa", "indexer")
VALID_MODES = ("decode", "verify", "extend", "prefill")
VALID_KV_QUANTS = ("none", "fp8", "nvfp4")
VALID_METADATA_STRATEGIES = ("runtime-update", "captured-update", "shared-once")
VALID_BACKEND_VARIANTS = ("base", "scheduled", "native_nvfp4", "specialized_m3")


@dataclass(frozen=True)
class PagedDecodePolicy:
    """Resolved policy for one paged attention / indexer plan.

    ``chunk_pages=None`` means the chunk size is resolved at runtime (decode
    chunk-pages LUT or planner heuristic); ``graph_ctas_per_sm=None`` means the
    planner heuristic resolves it. ``head_dim_qk/vo=0`` means not applicable
    (indexer family scores a single shared K vector). ``use_cuda_graph=None``
    means capture is caller-managed (indexer plans carry no graph flag).
    """

    family: str                       # dense | msa | indexer
    mode: str                         # decode | verify | extend | prefill
    kv_quant: str                     # none | fp8 | nvfp4  (KV/index-K storage)
    kv_dtype: str                     # torch storage dtype of the cache
    page_size: int
    head_dim_qk: int
    head_dim_vo: int
    gqa_group_size: int
    max_batch: int
    max_total_q: int
    max_page_table_width: int
    cache_page_capacity: int          # worst-case pages one request references
    split_kv: bool
    chunk_pages: int | None
    max_chunks_per_request: int | None
    graph_ctas_per_sm: int | None
    metadata_strategy: str            # runtime-update | captured-update | shared-once
    backend_variant: str              # base | scheduled | native_nvfp4 | specialized_m3
    use_cuda_graph: bool | None

    def __post_init__(self) -> None:
        if self.family not in VALID_FAMILIES:
            raise ValueError(f"family must be one of {VALID_FAMILIES}, got {self.family!r}")
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.kv_quant not in VALID_KV_QUANTS:
            raise ValueError(f"kv_quant must be one of {VALID_KV_QUANTS}, got {self.kv_quant!r}")
        if self.metadata_strategy not in VALID_METADATA_STRATEGIES:
            raise ValueError(
                f"metadata_strategy must be one of {VALID_METADATA_STRATEGIES}, "
                f"got {self.metadata_strategy!r}"
            )
        if self.backend_variant not in VALID_BACKEND_VARIANTS:
            raise ValueError(
                f"backend_variant must be one of {VALID_BACKEND_VARIANTS}, "
                f"got {self.backend_variant!r}"
            )

    def cache_key(self) -> tuple:
        """Stable, hashable key covering every field that pins a compiled plan."""
        return tuple(getattr(self, f.name) for f in fields(self))

    def describe(self) -> str:
        """One-line ``k=v`` rendering for debug output."""
        return " ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


def normalize_kv_quant(kv_quant: str | None, kv_dtype: object) -> str:
    """Map (planner kv_quant, torch cache dtype) onto the policy vocabulary.

    The planner only distinguishes none/nvfp4 (fp8 KV rides on the cache dtype
    with descale tensors); the policy names fp8 storage explicitly.
    """
    quant = str(kv_quant or "none")
    if quant != "none":
        return quant
    dtype_name = str(kv_dtype)
    if "float8" in dtype_name or dtype_name in ("fp8", "fp8_e4m3fn"):
        return "fp8"
    return "none"


def debug_paged_policy_enabled() -> bool:
    """Single debug gate shared with the planner's decode-graph policy print."""
    return os.environ.get("B12X_DEBUG_PAGED_POLICY") == "1"


def dense_decode_split_kv_enabled() -> bool:
    """Rollout knob for dense (non-block-sparse) decode split-KV.

    ``B12X_DENSE_DECODE_SPLIT_KV=0`` restores unsplit dense decode plans
    (instant rollback); default is ON. This is a PLAN-level switch: within a
    captured split plan the chunk count already adapts to runtime
    cache_seqlens via the chunk-pages LUT (short contexts collapse to a
    single chunk), so no separate min-context knob is needed. Chunk-size
    sweeps use the existing ``B12X_PAGED_DECODE_GRAPH_CHUNK_PAGES`` override.
    """
    return os.environ.get("B12X_DENSE_DECODE_SPLIT_KV", "1") != "0"


def attn_meta_once_enabled() -> bool:
    """Rollout knob for decode replay-metadata sharing across layers (Phase 4).

    Layers of one attention family (dense / MSA) bind the SAME shared scratch
    storage with identical plans and identical runtime page_table/seqlens, so
    their per-step replay metadata writes are byte-identical. With
    ``B12X_ATTN_META_ONCE=1`` (default) only the family's first layer performs
    the ~13-op metadata copy + runtime chunk update each decode step;
    followers only re-bind tensor refs. ``=0`` restores per-layer updates.
    """
    return os.environ.get("B12X_ATTN_META_ONCE", "1") != "0"


def msa_fused_indexer_enabled() -> bool:
    """Rollout knob for the one-pass fused MSA indexer (Phase 5).

    ``B12X_MSA_FUSED_INDEXER=0`` restores the scheduled scorer + select-tail
    chain (instant rollback); default is ON. Applies to the paged MSA q2k
    route (decode and multi-row verify rows share the kernel).
    """
    return os.environ.get("B12X_MSA_FUSED_INDEXER", "1") != "0"


def emit_resolved_policy(policy: PagedDecodePolicy, *, site: str) -> None:
    if debug_paged_policy_enabled():
        print(
            f"# b12x_resolved_policy site={site} {policy.describe()}",
            file=sys.stderr,
            flush=True,
        )
