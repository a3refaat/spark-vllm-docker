# KVarN × MiniMax-M3 (b12x) integration plan

Date: 2026-07-03. Status: Phase 0 PASSED, Phase 1 SERVING (2026-07-03).

PIVOT (user decision): the kvarn config drops b12x entirely — stock Triton
sparse impl + stock indexer + KVarN dense/draft. b12x existed for KV-dtype
support on GB10 (stock Triton historically hit ~26 t/s with fp8 KV; the gap
was nvfp4 dense/indexer/MTP support, which KVarN now covers). Phase 2 targets
the STOCK MiniMaxM3SparseTritonImpl (upstreamable), not b12x glue; the b12x
fp8/nvfp4 recipes remain separate production configs.

Phase 0 results (GB10/SM121): Triton JIT ok; pack of 114 tiles (full-model
one-block flush) = 1.25 ms; round-trip cos on iid-gaussian worst case K(4b)
0.995 / V(2b) 0.894; flash_attn_varlen_func works incl. 2-KV-head shapes.
PR #46812 backported as mods/add-kvarn-kv-quant (16 files verbatim via patch,
3 drifted hunks as anchor edits).

Phase 1 results: recipes/tmp-kvarn-phase1.yaml healthy — kvarn dense +
bf16 sparse via mods/fix-minimax-m3-kvarn-hybrid (single-anchor dtype
degrade), stock Triton indexer (bf16 side cache; fp8 indexer flag NOT
supported by stock impl). Coherent multi-turn (greedy-matches fp8 ground
truth); 24.3 tok/s short-context non-spec. KV pool 66,432 tokens @1.01x
(bf16 sparse dominates — Phase 2 is the capacity payoff). LESSON: KVarN's
fp16 pool auto-budget (0.5x post-weight usable) starved the KV cache on M3
(only 3 kvarn layers, max_num_seqs=1) → KVARN_POOL_MEM_FRAC=0.01 in the
recipe. Remaining Phase 1b: eagle3 spec (draft on kvarn) + 60k bench.

## What KVarN is

Variance-normalized KV-cache quantization ([arXiv:2606.03458](https://arxiv.org/abs/2606.03458),
Huawei CSL): per 128-token tile, per KV head — Hadamard rotation along head_dim
→ log-domain Sinkhorn variance normalization → asymmetric RTN at 4-bit K /
2-bit V, scales absorbed into per-channel (K) / per-token (V) fp16 vectors.
Calibration-free. An in-progress tile can't be quantized (tile-shared scales),
so a small fp16 **tail pool** holds each request's partial tail block plus a
permanent fp16 **sink block** (first 128 tokens). Kernels are Triton (JIT — no
CUDA build). Claimed: fp16-parity accuracy, ≥fp16 throughput, 3–5× KV capacity.

Sources, in order of relevance to us:
- **PR [vllm#46812](https://github.com/vllm-project/vllm/pull/46812)** — dense/GQA
  backend on **current vLLM main**. Verified: applies with ZERO code conflicts
  onto `origin/main` (e196268ba); ~4,800 new lines in self-contained files +
  ~290 lines across 10 integration points. **All Python** — deployable as a
  launch-time mod or via `build-and-copy.sh --apply-vllm-pr 46812`.
  Snapshot: `/tmp/kvarn_pr46812.diff`, worktree `/tmp/vllm-kvarn-pr`.
- **Fork `~/Software/Projects/ML-Workbench/KVarN`** (vLLM 0.23 base) — extra
  material not in the PR: the **sparse-MLA (DSA) port**
  (`vllm/v1/attention/backends/mla/flashmla_sparse_kvarn.py`) and the MLA tile
  spec (`KVARN_MLA_BACKEND_SPEC.md`) — our design templates for MSA.

## Why for us (measured constraints)

- Current fp8-KV pool: **74,880 tokens** total (65k recipe, util 0.93,
  concurrency 1.14×); the user's 131k fp8-eagle config barely fits. KV bytes
  are the binding constraint on max_model_len / concurrency on GB10.
- M3 sparse layer (2 KV heads × 128, g128) per-token-per-layer: bf16 1024 B,
  fp8 512 B, **KVarN k4v2 216 B** (13,824 B/tile/head incl. scales). ≈ **2.37×
  tokens vs fp8**, 4.7× vs bf16, before pool overhead. 131k stops being tight;
  ~196k+ becomes plausible; concurrency >1 at long context becomes real.
- Quality: k4v2 + rotation + sinkhorn claims fp16-parity (AIME-verified
  upstream on Qwen3); candidate replacement for both fp8 (capacity) and nvfp4
  (accuracy: e2m1 grid is ±20% mid-range, uncalibratable).
- Composes with W4A16 weights + spec decode (upstream validated AWQ-int4 + MTP;
  verify commits a tile only when its tokens are accepted).

## Architectural fit (the load-bearing observations)

1. **M3 MSA topk granularity == KVarN tile granularity.** The indexer selects
   128-token KV blocks (WIDTH128 tables); KVarN quantizes per 128-token tile
   (block_size 128 — already our page size). MSA sparse read = gather ≤topk
   whole tiles. No sub-tile access ever needed on the sparse path.
2. **The DSA port is a 1:1 template** (`flashmla_sparse_kvarn.py`): store =
   rotate + Triton scatter into fp16 pool (graph-safe, pure tensor ops); flush
   of full tiles in the **metadata builder between graph replays**; attend =
   topk → gather+dequant selected slots into a **rotated** bf16 workspace →
   remap indices → **unchanged bf16 sparse kernel**. Rotation handled q-side:
   `q_rot = q @ H` before the kernel, output un-rotated after (`o @ H`; H
   symmetric orthonormal) — ~8× cheaper than un-rotating KV. The b12x CuTe MSA
   kernel can run **oblivious** on the rotated workspace.
3. **The indexer stack is untouched.** Index-K lives in its own packed cache;
   topk selection is upstream of attention and independent of KV format. All
   b12x indexer work (fused MSA indexer, verify-grouped) carries over as-is.
4. **Graph discipline matches ours.** Their split — capturable pure-tensor
   store/attend, eager Python state mutation (slot alloc, fill tracking,
   sinkhorn flush) in `build()` — mirrors b12x meta-once. `UNIFORM_BATCH` CG
   support + spec-as-decode verify plan is compatible with our
   `mode:0 + FULL_DECODE_ONLY` production config (their fused-verify comment
   describes exactly the eager-attention-between-graph-segments cost we fixed).
5. **Hardened where we'd bleed:** fill tracking keyed by PHYSICAL block id
   (prefix-cache sharing / block-id recycling — "repetition-collapse / stale
   tile class"); retired-sink residency for multi-turn prefix hits; pool sized
   from the post-weights envelope with a concurrency cap; bf16 boundary casts
   (we serve bf16); head_size 128 supported.

## Deployment shape (answers "married to b12x?")

Per layer group, in the target end state:

| Layers | Attention kernel | KV format |
|---|---|---|
| 57 sparse MSA | **b12x CuTe MSA (kept — it's the perf)** | KVarN tiles + fp16 pool |
| 3 dense | stock `KVarNAttentionBackend` (b12x dense retired here) | KVarN |
| EAGLE3 draft | stock `KVarNAttentionBackend` | KVarN |
| index-K side cache | b12x fused indexer (unchanged) | packed fp8 (unchanged) |

b12x stays where it earns its keep (MSA kernels + indexer); the dense/draft
layers ride the upstream backend for free. If Phase 2 perf surprises us, the
fallback for MSA-decode is their fused dequant-attend Triton kernel driven by a
synthetic per-request block table of the topk tiles (viable because tiles are
whole blocks) — at the cost of leaving CuTe.

## Phases

### Phase 0 — feasibility gates (no serving changes)
- Apply `/tmp/kvarn_pr46812.diff` into the image's site-packages in an
  ephemeral container (`patch -p1`; image base g979b56a66 is ~2 weeks older
  than the PR base — expect at most trivial fuzz; if hunks fail, backport
  notes here). All-Python ⇒ testable without any wheel rebuild.
- GB10/SM121 Triton smoke: sinkhorn pack→unpack round trip (cos ≥ 0.999),
  fused decode kernel microbench vs fp8 b12x dense decode at M3-like shapes
  (2 KV heads is an unusually skinny GQA — check their kernel's occupancy).
- Decision: ship KVarN core as launch-time mod (fast iteration) vs
  `--apply-vllm-pr 46812` wheel rebuild (permanence). Both remain open.

### Phase 1 — dense + draft on KVarN, sparse stays fp8 (zero b12x changes)
- `kv_cache_dtype: kvarn_k4v2_g128`, block_size 128 (already ours).
- Dense + draft layers route to `KVarNAttentionBackend` automatically (standard
  `Attention` modules). Sparse layers: our mods own their cache spec — map
  `kvarn_*` → keep the fp8 spec + b12x MSA (hybrid allocator already handles
  heterogeneous groups incl. the index cache).
- Gates: coherence probe, eagle acceptance unchanged, bench @60k (expect ≈
  neutral; dense+draft KV is a minor slice), KV-pool tokens report.
- Value: proves KVarN e2e on GB10 inside our serving stack before any b12x work.

### Phase 2 — b12x MSA on KVarN tiles (the capacity win)
- **2a store:** sparse spec → uint8 records `[blocks, 2, 13824→aligned]`;
  `_insert_kv` sparse branch → reuse `_kvarn_scatter_store_kernel` + rotate
  (pool slots via block→slot indirection tensor). Flush + fill tracking +
  sink/retired-sink logic hosted in the b12x meta hook, lifted from
  `KVarNMetadataBuilder.build()` (keep their physical-block keying VERBATIM).
- **2b decode/verify read:** adapt the DSA gather kernel: q2k-selected tiles
  (≤16/req decode; row-union ≤64 verify) + pool blocks → rotated bf16
  workspace pages; synthetic page table → **unchanged** b12x MSA kernel;
  `q @ H` pre, `o @ H` post. Budget: ~4.5 MB/layer/step traffic vs ~1 MB
  direct fp8 — attention is a minor slice of M3 decode (MoE-bound), expect
  low-single-digit t/s cost at 60k; measure in verify_microbench first
  (offline harness extension, kernels testable without the server).
- **2c prefill/extend:** chunked context gather-dequant into a bounded
  workspace (their `_gather_request_kv` / MLA workspace pattern) ahead of
  b12x `_extend`.
- Gates (each before next): offline tile round-trip at M3 geometry → MSA
  parity vs bf16 reference (offline plans) → e2e coherence + long-context
  quality probe → bench matrix.
- Lesson applied from the fp8-descale bug: every per-row buffer sized by
  `max_seqs × (1+spec_tokens)`; loud guards, no silent short-slices.

### Phase 3 — evaluation matrix & tuning
- t/s @60k/@131k × {fp8, nvfp4, kvarn} × {eagle on/off}; KV tokens/pool
  report; acceptance; quality (long-context probes + a reasoning eval) —
  kvarn's claim is fp8-beating quality at 2.4× capacity: verify both axes.
- Tune: workspace sizing, batched-flush cadence, Triton configs on SM121,
  `KVARN_POOL_MEM_FRAC` on unified memory.

## Risks / open questions
- 2-KV-head GQA occupancy in their Triton decode kernel (designed for 4–16
  heads); mitigated by Phase 0 microbench + the CuTe-workspace design keeping
  their kernels off the sparse hot path.
- Image-ref drift vs PR base (Phase 0 gate); PR still in review — pin the
  diff snapshot, refresh deliberately.
- fp16 pool + workspace on unified memory: max_num_seqs=1 keeps the pool
  tiny, but verify the pool-vs-KV split logs on GB10.
- Draft on KVarN under mode:0 + FULL_DECODE_ONLY: their UNIFORM_BATCH path is
  graph-compatible on paper; validate acceptance + graphs in Phase 1.
- M3 MSA per-row causal semantics for the tail block must be preserved in the
  workspace assembly (pool tail tokens gathered with per-row seqlens exactly
  as b12x builds them today).
