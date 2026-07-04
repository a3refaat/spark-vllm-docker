# b12x Decode Baselines — KV/indexer dtype matrix (Phase 0)

Date: 2026-07-01. Raw data: `decode_baselines.tsv`. Bench: `tools/baselines/bench_baseline.py`.

## Setup

- MiniMax-M3-W4A16-GPTQ, dual GB10 (2×Spark) TP2, no-ray, **no speculative decoding**
- Image `vllm-node-minimax-m3-b12x` = `21788c2c74d6`, built from the b12x fork
  branch `minimax-m3` @ `37c8412` (port + fp8-MSA `@cute.jit` fix)
- Common config: max_model_len 65536 (**do not lower**: it pins the indexer
  page-table width at the 1024-page64 scheduled-path threshold), block_size 128,
  max_num_seqs 1, max_num_batched_tokens 2048, FULL_DECODE_ONLY graphs
- Single stream, temperature 0, 256 max new tokens, 2 reps/point, warm prefix
- gpu_memory_utilization per run (capacity-only; no effect on single-stream
  decode): bf16 0.93, fp8 0.925, nvfp4 0.92, nvfp4-idx 0.90

## Decode throughput (t/s, mean of 2 reps)

| main KV / indexer | ctx 100 | ctx 8k | ctx 32k | ctx 60k | Δ 100→60k |
|---|---:|---:|---:|---:|---:|
| bf16 / fp8       | 23.0 | **24.0** | **21.9** | **20.5** | −2.5 |
| fp8 / fp8        | **23.3** | 22.9 | 21.7 | **20.6** | −2.7 |
| nvfp4 / fp8      | 23.4 | 22.5 | 20.1 | 17.9 | −5.5 |
| nvfp4 / nvfp4    | 23.2 | 22.1 | 19.2 | 16.5 | −6.7 |

## Findings

1. **Short-context decode is dtype-insensitive** (~23.2±0.2 t/s everywhere):
   decode is MoE/GEMM-bound at small KV; attention dtype costs only emerge with
   context.
2. **bf16 ≈ fp8 main KV at every context** (bf16 marginally ahead at 8–32k).
   The fp8 descale path buys capacity, not speed, at these contexts.
3. **nvfp4 main KV costs ~1.6 t/s @32k and ~2.7 t/s @60k vs fp8** — the
   nvfp4→bf16 expand paths in main attention scale worse with context
   (matrix item: "native main-attention nvfp4 QK/PV" + dense split-KV).
4. **nvfp4 indexer costs another ~0.9 @32k / ~1.4 t/s @60k vs fp8 indexer** —
   this is the single-row nvfp4 indexer decode fallback (per-head expand +
   full token-logits materialization + torch pooling). Direct validation of
   the "native nvfp4 paged indexer single-row decode" matrix item
   (~0.7–2 ms/token ≈ 0.3–0.8 t/s predicted; measured ≈ 1.4 t/s @60k).
5. **Context scaling is ~2.4× steeper for the nvfp4 configs** (−5.5/−6.7 t/s
   vs −2.5/−2.7 for bf16/fp8 over 100→60k). The optimization plan's decode
   items (one-pass indexer, native nvfp4 scoring, dense split-KV) target
   exactly this gap; full recovery would put nvfp4-nvfp4 at parity with fp8
   while keeping the ~1.78× KV capacity win.

### Phase 2 addendum (2026-07-02, fork `6824dd3`)

Dense decode split-KV enabled in core (opt-in `force_split_kv`; graph-replay
re-chunks per step via the captured LUT update). Speedups vs unsplit @60k:
bf16 3.5×, fp8 5.5×, **nvfp4 14.9×** — split inverts the dtype order (nvfp4
fastest once the expand parallelizes). nvfp4 dense: 8.77 → 0.59 ms/token.
Projected e2e @60k once wired into production (Phase 3): fp8-fp8 ≈ +1.2 t/s,
nvfp4-nvfp4 ≈ +2.6 t/s. Upstream's specialized decode-graph split kernels
turned out page-64-only (wrong partials at page 128) — gated to
`page_tiles_per_entry == 1`; page-128 uses the generic split forward + merge.

## Bugs found & fixed during baselining

- **fp8 MSA decode kernel never compiled** (`DSLRuntimeError: range_constexpr
  should be preprocessed`): `_literal_qk_mma_into_sfrag_plane_fp8_raw` in
  `forward_paged.py` was missing `@cute.jit`. Pre-existing (reproduced on the
  pre-port vendor tree and the old production image); unnoticed because
  production runs nvfp4 and no harness covered fp8-MSA. Fixed in fork commit
  `37c8412`; validated vs bf16 MSA decode (cos 0.9997).
- **bf16 main KV crashed the fused qknorm+rope insert op**
  (`insert mode requires matching index_cache`): the fused op's full-insert
  mode cannot write the b12x packed index cache. Fixed in
  `mods/fix-minimax-m3-b12x-msa/run.sh`: always run the fused op
  norm+rope-only and write both caches via `_insert_kv` (bf16 writes natively
  through `reshape_and_cache_flash`).

## Per-component microbenches (graph-replay, TP2-rank geometry, batch 1)

Bench: `tools/baselines/microbench_matrix.py`; raw: `microbench_matrix.tsv`.
Shapes: 32q/2kv h128 page128 (dense + MSA × gqa16 topk16), indexer 2 heads
page64, page-table capacity pinned at 65536 tokens (serving parity).

**ms/token (µs/layer × layers), single-token decode:**

| component (×layers) | ctx | bf16 | fp8 | nvfp4 |
|---|---:|---:|---:|---:|
| dense-decode ×3 | 8k  | 0.53 | 0.52 | **1.24** |
|                 | 32k | 1.80 | 1.80 | **4.79** |
|                 | 60k | 3.26 | 3.36 | **8.78** |
| dense-split ×3 (Phase 2) | 8k | 0.30 | 0.20 | 0.19 |
|                 | 32k | 0.62 | 0.49 | 0.40 |
|                 | 60k | 0.92 | 0.68 | **0.59** |
| msa-decode ×57  | 8k  | 2.07 | 1.61 | 1.83 |
|                 | 32k | 2.01 | 1.59 | 1.83 |
|                 | 60k | 1.93 | 1.51 | 1.85 |
| indexer-q2k ×57 | 8k  | — | 1.93 | 2.78 |
|                 | 32k | — | 3.54 | 5.53 |
|                 | 60k | — | 5.20 | **8.54** |

### Reconciliation with e2e

- fp8-fp8 @60k: attention-side total ≈ 10.1 ms of the 48.5 ms/token step
  (~21%); nvfp4-nvfp4 ≈ 19.2 ms of 60.6 (~32%). MoE/GEMM dominates the rest.
- Predicted nvfp4-nvfp4 minus fp8-fp8 penalty @60k = +9.1 ms/tok from these
  three components; e2e measured +12.1 ms/tok — components explain ~75%
  (rest: nvfp4 KV writes, per-layer bind/metadata, page-table effects).
- Context growth 8k→60k on fp8: microbench predicts +6.1 ms/tok
  (dense +2.8, indexer +3.3, MSA ≈0); e2e shows +4.9 — directionally right.

### Component findings (updates the optimization matrix)

1. **NEW HEADLINE: nvfp4 dense decode is ~2.6× bf16/fp8** (8.78 vs 3.36
   ms/tok @60k) — the single largest attention-side line item in the nvfp4
   stack, bigger than the indexer gap. The nvfp4→bf16 expand path in the
   3 dense layers alone costs ~+5.4 ms/tok @60k vs fp8. Dense split-KV +
   a native/cheaper nvfp4 dense load path jump in priority together.
2. **Indexer is the largest fp8-path component at long context** (5.20 vs
   dense 3.36 ms/tok @60k) and scales linearly with context — confirms
   one-pass indexer as the top sparse-side item. nvfp4 indexer fallback adds
   +3.3 ms/tok @60k (≈ the ~1.2–1.4 t/s e2e gap between nvfp4-fp8 and
   nvfp4-nvfp4) — confirms native nvfp4 single-row decode item.
3. **MSA attention is small and context-flat** (top-16 blocks = fixed work;
   26–37 µs/layer). fp8 fastest; nvfp4 +0.3 ms/tok, bf16 +0.4 ms/tok
   aggregate. Matches the matrix's low ranking for MSA-kernel work.
4. **Dense decode is far off roofline**: @60k per-rank KV is ~61 MB (bf16)
   → ~0.22 ms at GB10 bandwidth, measured 1.09 ms/layer (~5×); nvfp4 reads
   ~15 MB and takes 2.93 ms (~40×). Under-parallelization (split-KV) plus
   expand cost — both matrix items validated with hard numbers.
5. bf16 ≈ fp8 dense timing (no bandwidth win) — dense decode is
   parallelism-bound, not bandwidth-bound, at these shapes.

## Gaps / follow-ups

- Add an **fp8-MSA decode case to the parity harness** (the gap that let the
  `@cute.jit` bug rot). Ported to `tools/nvfp4-kv/fp8_msa_decode_test.py`.
- A true bf16 *indexer* cache is not a b12x path (packed fp8/nvfp4 only);
  the bf16 rows above use the fp8 packed indexer.
- Microbench indexer numbers use the production fused Triton tails
  (q-quant + select); metadata build (meta-once, amortized) is untimed.
- Batch >1 and verify-shaped (spec-decode) sweeps are not yet covered.

### KVarN Phase 3 addendum (2026-07-03): EAGLE3 on the full KVarN stack

Recipe `recipes/tmp-kvarn-phase3-eagle3.yaml` = Phase-2 stack (kvarn main
sparse tiles + kvarn 4-bit indexer + fused decode) + EAGLE3
(`eagle-bf16-rtn-int4`, spec_tokens=3, draft TP2). The DRAFT inherits the
global kvarn dtype on the stock dense kvarn backend (3456 B/token/rank vs
16384 bf16 for its 32-head full-MHA KV); no `attention_backend` override.

Correctness work: `KVarNSparseGroup.builder_step` flush is now COMMIT-gated
(walk-back from `num_computed_tokens` like the dense kvarn builder) — packing
on scheduled fill would freeze rejected draft tokens into immutable int4
records and orphan the pool rewrites. Gates: `kvarn_spec_test.py` (CLEAN vs
SPEC byte-identical records under rejection; q_len=4 row-exactness for fused
decode + indexer scorer; commit-gated reclaim). Serving: two-turn exact
quote-back across ~11 tile boundaries generated under spec decode.

| metric | phase2 (no spec) | phase3 EAGLE3 |
|---|---:|---:|
| decode @512 | 23.1 t/s | **39.3–41.9 t/s** (+70–81%) |
| decode @16k | 22.3 t/s | **33.2–36.8 t/s** (+49–65%) |
| decode @65k | 21.4 t/s | **28.4–31.0 t/s** (+33–45%) |
| KV pool tokens | 314,496 | 133,504 (2.04x @65k) |
| accept len / rate | — | 2.9–3.5 / 62–83% |

(Ranges = two bench launches; spec-decode throughput varies with per-text
acceptance.) KV capacity recovery (2026-07-04): the drop was NOT the draft
KV (442 kB per 128-token block = 17%) nor the drafter weights (int4 RTN
everything incl. embed/lm_head; dequant-on-lookup embedding keeps it packed
— ~0.9 GiB of the 105.8 total). Breakdown via the native profiling log:
non-torch forward memory jumped ~1 → 3–3.7 GiB with the drafter. Two fixes:
1. `mods/fix-cudagraph-mem-profile-double-charge`: with
   VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 the profiling-time graph
   capture still runs INSIDE the memory-profiling window and its residue
   inflates non_torch_increase (a double charge — the estimate itself is
   never subtracted). Skipping it (VLLM_SKIP_CUDAGRAPH_MEM_PROFILE=1)
   recovered ~0.4 GiB/rank; post-KV-alloc capture needs only 0.06–0.15 GiB
   of the (1-util) margin.
2. util 0.93 → 0.935 (+0.6 GiB; 0.9375 is knife-edge vs startup free — it
   failed once at 114.07 free vs 114.08 needed).
Final: 113,152 → 174,464 tokens (2.66x @65k), +54%. Remaining ~2 GiB of
drafter-induced non-torch is load-time page-cache residue + NCCL asymmetry
on unified memory (TP1 governs, 0.7 GiB above TP0) — charged by the
free-memory-based profiler though partially reclaimable; further recovery
would need NCCL buffer forensics. Post-fix bench: 38.7/39.7/29.6 t/s @
512/16k/65k, acceptance len 3.54 (84.7%).

### KVarN Phase 3b addendum (2026-07-04): EAGLE3 at 131k context

Root cause of the ctx-scaling wall, found via allocator history
(`torch.cuda.memory._record_memory_history` promoted through a tmp mod):
the dense-kvarn backend's FA materialize scratch is floored at
max_model_len tokens per (device,D,Hk) key — 16.4 KB/token for the EAGLE
draft's 32-kv-head shape (2x1.06 GiB @131k, 2x2.13 @196k), charged at
profiling and starving KV. Verify steps never touch it (fused-verify
path); the only long-context user is chunked-prefill continuation.

Fix: `mods/fix-kvarn-dense-fa-scratch-cap` — KVARN_FA_SCRATCH_CAP_TOKENS
env bounds the scratch (32768 in the recipe = 512 MiB draft cost), and
B==1 contexts beyond the cap take a new KV-windowed materialize+FA path:
full windows below the cached boundary run causal=False, the diagonal
tail keeps FA's bottom-right causal mask, partials merge with
flash-decoding LSE math. Offline gate: windowed==single-call cos
0.9999957. Serving gate: 67k-token needle retrieval EXACT (every prefill
chunk past 32k ran the chunked path on the target's dense layers).

Also learned at high cost (both boxes hard-reset): NEVER
num_gpu_blocks_override past profiler-available on GB10 — unified memory
has no clean cudaMalloc failure; overcommit evicts file-backed weights
(instanttensor) and the box thrashes to death. The 0.935+ util startup
check is knife-edge on the head node (APIServer+EngineCore spin up before
the check); 0.93 + honest accounting is the stable regime. /dev/shm psm_*
segments leak from crashed multiproc runs — clear between failed launches.

Config: 131072 max_model_len, 1024 max_num_batched_tokens, util 0.93,
fa-cap 32768. torch peak 3.38 -> 0.72 GiB; KV 170,752 tokens (1.30x @131k).

| ctx | decode t/s | ttft |
|---|---:|---:|
| 512 | 39.7 | 1.3s |
| 65k | 30.4 | 1.9s (warm) |
| 120k | 26.4 | 2.0s (warm) |

Acceptance at 120k: len 3.36, 78.6% — the drafter holds up at depth.
EAGLE3 @262k remains out of reach on 121 GB boxes (needs 4.98 GiB/rank KV
vs ~3.3 honest available); the no-drafter stack fits ~224k if wanted.
