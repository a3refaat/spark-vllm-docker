# MiniMax-M3-W4A16-GPTQ on 2x DGX Spark (GB10) — Deployment Package

Serve [`Sebesky/MiniMax-M3-W4A16-GPTQ`](https://huggingface.co/Sebesky/MiniMax-M3-W4A16-GPTQ)
(Marlin W4A16 MoE) across two GB10 nodes at TP2, with EAGLE3 speculative
decoding and a choice of three quantized-KV attention stacks. One container
image backs all recipes; the per-recipe vLLM integration mods are applied
into the fresh container at launch, so mutually-exclusive stacks (b12x vs
KVarN) never collide.

## What's in the box

| recipe | KV cache | attention stack | ctx | headline |
|---|---|---|---|---|
| `minimax-m3-w4a16-gptq-b12x-fp8-eagle3` | fp8 | b12x (CuTe-DSL/Triton, SM121) | 131k | ~34 t/s @60k, accept 2.96 tok/step |
| `minimax-m3-w4a16-gptq-b12x-nvfp4-eagle3` | nvfp4 | b12x, draft KV also nvfp4 | 196k | densest b12x KV (~72 B/head) |
| `minimax-m3-w4a16-gptq-kvarn-eagle3` | KVarN k4v2 (4-bit K / 2-bit V) | stock vLLM Triton + PR #46812 backport | 131k | 39.7/30.4/26.4 t/s @512/65k/120k, 170,752-token KV pool, accept 78.6% @120k |

All three: reasoning parser + tool calling (`minimax_m3`), thinking mode on,
`FULL_DECODE_ONLY` cuda graphs, EAGLE3 draft at TP2 with `spec_tokens: 3`.
Benchmark narratives: `runs/baselines/SUMMARY.md`, `runs/eagle3-20260702/REPORT.md`.

Which one?
- **kvarn-eagle3** — best measured decode speed and KV density; everything is
  stock Triton + the KVarN backport (no CuTe DSL in the hot path).
- **b12x-fp8-eagle3** — the conservative quantization (fp8 KV everywhere).
- **b12x-nvfp4-eagle3** — maximum context per byte on the b12x stack.

## Prerequisites

- 2x NVIDIA DGX Spark (GB10, 121 GiB unified memory each), ConnectX link
  between them (link-local addressing works; see `autodiscover.sh` to
  generate `.env` with `HEAD_IP` / `PEER_NODES` / `COPY_HOSTS`).
- Docker with the NVIDIA container runtime on both nodes; passwordless SSH
  from the head node to the worker.
- `HF_TOKEN` exported in your shell (recipes forward it into the container).
- ~250 GB free NVMe per node: model (~110 GB) + image + build cache.

## 1. Build the image (head node)

```bash
./build-deploy.sh -c <worker-ip>
```

This builds `vllm-node-minimax-m3` (vLLM pinned at `979b56a66c96` from
source, with the `minimax-m3-fused-fp8-kv` source patch and the Rust tool
parser — 1-2 h cold, ccache-fast after), layers the vendored b12x kernel
library on top as `vllm-node-minimax-m3-b12x` (`Dockerfile.deploy`), and
distributes the result to the worker via save|load. The b12x tree is
vendored at the exact tested commit (`vendor/b12x/.vendored-from-commit`);
it is pure Python/JIT — no wheel builds, no dependency changes.

## 2. Download the model (both nodes)

```bash
./hf-download.sh Sebesky/MiniMax-M3-W4A16-GPTQ -c <worker-ip>
```

### EAGLE3 drafter

The recipes pull the int4-RTN EAGLE3 drafter (single Llama-style decoder
block, ~1.6 GB, embed_tokens/lm_head quantized too) from
[`Sebesky/MiniMax-M3-EAGLE3-RTN-INT4`](https://huggingface.co/Sebesky/MiniMax-M3-EAGLE3-RTN-INT4)
via the recipes' `drafter_model:` default. Prefetching to both nodes is
**required** — the serving containers run with `HF_HUB_OFFLINE=1` (weights
must come from the mounted host cache; no downloads at load time):

```bash
./hf-download.sh Sebesky/MiniMax-M3-EAGLE3-RTN-INT4 -c <worker-ip>
```

Do not substitute a bf16 or "slim" (shared embed/lm_head) drafter — the
int4-everything variant with its own embed/lm_head is the validated one;
the loader mod (`fix-eagle3-draft-embed-quant`) depends on the draft quant
config.

## 3. Serve

```bash
# recommended first run
./run-recipe.sh --no-ray -d recipes/minimax-m3-w4a16-gptq-kvarn-eagle3.yaml

# or the b12x variants
./run-recipe.sh --no-ray -d recipes/minimax-m3-w4a16-gptq-b12x-fp8-eagle3.yaml
./run-recipe.sh --no-ray -d recipes/minimax-m3-w4a16-gptq-b12x-nvfp4-eagle3.yaml
```

`run-recipe.sh` builds the image if missing, starts both containers, applies
the recipe's mod stack inside them, and launches `vllm serve` across the
nodes with `--distributed-executor-backend mp`. Wait for
`http://<head>:8000/health` → 200 (model load + Triton warmup takes several
minutes), then:

```bash
curl http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Sebesky/MiniMax-M3-W4A16-GPTQ",
  "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64}'
```

Switching recipes: stop the cluster first (`./launch-cluster.sh stop`) —
mods are applied to a fresh container per launch and must not stack.

## GB10 operational notes (unified memory — read before tuning)

Hard-won on this exact setup; ignoring these can wedge a node hard enough to
need a power cycle:

1. **Never set `num_gpu_blocks_override` above what the profiler reports.**
   GB10 unified memory has no clean `cudaMalloc` failure: overcommit silently
   evicts the file-backed model weights and the box thrashes itself to death
   on NVMe. The profiler's accounting is authoritative here.
2. **Keep `gpu_memory_utilization` at 0.93.** The startup free-memory check
   runs after the API server and engine processes spin up on the head node;
   0.935+ passes or fails depending on boot-to-boot page-cache state. A
   failure is a clean `ValueError` (safe to retry), but 0.93 just works.
3. **Drop page caches on BOTH nodes before a launch**, never during/after
   model load (`instanttensor` weights are file-backed and must stay cached):
   ```bash
   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches   # on each node
   ```
4. **After a crashed launch**, clean up before retrying — leaked shared
   memory and dead containers eat multiple GiB of the startup budget:
   ```bash
   docker rm -f vllm_node 2>/dev/null; sudo find /dev/shm -name 'psm_*' -delete
   ```
5. KV capacity varies ±0.5 GiB run-to-run with boot/page-cache state; don't
   chase small deltas across reboots.

## Tuning levers

`run-recipe.sh` accepts `--max-model-len`, `--gpu-mem`, `--tp`, `--port`
overrides directly; everything else is edited under the recipe's `defaults:`.

- `max_model_len` — KVarN\@131k leaves ~1.3x concurrency headroom; the KVarN
  stack without the drafter profiles to ~224k if you trade EAGLE3 away
  (`runs/baselines/SUMMARY.md` has the accounting).
- `max_num_batched_tokens` (1024) — raising to 2048 speeds long-prompt
  prefill but costs ~0.75 GiB of KV via the profiling peak.
- `spec_tokens` (3) — validated sweet spot; 4 was model-rejected.

## Layout of the package

```
Dockerfile.deploy        image: base + vendored b12x (+ build-time smoke tests)
build-deploy.sh          one-command build + distribution
vendor/b12x/             b12x kernel library @ tested commit
mods/                    launch-time vLLM patches (self-contained diffs/scripts)
  add-kvarn-kv-quant/            KVarN PR #46812 backport
  fix-minimax-m3-kvarn-sparse/   sparse MSA on KVarN tiles + fused prefill/decode
  fix-minimax-m3-kvarn-indexer/  indexer K-cache on 4-bit tiles
  fix-kvarn-dense-fa-scratch-cap/  bounded FA scratch + LSE-windowed long-ctx path
  fix-eagle3-draft-embed-quant/  int4 drafter embed/lm_head loading
  minimax-m3-gptq-b12x-eagle3/   consolidated b12x stack mod group
  ... (see each recipe's mods: list; every mod dir has a documented run.sh)
recipes/                 the three production recipes (this file's table)
runs/                    benchmark data + narratives backing every number here
```
