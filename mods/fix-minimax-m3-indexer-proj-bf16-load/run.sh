#!/bin/bash
set -euo pipefail

# MiniMax-M3: load the (bf16) lightning-indexer projections that the stock
# loader silently drops on compressed-tensors / GPTQ checkpoints.
#
# ROOT CAUSE
#   The sparse layers fuse [q | k | v | index_q | index_k] into a single
#   quantized column-parallel GEMM (MinimaxM3QKVParallelLinearWithIndexer,
#   quant_config => compressed-tensors WNA16 => only `weight_packed`/`_scale`/
#   `_shape` params). But the checkpoint keeps index_q_proj/index_k_proj
#   UNQUANTIZED: the quant config `ignore` list has `re:.*index_.*proj.*` and
#   `re:.*indexer.*`, so they ship as plain bf16 `index_q_proj.weight` /
#   `index_k_proj.weight`. The model's stacked_params_mapping remaps those to
#   `...qkv_proj.weight`, which DOES NOT EXIST on a quantized fused module
#   (it has `weight_packed`, not `weight`). The load loop hits
#   `if name not in params_dict: continue` and DROPS them, with no warning.
#   => the lightning indexer runs on uninitialized projection weights => its
#   per-block scores are garbage => top-k block SELECTION is wrong. This is
#   invisible for prompts <= sparse_topk_blocks*sparse_block_size (= 16*128 =
#   2048 tokens, where top-k selects *every* block so scores don't matter) and
#   produces degenerate/garbage output above it. KV-dtype independent.
#
# FIX (no CUDA kernel change, preserves the authors' bf16 indexer precision)
#   linear.py MinimaxM3QKVParallelLinearWithIndexer:
#     - keep the quantized fused weight exactly as-is (q/k/v load unchanged; the
#       index rows of the packed weight stay unused),
#     - add a dedicated bf16 `index_weight` Parameter [iq_local+ik_local, hidden]
#       with a custom loader that shards index_q like the KV heads and replicates
#       index_k (mirrors the bf16 QKV `weight_loader`),
#     - override forward() to overwrite ONLY the index columns of the fused GEMM
#       output with `F.linear(x, index_weight)`. The downstream fused
#       qknorm/rope/insert kernel then sees correct index_q/index_k.
#   nvidia/model.py load_weights:
#     - route `.index_q_proj.weight` / `.index_k_proj.weight` to
#       `.qkv_proj.index_weight` (instead of the nonexistent `.qkv_proj.weight`).
#
# SAFETY: q/k/v quantized loading and the fused kernel are untouched. Pure
# Python edits (no vLLM rebuild). No-op for dense layers (no index_*_proj keys).

PYTHON=${PYTHON:-python3}
$PYTHON - <<'PY'
from pathlib import Path
import importlib.util, shutil, py_compile

spec = importlib.util.find_spec("vllm")
vroot = Path(spec.submodule_search_locations[0])
L = vroot / "model_executor/layers/linear.py"
M = vroot / "models/minimax_m3/nvidia/model.py"
assert L.exists(), f"missing {L}"
assert M.exists(), f"missing {M}"

# ----------------------------------------------------------------- linear.py
lt = L.read_text()
if "self.index_weight" in lt:
    print("indexer bf16 projection already patched:", L)
else:
    init_anchor = (
        "            quant_config=quant_config,\n"
        "            prefix=prefix,\n"
        "        )\n"
        "\n"
        "    def validate_shard_id(self, loaded_shard_id: str | None) -> None:\n"
    )
    assert lt.count(init_anchor) == 1, "anchor: fused-linear __init__ tail / validate_shard_id"
    init_new = (
        "            quant_config=quant_config,\n"
        "            prefix=prefix,\n"
        "        )\n"
        "        # mod (indexer-proj-bf16-load): the checkpoint keeps\n"
        "        # index_q_proj/index_k_proj UNQUANTIZED (bf16) while this fused\n"
        "        # module is quantized, so they cannot load into the packed qkv\n"
        "        # weight. Hold them in a dedicated bf16 weight and splice their\n"
        "        # output into the index columns of the fused GEMM in forward().\n"
        "        _idx_rows = self.num_index_heads * self.head_size + self.index_head_size\n"
        "        _idx_dev = next((p.device for p in self.parameters()), None)\n"
        "        self.index_weight = Parameter(\n"
        "            torch.empty(\n"
        "                _idx_rows, self.hidden_size,\n"
        "                dtype=self.params_dtype, device=_idx_dev,\n"
        "            ),\n"
        "            requires_grad=False,\n"
        "        )\n"
        "        self.index_weight.output_dim = 0\n"
        "        self.index_weight.weight_loader = self._load_index_shard\n"
        "\n"
        "    def _load_index_shard(self, param, loaded_weight, loaded_shard_id):\n"
        "        # Mirror the bf16 QKV sharding: index_q rides the KV-head\n"
        "        # replication factor; index_k is one head replicated to all ranks.\n"
        "        h = self.head_size\n"
        "        if loaded_shard_id == \"index_q\":\n"
        "            shard_size = self.num_index_heads * h\n"
        "            shard_offset = 0\n"
        "            shard_rank = self.tp_rank // self.num_kv_head_replicas\n"
        "        elif loaded_shard_id == \"index_k\":\n"
        "            shard_size = self.index_head_size\n"
        "            shard_offset = self.num_index_heads * h\n"
        "            shard_rank = 0  # replicated\n"
        "        else:\n"
        "            raise ValueError(\n"
        "                f\"_load_index_shard expects index_q/index_k, got {loaded_shard_id!r}\"\n"
        "            )\n"
        "        param_data = param.data.narrow(0, shard_offset, shard_size)\n"
        "        loaded = loaded_weight.narrow(0, shard_rank * shard_size, shard_size)\n"
        "        assert param_data.shape == loaded.shape, (\n"
        "            f\"index shard {loaded_shard_id}: {tuple(param_data.shape)} \"\n"
        "            f\"vs {tuple(loaded.shape)}\"\n"
        "        )\n"
        "        param_data.copy_(loaded)\n"
        "\n"
        "    def forward(self, input_):\n"
        "        bias = self.bias if not self.skip_bias_add else None\n"
        "        output = self.quant_method.apply(self, input_, bias)\n"
        "        # Overwrite the (unused, quantized) index columns of the fused\n"
        "        # GEMM with the dedicated bf16 index projection.\n"
        "        h = self.head_size\n"
        "        idx_start = self.num_heads * h + 2 * self.num_kv_heads * h\n"
        "        idx = torch.nn.functional.linear(input_, self.index_weight)\n"
        "        output[..., idx_start : idx_start + idx.shape[-1]] = idx\n"
        "        if not self.return_bias:\n"
        "            return output\n"
        "        output_bias = self.bias if self.skip_bias_add else None\n"
        "        return output, output_bias\n"
        "\n"
        "    def validate_shard_id(self, loaded_shard_id: str | None) -> None:\n"
    )
    lt = lt.replace(init_anchor, init_new, 1)
    L.write_text(lt)
    py_compile.compile(str(L), doraise=True)
    print("Patched fused indexer bf16 projection (linear.py):", L)

# ----------------------------------------------------------------- model.py
mt = M.read_text()
map_old = (
    '            (".qkv_proj", ".index_q_proj", "index_q"),\n'
    '            (".qkv_proj", ".index_k_proj", "index_k"),\n'
)
map_new = (
    "            # mod (indexer-proj-bf16-load): the bf16 index projections live\n"
    "            # in a dedicated qkv_proj.index_weight (they cannot load into the\n"
    "            # quantized fused qkv weight_packed). Match the full `.weight`\n"
    "            # suffix so the remapped name targets the Parameter exactly.\n"
    '            (".qkv_proj.index_weight", ".index_q_proj.weight", "index_q"),\n'
    '            (".qkv_proj.index_weight", ".index_k_proj.weight", "index_k"),\n'
)
if ".qkv_proj.index_weight" in mt:
    print("model.py index mapping already patched:", M)
elif map_old in mt:
    mt = mt.replace(map_old, map_new, 1)
    M.write_text(mt)
    py_compile.compile(str(M), doraise=True)
    print("Patched index proj weight mapping (model.py):", M)
else:
    raise SystemExit("anchor not found: stacked_params_mapping index_q/index_k entries")

for d in (L.parent / "__pycache__", M.parent / "__pycache__"):
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
print("MiniMax-M3 indexer bf16 projection load fix applied.")
PY
