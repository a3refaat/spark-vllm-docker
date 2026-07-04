#!/bin/bash
set -euo pipefail

# The EAGLE3 draft model (llama_eagle3.py) creates its lm_head with the drafter's
# quant_config but creates embed_tokens WITHOUT one. For a self-contained int4
# drafter that keeps its OWN (quantized) embed_tokens, the embedding must be built
# with the draft quant_config or the int4 weight_packed/weight_scale can't load
# (param is created bf16). This adds quant_config=self.quant_config to that call.
# No-op (idempotent) if already patched; harmless for bf16 drafters (None config).

PYTHON=${PYTHON:-python3}
$PYTHON - <<'PY'
from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama_eagle3.py"
)
text = path.read_text()

old = (
    "        self.embed_tokens = VocabParallelEmbedding(\n"
    "            self.config.vocab_size,\n"
    "            self.config.hidden_size,\n"
    "            prefix=maybe_prefix(prefix, \"embed_tokens\"),\n"
    "        )"
)
new = (
    "        self.embed_tokens = VocabParallelEmbedding(\n"
    "            self.config.vocab_size,\n"
    "            self.config.hidden_size,\n"
    "            quant_config=self.quant_config,  # __EAGLE3_EMBED_QUANT__\n"
    "            prefix=maybe_prefix(prefix, \"embed_tokens\"),\n"
    "        )"
)

if "__EAGLE3_EMBED_QUANT__" in text:
    print("EAGLE3 draft embed quant_config already patched:", path)
    raise SystemExit(0)
if old not in text:
    raise SystemExit("Could not find draft embed_tokens VocabParallelEmbedding block to patch")
text = text.replace(old, new, 1)
path.write_text(text)
print("Patched draft embed_tokens to use draft quant_config:", path)
PY

# --- Patch 2: _maybe_share_embeddings compares draft-vs-target embed via
# target_embed_tokens.weight WITHOUT a hasattr guard. A self-contained int4
# drafter has has_own_embed_tokens=True -> enters that compare branch -> the
# COMPRESSED target embed (CompressedTensorsEmbeddingWNA16Int) has no `.weight`
# -> AttributeError. lm_head's analogous branch is already hasattr-guarded; mirror
# that here so it falls through to "keep separate draft embed" (what we want).
$PYTHON - <<'PY'
from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py"
)
text = path.read_text()

old = (
    "                elif (\n"
    "                    isinstance(target_embed_tokens.weight, torch.Tensor)\n"
    "                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)"
)
new = (
    "                elif (  # __EAGLE3_EMBED_SHARE_GUARD__\n"
    "                    hasattr(target_embed_tokens, \"weight\")\n"
    "                    and hasattr(self.model.model.embed_tokens, \"weight\")\n"
    "                    and isinstance(target_embed_tokens.weight, torch.Tensor)\n"
    "                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)"
)

if "__EAGLE3_EMBED_SHARE_GUARD__" in text:
    print("EAGLE3 share-embeddings guard already patched:", path)
elif old not in text:
    raise SystemExit("Could not find _maybe_share_embeddings compare block to patch")
else:
    text = text.replace(old, new, 1)
    path.write_text(text)
    print("Patched _maybe_share_embeddings with hasattr guard:", path)
PY

$PYTHON -m py_compile \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama_eagle3.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py
echo "EAGLE3 draft embed quant_config + share-guard mod applied."
