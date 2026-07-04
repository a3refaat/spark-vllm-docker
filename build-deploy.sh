#!/bin/bash
# build-deploy.sh — one-command build of the MiniMax-M3 2xGB10 deployment image
# ==============================================================================
# Produces vllm-node-minimax-m3-b12x (see Dockerfile.deploy):
#   1. base:  vLLM @ pinned ref + minimax-m3-fused-fp8-kv source patch + Rust
#             tool parser (built by build-and-copy.sh; ~1-2h cold, ccache-fast
#             after)
#   2. layer: vendored b12x kernel library (.pth, smoke-imported at build)
#   3. copy:  optional save|load distribution to the worker node(s)
#
# Usage:
#   ./build-deploy.sh                       # build on this node only
#   ./build-deploy.sh -c 169.254.83.4       # build + distribute to worker(s)
#   ./build-deploy.sh --skip-base           # only rebuild the b12x layer
#
# After building, serve with any production recipe, e.g.:
#   ./run-recipe.sh --no-ray -d recipes/minimax-m3-w4a16-gptq-kvarn.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VLLM_REF=979b56a66c969ab67655d2155ab4c6c5bed15f65
BASE_TAG=vllm-node-minimax-m3
DEPLOY_TAG=vllm-node-minimax-m3-b12x

SKIP_BASE=false
COPY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-base) SKIP_BASE=true ;;
        -c|--copy-to) COPY_ARGS+=(-c "$2" --copy-parallel); shift ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1 (see --help)"; exit 1 ;;
    esac
    shift
done

if [[ "$SKIP_BASE" == "false" ]]; then
    if docker image inspect "$BASE_TAG" >/dev/null 2>&1; then
        echo "== base $BASE_TAG already present (delete it to force a rebuild)"
    else
        echo "== building base $BASE_TAG (vLLM @ ${VLLM_REF:0:12} + fp8-kv patch + rust)"
        ./build-and-copy.sh \
            --vllm-ref "$VLLM_REF" \
            --apply-vllm-patch minimax-m3-fused-fp8-kv.patch \
            --build-rust \
            -t "$BASE_TAG"
    fi
fi

echo "== building $DEPLOY_TAG (b12x layer, vendored @ $(cat vendor/b12x/.vendored-from-commit 2>/dev/null | cut -c1-12))"
docker build -f Dockerfile.deploy -t "$DEPLOY_TAG" .

if [[ ${#COPY_ARGS[@]} -gt 0 ]]; then
    echo "== distributing $DEPLOY_TAG to worker(s)"
    ./build-and-copy.sh --no-build "${COPY_ARGS[@]}" -t "$DEPLOY_TAG"
fi

echo "== done: $DEPLOY_TAG"
echo "   next: ./hf-download.sh Sebesky/MiniMax-M3-W4A16-GPTQ  (plus the EAGLE3 drafter, see DEPLOYMENT.md)"
echo "         ./run-recipe.sh --no-ray -d recipes/minimax-m3-w4a16-gptq-kvarn.yaml"
