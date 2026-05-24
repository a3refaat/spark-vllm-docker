#!/bin/bash
set -euo pipefail

# DGX Spark / GB10 needs a recent Nsight Systems build.  The Ubuntu SBSA apt
# repo currently provides 2024.2, which traces NVTX but fails CUDA/CUPTI on this
# stack.  Install the current public SBSA .run package instead.
NSYS_VERSION="2026.2.1"
NSYS_BUILD="2026.2.1.210-3763964"
NSYS_DIR="/opt/nvidia/nsight-systems/${NSYS_VERSION}"
NSYS_BIN="${NSYS_DIR}/target-linux-sbsa-armv8/nsys"
NSYS_URL="https://developer.download.nvidia.com/devtools/nsight-systems/NsightSystems-linux-sbsa-public-${NSYS_BUILD}.run"

if [[ -x "$NSYS_BIN" ]]; then
  ln -sf "$NSYS_BIN" /usr/local/bin/nsys
  echo "nsight-systems already installed: $(nsys --version 2>/dev/null | head -1)"
  if command -v sqlite3 >/dev/null 2>&1; then
    exit 0
  fi
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl sqlite3
rm -rf /var/lib/apt/lists/*

installer="/tmp/nsight-systems-${NSYS_BUILD}.run"
echo "Installing Nsight Systems ${NSYS_BUILD} for SBSA..."
curl -L --retry 3 -o "$installer" "$NSYS_URL"
rm -rf "$NSYS_DIR"
# Makeself consumes --accept/--quiet; installer-specific options go after --.
sh "$installer" --accept --quiet -- -targetpath="$NSYS_DIR" -noprompt >/tmp/nsight-systems-install.log 2>&1 || {
  cat /tmp/nsight-systems-install.log >&2 || true
  exit 1
}
rm -f "$installer"
ln -sf "$NSYS_BIN" /usr/local/bin/nsys

nsys --version
sqlite3 --version | head -1
