#!/usr/bin/env python3
"""Strip the heavy ``vllm-rs`` server binary from vLLM's ``tools/build_rust.py``.

vLLM's ``rust_extensions()`` builds two artifacts:

  1. ``vllm.vllm-rs``          -- a full server/CLI binary (Binding.Exec) that
                                 pulls in the whole Rust server stack plus a
                                 vendored OpenSSL build.
  2. ``vllm._rust_tool_parser`` -- a small, pure-Rust PyO3 cdylib that powers
                                 ``--tool-call-parser minimax_m3`` (and the other
                                 Rust tool parsers).

We only need (2). Removing (1) keeps the Docker build fast and low-risk (no
OpenSSL/server compile) while still producing the tool-parser extension.

Run from the vLLM source root. Idempotent; verifies the result.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
path = ROOT / "tools" / "build_rust.py"
text = path.read_text()

VLLM_RS_BLOCK = '''        RustExtension(
            target="vllm.vllm-rs",
            path="rust/src/cmd/Cargo.toml",
            args=["--bin", "vllm-rs"],
            features=["native-tls-vendored"],
            binding=Binding.Exec,
            optional=optional,
        ),
'''

if "vllm.vllm-rs" not in text:
    print("strip-vllm-rs-binary: vllm-rs extension already absent; nothing to do")
    sys.exit(0)

if VLLM_RS_BLOCK not in text:
    print(
        "ERROR: could not find the exact vllm-rs RustExtension block in "
        f"{path}. Upstream may have changed its formatting; update this script.",
        file=sys.stderr,
    )
    sys.exit(1)

text = text.replace(VLLM_RS_BLOCK, "", 1)
path.write_text(text)

# Sanity: exactly the tool-parser extension remains.
if "vllm.vllm-rs" in text or "vllm._rust_tool_parser" not in text:
    print("ERROR: post-strip verification failed", file=sys.stderr)
    sys.exit(1)

print("strip-vllm-rs-binary: removed vllm-rs binary; keeping _rust_tool_parser only")
