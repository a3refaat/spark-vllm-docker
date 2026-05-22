#!/bin/bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_DIR="$(python3 - <<'PY'
import os
import vllm
print(os.path.dirname(vllm.__file__))
PY
)"

if [[ -z "$VLLM_DIR" || ! -d "$VLLM_DIR" ]]; then
  echo "Could not locate vLLM install directory" >&2
  exit 1
fi

cp "$MOD_DIR/vllm/_anneal_reload.py" "$VLLM_DIR/_anneal_reload.py"
cp "$MOD_DIR/vllm/startup_hook.py" "$VLLM_DIR/_anneal_startup_hook.py"
cp "$MOD_DIR/vllm/_anneal_allowlist.json" "$VLLM_DIR/_anneal_allowlist.json"

python3 - "$VLLM_DIR/__init__.py" <<'PY'
from pathlib import Path
import sys

init_path = Path(sys.argv[1])
marker = "# anneal-startup-hook"
hook_code = f'''
{marker}
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        '_anneal_startup_hook',
        __import__('os').path.join(__import__('os').path.dirname(__file__), '_anneal_startup_hook.py'))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
except Exception:
    pass
'''
text = init_path.read_text()
if marker not in text:
    init_path.write_text(text + hook_code)
PY

echo "Anneal hot-reload patch installed in $VLLM_DIR"
