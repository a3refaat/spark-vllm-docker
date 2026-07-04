#!/bin/bash
# Evict the DRAFTER checkpoint's page-cache residue right after drafter
# load. The drafter loads via default_loader (mmap+copy): ~1.6 GB of clean
# page-cache pages sit inside the memory-accounting window on GB10 unified
# memory and get charged against available KV cache. Weights are already
# copied into torch tensors, so POSIX_FADV_DONTNEED is safe. Targeted
# fadvise ONLY — never drop_caches (instanttensor target weights are
# file-backed and must stay cached).
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import py_compile
P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")
s = P.read_text()
old = (
    '                if hasattr(self, "drafter"):\n'
    '                    logger.info_once("Loading drafter model...")\n'
    '                    if hasattr(self.drafter, "load_model"):\n'
    '                        self.drafter.load_model(self.model)\n'
)
if "_kvarn_fadvise_drafter" not in s:
    assert s.count(old) == 1, "drafter load anchor not found"
    s = s.replace(old, old +
        "                    _kvarn_fadvise_drafter(self.vllm_config)\n", 1)
    helper = '''
def _kvarn_fadvise_drafter(vllm_config) -> None:
    """Drop the drafter checkpoint's page-cache residue (weights already
    copied into torch tensors). Unified-memory accounting otherwise charges
    it against the KV budget. (mod: fix-eagle3-drafter-pagecache)"""
    import os
    try:
        spec = vllm_config.speculative_config
        root = getattr(spec.draft_model_config, "model", None) if spec else None
        if not root or not os.path.isdir(root):
            return
        freed = 0
        for dirpath, _, names in os.walk(root):
            for n in names:
                try:
                    fd = os.open(os.path.join(dirpath, n), os.O_RDONLY)
                    try:
                        freed += os.fstat(fd).st_size
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass
        logger.info("Drafter page-cache evicted: %.2f GiB advised DONTNEED",
                    freed / (1 << 30))
    except Exception as e:  # never break loading over cache hygiene
        logger.warning("drafter fadvise skipped: %s", e)

'''
    s = s.replace("\nclass GPUModelRunner", helper + "\nclass GPUModelRunner", 1)
    P.write_text(s)
    py_compile.compile(str(P), doraise=True)
print("drafter fadvise patch applied")
PY
