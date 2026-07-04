#!/bin/bash
# TEMP: record allocator history w/ stacks across the memory-profiling
# window; dump snapshot + log the largest live allocations at peak.
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import py_compile
P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")
s = P.read_text()
a = "        with memory_profiling(\n"
if "MEMHIST" not in s:
    assert s.count(a) == 1
    s = s.replace(a,
        "        torch.cuda.memory._record_memory_history(  # MEMHIST\n"
        "            max_entries=200000)\n" + a, 1)
    b = "        # Use the pre-cudagraph torch peak to avoid double-counting.\n"
    assert s.count(b) == 1
    inj = '''        try:  # MEMHIST dump + top allocations
            import pickle
            snap = torch.cuda.memory._snapshot()
            with open("/tmp/mem_snapshot.pkl", "wb") as f:
                pickle.dump(snap, f)
            sizes = {}
            for seg in snap["segments"]:
                for blk in seg.get("blocks", []):
                    if blk.get("state") != "active_allocated":
                        continue
                    fr = blk.get("frames") or []
                    parts = []
                    for x in fr[:6]:
                        fn = str(x.get("filename", "?")).split("/")[-1]
                        parts.append(fn + ":" + str(x.get("line", 0)))
                    key = "|".join(parts) or "<no-stack>"
                    sizes[key] = sizes.get(key, 0) + blk["size"]
            for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:12]:
                logger.info("MEMHIST %8.1f MiB  %s", v / (1 << 20), k)
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception as e:
            logger.warning("MEMHIST failed: %s", e)
'''
    s = s.replace(b, inj + b, 1)
    P.write_text(s)
    py_compile.compile(str(P), doraise=True)
print("mem-history patch applied")
PY
