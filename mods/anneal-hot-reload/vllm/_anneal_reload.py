"""
Anneal hot-reload control socket for vLLM.

Injected into the vLLM install at setup time. Runs a lightweight HTTP server
on 127.0.0.1:8765 inside the inference server process. Provides endpoints for
kernel reload, profiler control, and health checks.

This file is copied into the vLLM install path — it is NOT imported from the
anneal package at runtime.
"""

from __future__ import annotations

import errno
import gc
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("anneal.reload")

# ── Globals set at startup ───────────────────────────────────────────────

_allowlist: List[str] = []
_reload_lock = threading.Lock()
_profiler_lock = threading.Lock()
_active_profiler: Any = None
_active_profiler_dir: Optional[str] = None
_rpc_targets: List[Any] = []

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 8765


def _load_allowlist() -> List[str]:
    """Load the reload allowlist from the config written at setup time."""
    config_path = Path(__file__).parent / "_anneal_allowlist.json"
    if config_path.exists():
        data = json.loads(config_path.read_text())
        return data.get("allowlist", [])
    return []


def _check_allowlist(module_path: str) -> bool:
    """Return True if module_path is permitted by the allowlist."""
    if not _allowlist:
        return False
    return any(module_path.startswith(prefix) for prefix in _allowlist)


# ── Triton reload ────────────────────────────────────────────────────────

def _reload_triton(module_path: str) -> Dict[str, Any]:
    """Reload a Triton module: clear caches, reimport, rebind parents."""
    import triton

    if module_path not in sys.modules:
        # Try importing it first
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            return {"success": False, "error": f"Module not found: {module_path}: {e}"}

    module = sys.modules[module_path]
    module_file = getattr(module, "__file__", None)

    # 1. Clear Triton file-system cache for this module
    triton_cache = Path.home() / ".triton" / "cache"
    if triton_cache.exists() and module_file:
        module_name = Path(module_file).stem
        for entry in triton_cache.iterdir():
            if entry.is_dir():
                # Triton cache dirs contain compiled kernels; clear matching ones
                for sub in entry.iterdir():
                    if module_name in sub.name:
                        try:
                            if sub.is_dir():
                                shutil.rmtree(sub)
                            else:
                                sub.unlink()
                        except OSError:
                            pass

    # 2. Clear in-memory JIT caches on @triton.jit decorated functions
    for attr_name in dir(module):
        attr = getattr(module, attr_name, None)
        if hasattr(attr, "cache"):
            try:
                attr.cache.clear()
            except Exception:
                pass
        # Also handle triton JITFunction objects
        if isinstance(attr, triton.runtime.jit.JITFunction):
            if hasattr(attr, "cache"):
                attr.cache.clear()

    # 3. Reload the module
    try:
        importlib.reload(module)
    except Exception as e:
        return {"success": False, "error": f"Reload failed: {e}\n{traceback.format_exc()}"}

    # 4. Rebind imports in parent modules
    reloaded = sys.modules[module_path]
    parts = module_path.split(".")
    leaf_name = parts[-1]

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is reloaded:
            continue
        if mod_name == module_path:
            continue
        # Check if this module imported something from the reloaded module
        for attr_name in dir(mod):
            try:
                attr = getattr(mod, attr_name)
                # If the attribute came from the old module, rebind it
                old_module = getattr(module, attr_name, None)
                new_val = getattr(reloaded, attr_name, None)
                if new_val is not None and attr is not new_val:
                    if hasattr(attr, "__module__") and attr.__module__ == module_path:
                        setattr(mod, attr_name, new_val)
            except Exception:
                pass
        # Also rebind if the module itself is referenced as an attribute
        if hasattr(mod, leaf_name) and getattr(mod, leaf_name) is module:
            setattr(mod, leaf_name, reloaded)

    # 5. Smoke test: try calling the primary exported function with trivial input
    # We attempt a lightweight verification but don't fail the reload if there's
    # no obvious callable to test
    smoke_error = None
    # Skip smoke test — the caller (experiment.py) does correctness verification

    return {"success": True, "error": None}


# ── CUDA reload ──────────────────────────────────────────────────────────

def _reload_cuda(module_path: str) -> Dict[str, Any]:
    """Reload a CUDA kernel via torch.utils.cpp_extension.load_inline."""
    import torch.utils.cpp_extension

    if module_path not in sys.modules:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            return {"success": False, "error": f"Module not found: {module_path}: {e}"}

    module = sys.modules[module_path]

    # Convention: anneal-managed CUDA modules expose load_and_bind()
    load_fn = getattr(module, "load_and_bind", None)
    if load_fn is None:
        return {"success": False, "error": f"Module {module_path} has no load_and_bind() function"}

    # Read the source file to get new CUDA source
    source_file = getattr(module, "__file__", None)
    if not source_file:
        return {"success": False, "error": f"Cannot determine source file for {module_path}"}

    try:
        source_text = Path(source_file).read_text()
    except OSError as e:
        return {"success": False, "error": f"Cannot read source: {e}"}

    # Hash-based build directory for caching
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()[:16]
    build_dir = Path(tempfile.gettempdir()) / "anneal_cuda_builds" / f"{module_path}_{source_hash}"
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Reload the module (which should re-read the modified source and recompile)
        importlib.reload(module)
        reloaded = sys.modules[module_path]

        # Call load_and_bind to compile and rebind
        if hasattr(reloaded, "load_and_bind"):
            reloaded.load_and_bind()

        # Rebind in parent modules
        parts = module_path.split(".")
        leaf_name = parts[-1]
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or mod is reloaded or mod_name == module_path:
                continue
            if hasattr(mod, leaf_name) and getattr(mod, leaf_name) is module:
                setattr(mod, leaf_name, reloaded)
            for attr_name in dir(mod):
                try:
                    attr = getattr(mod, attr_name)
                    if hasattr(attr, "__module__") and attr.__module__ == module_path:
                        new_val = getattr(reloaded, attr_name, None)
                        if new_val is not None:
                            setattr(mod, attr_name, new_val)
                except Exception:
                    pass

        return {"success": True, "error": None}

    except Exception as e:
        return {"success": False, "error": f"CUDA reload failed: {e}\n{traceback.format_exc()}"}


# ── Profiler control ─────────────────────────────────────────────────────

def _profiler_start(output_dir: str) -> Dict[str, Any]:
    global _active_profiler, _active_profiler_dir
    import torch.profiler

    with _profiler_lock:
        if _active_profiler is not None:
            return {"success": False, "error": "Profiler session already active"}

        os.makedirs(output_dir, exist_ok=True)
        _active_profiler_dir = output_dir

        _active_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            with_stack=False,
        )
        _active_profiler.__enter__()
        return {"success": True}


def _profiler_stop() -> Dict[str, Any]:
    global _active_profiler, _active_profiler_dir
    import torch.profiler

    with _profiler_lock:
        if _active_profiler is None:
            return {"success": False, "error": "No active profiler session"}

        try:
            _active_profiler.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"Profiler stop error: {e}")

        trace_files = []
        if _active_profiler_dir:
            trace_dir = Path(_active_profiler_dir)
            trace_files = [str(p) for p in trace_dir.glob("*.json")] + \
                          [str(p) for p in trace_dir.glob("*.json.gz")]

        _active_profiler = None
        _active_profiler_dir = None
        return {"success": True, "trace_files": trace_files}


def _worker_profiler_start(worker: Any, output_dir: str) -> Dict[str, Any]:
    result = _profiler_start(output_dir)
    try:
        result["worker_class"] = worker.__class__.__name__
        result["pid"] = os.getpid()
    except Exception:
        pass
    return result


def _worker_profiler_stop(worker: Any) -> Dict[str, Any]:
    result = _profiler_stop()
    try:
        result["worker_class"] = worker.__class__.__name__
        result["pid"] = os.getpid()
    except Exception:
        pass
    return result


def _profiler_start_distributed(output_dir: str) -> Dict[str, Any]:
    if _is_ray_worker_process():
        result = _profiler_start(output_dir)
        result["distributed"] = False
        result["ray_worker_local"] = True
        return result
    targets = _get_collective_rpc_targets()
    if targets:
        try:
            worker_results = targets[0].collective_rpc(_worker_profiler_start, timeout=120, args=(output_dir,))
            success = all(r.get("success", False) for r in worker_results)
            return {"success": success, "distributed": True, "worker_results": worker_results,
                    "error": "; ".join(r.get("error", "") for r in worker_results if not r.get("success", False))}
        except Exception as e:
            logger.warning("Distributed profiler start failed: %s", e)
    result = _profiler_start(output_dir)
    result["distributed"] = False
    return result


def _profiler_stop_distributed() -> Dict[str, Any]:
    if _is_ray_worker_process():
        result = _profiler_stop()
        result["distributed"] = False
        result["ray_worker_local"] = True
        return result
    targets = _get_collective_rpc_targets()
    if targets:
        try:
            worker_results = targets[0].collective_rpc(_worker_profiler_stop, timeout=120)
            success = all(r.get("success", False) for r in worker_results)
            trace_files: List[str] = []
            for r in worker_results:
                trace_files.extend(r.get("trace_files", []) or [])
            return {"success": success, "distributed": True, "worker_results": worker_results,
                    "trace_files": trace_files,
                    "error": "; ".join(r.get("error", "") for r in worker_results if not r.get("success", False))}
        except Exception as e:
            logger.warning("Distributed profiler stop failed: %s", e)
    result = _profiler_stop()
    result["distributed"] = False
    return result


# ── Nsight Systems support ───────────────────────────────────────────────


def _cuda_profiler_api(action: str) -> Dict[str, Any]:
    """Call cudaProfilerStart/Stop in the current worker process."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cudart = torch.cuda.cudart()
        fn = cudart.cudaProfilerStart if action == "start" else cudart.cudaProfilerStop
        rc = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {"success": int(rc) == 0, "return_code": int(rc), "pid": os.getpid(), "action": action}
    except Exception as e:
        return {"success": False, "error": f"cudaProfiler{action.title()} failed: {e}\n{traceback.format_exc()}", "pid": os.getpid()}


def _worker_cuda_profiler(worker: Any, action: str) -> Dict[str, Any]:
    result = _cuda_profiler_api(action)
    try:
        result["worker_class"] = worker.__class__.__name__
        runner = getattr(worker, "model_runner", None)
        result["runner_class"] = runner.__class__.__name__ if runner else ""
        import torch
        result["device"] = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    except Exception:
        pass
    return result


def _cuda_profiler_distributed(action: str) -> Dict[str, Any]:
    if _is_ray_worker_process():
        result = _cuda_profiler_api(action)
        result["distributed"] = False
        result["ray_worker_local"] = True
        return result
    targets = _get_collective_rpc_targets()
    if targets:
        try:
            worker_results = targets[0].collective_rpc(_worker_cuda_profiler, timeout=120, args=(action,))
            success = all(r.get("success", False) for r in worker_results)
            return {"success": success, "distributed": True, "worker_results": worker_results,
                    "error": "; ".join(r.get("error", "") for r in worker_results if not r.get("success", False))}
        except Exception as e:
            logger.warning("Distributed cuda profiler %s failed: %s", action, e)
    result = _cuda_profiler_api(action)
    result["distributed"] = False
    return result


def _worker_runtime_info(worker: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "success": True,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "worker_class": worker.__class__.__name__,
    }
    try:
        import torch
        info["device"] = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
        props = torch.cuda.get_device_properties(torch.cuda.current_device()) if torch.cuda.is_available() else None
        if props is not None:
            info["device_name"] = props.name
            info["device_uuid"] = str(getattr(props, "uuid", ""))
    except Exception as e:
        info["device_error"] = str(e)
    try:
        rank = getattr(worker, "rank", None)
        local_rank = getattr(worker, "local_rank", None)
        if rank is not None:
            info["rank"] = int(rank)
        if local_rank is not None:
            info["local_rank"] = int(local_rank)
        runner = getattr(worker, "model_runner", None)
        info["runner_class"] = runner.__class__.__name__ if runner else ""
    except Exception:
        pass
    return info


def _runtime_workers_distributed() -> Dict[str, Any]:
    if _is_ray_worker_process():
        return {"success": True, "distributed": False, "workers": [_worker_runtime_info(None)], "ray_worker_local": True}
    targets = _get_collective_rpc_targets()
    if targets:
        try:
            worker_results = targets[0].collective_rpc(_worker_runtime_info, timeout=120)
            workers = []
            for i, r in enumerate(worker_results):
                r = dict(r)
                r.setdefault("rank", i)
                workers.append(r)
            return {"success": True, "distributed": True, "worker_count": len(workers), "workers": workers}
        except Exception as e:
            logger.warning("Distributed runtime worker discovery failed: %s", e)
            return {"success": False, "distributed": True, "error": str(e)}
    return {"success": True, "distributed": False, "workers": [{"success": True, "pid": os.getpid(), "hostname": socket.gethostname()}]}


# ── Live vLLM resolver + CUDA graph refresh ──────────────────────────────

_LIBRARY_PREFIXES = (
    "flashinfer", "cublas", "cudnn", "nccl", "cutlass", "sm80", "sm90",
    "void cutlass", "ampere_", "at::native", "void at::native",
)


def _strip_triton_suffix(kernel_name: str) -> str:
    """Best-effort stripping of Triton specialization/hash suffixes."""
    name = kernel_name.split("(", 1)[0].strip()
    # Repeatedly strip trailing hash-ish chunks while keeping normal names like
    # rms_norm_kernel intact. Triton suffixes usually contain hex/digit chunks.
    while True:
        new = re.sub(r"_[0-9a-f]{6,}$", "", name)
        if new == name:
            return name
        name = new


def _module_path_from_file(path: str) -> str:
    try:
        import vllm
        base = Path(vllm.__file__).resolve().parent
        p = Path(path).resolve()
        rel = p.relative_to(base.parent)
        return ".".join(rel.with_suffix("").parts)
    except Exception:
        return ""


def _is_triton_jit_function(obj: Any) -> bool:
    try:
        from triton.runtime.jit import JITFunction
        return isinstance(obj, JITFunction)
    except Exception:
        return hasattr(obj, "fn") and hasattr(obj, "raw_src")


def _resolution_from_callable(kernel_name: str, module_name: str, attr_name: str, obj: Any) -> Optional[Dict[str, Any]]:
    names = {attr_name}
    for field in ("__name__", "_fn_name", "name"):
        val = getattr(obj, field, None)
        if val:
            names.add(str(val))
    fn = getattr(obj, "fn", None)
    if fn is not None:
        names.add(getattr(fn, "__name__", attr_name))

    if not any(kernel_name == n or kernel_name.startswith(n + "_") or _strip_triton_suffix(kernel_name) == n for n in names):
        return None

    source_obj = fn if fn is not None else obj
    source_file = inspect.getsourcefile(source_obj) or inspect.getfile(source_obj)
    line = getattr(getattr(source_obj, "__code__", None), "co_firstlineno", None)
    return {
        "kernel_name": kernel_name,
        "matched_name": sorted(names)[0],
        "source_file": source_file,
        "line": line,
        "module_path": module_name or _module_path_from_file(source_file),
        "backend": "triton" if _is_triton_jit_function(obj) else "python",
        "modifiable": bool(module_name and _check_allowlist(module_name)),
        "source": "live_sys_modules",
    }


def _resolve_kernel_local(kernel_name: str) -> Dict[str, Any]:
    """Resolve using the live process state, not static host-side grep.

    This intentionally reuses vLLM's already-imported runtime: loaded modules,
    Triton JITFunction objects, CompilationConfig traced files, and
    static_forward_context. It is the ground-truth view used by the server.
    """
    lower = kernel_name.lower()
    if lower.startswith(_LIBRARY_PREFIXES):
        return {
            "success": True,
            "kernel_name": kernel_name,
            "resolutions": [{
                "kernel_name": kernel_name,
                "backend": "library",
                "modifiable": False,
                "source": "library_prefix",
                "reason": "library kernel; optimize caller or skip",
            }],
        }

    resolutions: List[Dict[str, Any]] = []

    # 1. Loaded vLLM modules and Triton JIT/Python callables.
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("vllm") or module is None:
            continue
        try:
            items = list(vars(module).items())
        except Exception:
            continue
        for attr_name, obj in items:
            try:
                if _is_triton_jit_function(obj) or callable(obj):
                    resolved = _resolution_from_callable(kernel_name, module_name, attr_name, obj)
                    if resolved:
                        resolutions.append(resolved)
            except Exception:
                continue

    # 2. torch.compile / Inductor generated kernels: use vLLM-tracked files and
    # inductor cache locations. These are usually not v1-modifiable, but the
    # traced files identify the upstream vLLM source that produced the IR.
    if lower.startswith(("triton_poi_", "triton_red_", "triton_per_fused_", "triton_")):
        traced_files = sorted(_collect_traced_files())
        resolutions.append({
            "kernel_name": kernel_name,
            "backend": "inductor_triton",
            "modifiable": False,
            "source": "vllm_compilation_config.traced_files",
            "traced_files": traced_files,
            "reason": "Inductor-generated kernel; modify upstream traced source or compilation boundary, not cache artifact",
        })

    # 3. CUDA/custom op source hints from loaded .so and csrc when present.
    if not resolutions:
        cuda_hits = _resolve_cuda_symbol(kernel_name)
        resolutions.extend(cuda_hits)

    # Prefer allowlisted, modifiable resolutions first.
    resolutions.sort(key=lambda r: (not r.get("modifiable", False), r.get("backend", ""), r.get("source_file", "")))
    return {"success": True, "kernel_name": kernel_name, "resolutions": resolutions}


def _collect_traced_files() -> set[str]:
    files: set[str] = set()
    for obj in gc.get_objects():
        try:
            cc = getattr(obj, "compilation_config", None)
            traced = getattr(cc, "traced_files", None)
            if traced:
                files.update(str(x) for x in traced)
        except Exception:
            pass
    return files


def _resolve_cuda_symbol(kernel_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        import vllm
        vllm_dir = Path(vllm.__file__).resolve().parent
    except Exception:
        return results

    # Source may be absent in wheel installs. If present, search csrc only.
    for csrc in [vllm_dir / "csrc", vllm_dir.parent / "csrc"]:
        if csrc.exists():
            for path in csrc.rglob("*"):
                if path.suffix not in (".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp"):
                    continue
                try:
                    text = path.read_text(errors="ignore")
                except Exception:
                    continue
                if kernel_name in text:
                    results.append({
                        "kernel_name": kernel_name,
                        "source_file": str(path),
                        "module_path": "",
                        "backend": "cuda",
                        "modifiable": False,
                        "source": "live_vllm_csrc",
                    })
    return results


def _process_cmdline() -> str:
    try:
        return Path("/proc/self/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except Exception:
        return ""


def _is_ray_worker_process() -> bool:
    return "RayWorkerProc.run" in _process_cmdline()


_RPC_TARGET_MODULE_PREFIXES = (
    "vllm.v1.engine.core",
    "vllm.v1.engine.llm_engine",
    "vllm.v1.engine.core_client",
    "vllm.v1.executor.",
    "vllm.engine.",
    "vllm.executor.",
)


def _get_collective_rpc_targets() -> List[Any]:
    return _rpc_targets or _find_collective_rpc_targets()


def _find_collective_rpc_targets() -> List[Any]:
    targets: List[Any] = []
    seen: set[int] = set()
    for obj in gc.get_objects():
        try:
            if id(obj) in seen:
                continue
            cls = obj.__class__
            cls_mod = getattr(cls, "__module__", "")
            if not cls_mod.startswith(_RPC_TARGET_MODULE_PREFIXES):
                continue
            # Check the class dictionary/MRO first.  This avoids dynamic
            # __getattr__ probes on vLLM platform objects that emit warnings.
            class_method = getattr(cls, "collective_rpc", None)
            if not callable(class_method):
                continue
            method = getattr(obj, "collective_rpc", None)
            if callable(method):
                targets.append(obj)
                seen.add(id(obj))
        except Exception:
            pass
    # Prefer the EngineCore coordinator over executor/client wrappers.
    targets.sort(key=lambda o: (o.__class__.__module__ != "vllm.v1.engine.core", o.__class__.__name__))
    return targets


def _find_model_runners() -> List[Any]:
    runners: List[Any] = []
    seen: set[int] = set()
    for obj in gc.get_objects():
        try:
            if id(obj) in seen:
                continue
            cls = obj.__class__
            cls_mod = getattr(cls, "__module__", "")
            # Avoid triggering __getattr__ on unrelated lazy modules (notably
            # transformers image processors) when searching for capture_model.
            if not cls_mod.startswith("vllm"):
                continue
            capture = getattr(obj, "capture_model", None)
            if callable(capture) and hasattr(obj, "compilation_config"):
                runners.append(obj)
                seen.add(id(obj))
        except Exception:
            pass
    return runners


def _worker_reload_recapture(worker: Any, module_path: str, backend: str) -> Dict[str, Any]:
    """Callable sent through vLLM collective_rpc to execute on workers."""
    if backend == "triton":
        result = _reload_triton(module_path)
    elif backend == "cuda":
        result = _reload_cuda(module_path)
    else:
        result = {"success": False, "error": f"Unknown backend: {backend}"}
    if not result.get("success"):
        return result
    refresh = _refresh_cuda_graphs_local()
    result["cudagraph_refresh"] = refresh
    result["success"] = bool(refresh.get("success", True))
    if not result["success"]:
        result["error"] = refresh.get("error", "CUDA graph refresh failed")
    return result


def _worker_resolve_kernel(worker: Any, kernel_name: str) -> Dict[str, Any]:
    result = _resolve_kernel_local(kernel_name)
    try:
        runner = getattr(worker, "model_runner", None)
        cc = getattr(runner, "compilation_config", None)
        result["worker_context"] = {
            "worker_class": worker.__class__.__name__,
            "runner_class": runner.__class__.__name__ if runner else "",
            "cudagraph_mode": str(getattr(cc, "cudagraph_mode", "")),
            "traced_files_count": len(getattr(cc, "traced_files", []) or []),
            "static_forward_context_keys": list((getattr(cc, "static_forward_context", {}) or {}).keys())[:200],
        }
    except Exception:
        pass
    return result


def _worker_refresh_cuda_graphs(worker: Any) -> Dict[str, Any]:
    """Refresh CUDA graphs on a concrete vLLM worker without scanning gc.

    The generic gc scanner can touch lazy objects from transformers/torch and
    emit thousands of warnings.  The collective_rpc worker argument already has
    the model_runner we need.
    """
    try:
        import torch
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        try:
            from vllm.v1.worker.workspace import unlock_workspace
            unlock_workspace()
        except Exception:
            pass

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        CUDAGraphWrapper.clear_all_graphs()
        runner = getattr(worker, "model_runner", None)
        if runner is None or not callable(getattr(runner, "capture_model", None)):
            return {"success": True, "captured": [], "runner_count": 0, "worker": worker.__class__.__name__}
        size = runner.capture_model()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return {
            "success": True,
            "captured": [{"runner": runner.__class__.__name__, "bytes": int(size or 0)}],
            "runner_count": 1,
            "worker": worker.__class__.__name__,
            "pid": os.getpid(),
        }
    except Exception as e:
        return {"success": False, "error": f"CUDA graph refresh failed: {e}\n{traceback.format_exc()}"}


def _resolve_kernel_distributed(kernel_name: str) -> Dict[str, Any]:
    if _is_ray_worker_process():
        result = _resolve_kernel_local(kernel_name)
        result["distributed"] = False
        result["worker_count"] = 1
        result["ray_worker_local"] = True
        return result
    targets = _get_collective_rpc_targets()
    if targets:
        # Use vLLM's own executor/Ray RPC path so resolution runs in the same
        # worker processes that own the model, compiled IR, and CUDA graphs.
        target = targets[0]
        try:
            worker_results = target.collective_rpc(_worker_resolve_kernel, timeout=120, args=(kernel_name,))
            merged: List[Dict[str, Any]] = []
            contexts: List[Dict[str, Any]] = []
            seen = set()
            for wr in worker_results:
                contexts.append(wr.get("worker_context", {}))
                for r in wr.get("resolutions", []):
                    key = json.dumps(r, sort_keys=True, default=str)
                    if key not in seen:
                        merged.append(r)
                        seen.add(key)
            merged.sort(key=lambda r: (not r.get("modifiable", False), r.get("backend", ""), r.get("source_file", "")))
            return {
                "success": True,
                "kernel_name": kernel_name,
                "distributed": True,
                "worker_count": len(worker_results),
                "worker_contexts": contexts,
                "resolutions": merged,
            }
        except Exception as e:
            logger.warning("Distributed kernel resolution failed: %s", e)
    result = _resolve_kernel_local(kernel_name)
    result["distributed"] = False
    result["worker_count"] = 1
    return result


def _refresh_cuda_graphs_local() -> Dict[str, Any]:
    """Use vLLM's own CUDA graph mechanisms to clear and recapture graphs."""
    try:
        import torch
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        try:
            from vllm.v1.worker.workspace import unlock_workspace
            unlock_workspace()
        except Exception:
            pass

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        CUDAGraphWrapper.clear_all_graphs()
        runners = _find_model_runners()
        captured = []
        for runner in runners:
            try:
                size = runner.capture_model()
                captured.append({"runner": runner.__class__.__name__, "bytes": int(size or 0)})
            except Exception as e:
                captured.append({"runner": runner.__class__.__name__, "error": str(e)})
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        errors = [x for x in captured if "error" in x]
        if errors:
            return {"success": False, "captured": captured, "error": "; ".join(e["error"] for e in errors)}
        return {"success": True, "captured": captured, "runner_count": len(runners)}
    except Exception as e:
        return {"success": False, "error": f"CUDA graph refresh failed: {e}\n{traceback.format_exc()}"}


def _refresh_cuda_graphs_distributed() -> Dict[str, Any]:
    if _is_ray_worker_process():
        return {
            "success": False,
            "distributed": False,
            "ray_worker_local": True,
            "error": "CUDA graph refresh from a single RayWorker is unsafe in TP; bind control to EngineCore for collective refresh",
        }
    targets = _get_collective_rpc_targets()
    if targets:
        try:
            worker_results = targets[0].collective_rpc(_worker_refresh_cuda_graphs, timeout=300)
            success = all(r.get("success", False) for r in worker_results)
            return {"success": success, "distributed": True, "worker_results": worker_results}
        except Exception as e:
            logger.warning("Distributed CUDA graph refresh failed: %s", e)
    result = _refresh_cuda_graphs_local()
    result["distributed"] = False
    return result


def _reload_and_refresh(module_path: str, backend: str) -> Dict[str, Any]:
    if _is_ray_worker_process():
        return {
            "success": False,
            "distributed": False,
            "ray_worker_local": True,
            "error": "reload+graph refresh from a single RayWorker is unsafe in TP; bind control to EngineCore for collective reload",
        }
    targets = _get_collective_rpc_targets()
    if targets:
        try:
            worker_results = targets[0].collective_rpc(_worker_reload_recapture, timeout=300, args=(module_path, backend))
            success = all(r.get("success", False) for r in worker_results)
            return {"success": success, "distributed": True, "worker_results": worker_results,
                    "error": "; ".join(r.get("error", "") for r in worker_results if not r.get("success", False))}
        except Exception as e:
            logger.warning("Distributed reload+recapture failed: %s", e)

    if backend == "triton":
        result = _reload_triton(module_path)
    elif backend == "cuda":
        result = _reload_cuda(module_path)
    else:
        return {"success": False, "error": f"Unknown backend: {backend}"}
    if result.get("success"):
        result["cudagraph_refresh"] = _refresh_cuda_graphs_local()
        if not result["cudagraph_refresh"].get("success", True):
            result["success"] = False
            result["error"] = result["cudagraph_refresh"].get("error", "CUDA graph refresh failed")
    result["distributed"] = False
    return result


# ── Health ───────────────────────────────────────────────────────────────

def _health() -> Dict[str, Any]:
    result = {"status": "ok", "server": "vllm", "pid": os.getpid()}
    try:
        import vllm
        result["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except ImportError:
        result["vllm_version"] = "not_found"

    try:
        targets = _get_collective_rpc_targets()
        result["rpc_target_count"] = len(targets)
        result["rpc_targets"] = [
            f"{t.__class__.__module__}.{t.__class__.__name__}"
            for t in targets[:10]
        ]
    except Exception as e:
        result["rpc_target_error"] = str(e)

    return result


# ── HTTP handler ─────────────────────────────────────────────────────────

class _AnnealHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the anneal control socket."""

    def log_message(self, format, *args):
        logger.debug(format, *args)

    def _send_json(self, data: Dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(_health())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = self._read_body()
        except Exception as e:
            self._send_json({"error": f"Invalid request body: {e}"}, 400)
            return

        if self.path == "/reload":
            module_path = body.get("module_path", "")
            backend = body.get("backend", "triton")
            recapture = body.get("recapture", True)

            if not module_path:
                self._send_json({"error": "module_path required"}, 400)
                return

            if not _check_allowlist(module_path):
                self._send_json({
                    "error": f"Module {module_path} not in allowlist. "
                             f"Allowed prefixes: {_allowlist}"
                }, 403)
                return

            with _reload_lock:
                if recapture:
                    result = _reload_and_refresh(module_path, backend)
                elif backend == "triton":
                    result = _reload_triton(module_path)
                elif backend == "cuda":
                    result = _reload_cuda(module_path)
                else:
                    self._send_json({"error": f"Unknown backend: {backend}"}, 400)
                    return

            status = 200 if result["success"] else 500
            self._send_json(result, status)

        elif self.path == "/resolve_kernel":
            kernel_name = body.get("kernel_name", "") or body.get("name", "")
            if not kernel_name:
                self._send_json({"error": "kernel_name required"}, 400)
                return
            result = _resolve_kernel_distributed(kernel_name)
            self._send_json(result, 200 if result.get("success") else 500)

        elif self.path == "/cudagraph/refresh":
            with _reload_lock:
                result = _refresh_cuda_graphs_distributed()
            self._send_json(result, 200 if result.get("success") else 500)

        elif self.path == "/runtime/workers":
            result = _runtime_workers_distributed()
            self._send_json(result, 200 if result.get("success") else 500)

        elif self.path == "/cuda_profiler/start":
            result = _cuda_profiler_distributed("start")
            self._send_json(result, 200 if result.get("success") else 500)

        elif self.path == "/cuda_profiler/stop":
            result = _cuda_profiler_distributed("stop")
            self._send_json(result, 200 if result.get("success") else 500)

        elif self.path == "/profiler/start":
            output_dir = body.get("output_dir", "")
            if not output_dir:
                self._send_json({"error": "output_dir required"}, 400)
                return
            result = _profiler_start_distributed(output_dir)
            status = 200 if result.get("success") else 409
            self._send_json(result, status)

        elif self.path == "/profiler/stop":
            result = _profiler_stop_distributed()
            status = 200 if result.get("success") else 409
            self._send_json(result, status)

        else:
            self._send_json({"error": "not found"}, 404)


# ── Server startup ───────────────────────────────────────────────────────

_server_instance: Optional[HTTPServer] = None
_bind_thread_started = False


def _runtime_ready_for_control() -> bool:
    """Return True in a process that owns vLLM runtime state/GPU work.

    vLLM imports the top-level package in API, Ray helper, and worker processes.
    Binding the fixed control port in the API process is wrong: profiling then
    captures no GPU kernels.  In Ray deployments, bind from RayWorkerProc.run;
    in non-Ray deployments, bind once model runners/collective RPC targets are
    visible.  ANNEAL_BIND_IMMEDIATE is an escape hatch for debugging.
    """
    if os.environ.get("ANNEAL_BIND_IMMEDIATE", "").lower() in ("1", "true", "yes"):
        return True

    try:
        cmdline = Path("/proc/self/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except Exception:
        cmdline = ""

    if "vllm serve" in cmdline:
        return False

    # In Ray TP deployments, bind in EngineCore (the process that can dispatch
    # collective_rpc to *all* TP workers). Binding to a single RayWorker makes
    # CUDA graph capture hang because the peer TP rank never enters collectives.
    if "VLLM::EngineCore" in cmdline:
        try:
            return bool(_find_collective_rpc_targets())
        except Exception:
            return False

    # RayWorker binding is useful for narrow profiling debug, but unsafe for
    # graph refresh/reload in TP. Keep it opt-in only.
    if "RayWorkerProc.run" in cmdline:
        return os.environ.get("ANNEAL_BIND_RAY_WORKER", "").lower() in ("1", "true", "yes")

    # Non-Ray fallback is opt-in so helper processes do not win the port.
    if os.environ.get("ANNEAL_BIND_NON_RAY", "").lower() not in ("1", "true", "yes"):
        return False

    try:
        if _find_collective_rpc_targets():
            return True
        if _find_model_runners():
            return True
    except Exception:
        pass
    return False


def _bind_control_socket_when_ready() -> None:
    global _allowlist, _server_instance, _rpc_targets

    timeout = float(os.environ.get("ANNEAL_CONTROL_BIND_TIMEOUT_SEC", "1800"))
    poll = float(os.environ.get("ANNEAL_CONTROL_BIND_POLL_SEC", "2"))
    deadline = time.time() + timeout

    while time.time() < deadline:
        if _server_instance is not None:
            return
        if _runtime_ready_for_control():
            break
        time.sleep(poll)
    else:
        logger.info("Anneal: no vLLM runtime state found in this process; not binding control socket")
        return

    _rpc_targets = _find_collective_rpc_targets()
    _allowlist = _load_allowlist()
    if not _allowlist:
        logger.warning("Anneal: no allowlist configured. All reload requests will be denied.")

    try:
        server = HTTPServer((CONTROL_HOST, CONTROL_PORT), _AnnealHandler)
        _server_instance = server
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="anneal-control")
        thread.start()
        logger.info(f"Anneal control socket listening on {CONTROL_HOST}:{CONTROL_PORT}")
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            logger.info(
                f"Anneal control socket already listening on {CONTROL_HOST}:{CONTROL_PORT}; "
                "skipping duplicate startup in this vLLM process"
            )
        else:
            logger.error(f"Anneal: failed to start control socket: {e}")


def start_control_socket() -> None:
    """Start the control socket in a daemon thread. Called from startup hook."""
    global _bind_thread_started
    if _bind_thread_started:
        return
    _bind_thread_started = True
    thread = threading.Thread(
        target=_bind_control_socket_when_ready,
        daemon=True,
        name="anneal-control-bind",
    )
    thread.start()
