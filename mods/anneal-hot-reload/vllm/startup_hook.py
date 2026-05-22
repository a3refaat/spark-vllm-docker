"""
Anneal startup hook for vLLM.

Injected into the vLLM install at setup time. Loaded via importlib from
vllm/__init__.py. Starts the anneal control socket in a background thread.

This file is copied into the vLLM install path — it is NOT imported from the
anneal package at runtime.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

logger = logging.getLogger("anneal.startup")


def _anneal_startup():
    """Initialize anneal within the vLLM process."""
    try:
        reload_path = os.path.join(os.path.dirname(__file__), "_anneal_reload.py")
        spec = importlib.util.spec_from_file_location("_anneal_reload", reload_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_anneal_reload"] = mod
        spec.loader.exec_module(mod)
        mod.start_control_socket()
    except FileNotFoundError:
        logger.error("Anneal: _anneal_reload.py not found in vLLM install path")
    except Exception as e:
        logger.error(f"Anneal: startup hook failed: {e}")


# Auto-execute on import
_anneal_startup()
