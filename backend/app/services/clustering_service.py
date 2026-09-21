"""Runs every registered clustering plugin over a config-driven hyperparameter grid."""
from __future__ import annotations

import logging

import numpy as np

from app.core.config import DEFAULT_HYPERPARAMETER_CONFIG
from app.plugins import PLUGIN_REGISTRY
from app.plugins.base import PluginRun

logger = logging.getLogger(__name__)


def run_all(
    X: np.ndarray,
    config: dict | None = None,
    progress_cb=None,
    algorithms: list[str] | None = None,
    on_algorithm_status=None,
) -> list[PluginRun]:
    """config: {"kmeans": {...}, "dbscan": {...}, ...} overriding DEFAULT_HYPERPARAMETER_CONFIG
    per algorithm. progress_cb(pct: float, message: str), if given, is called after each
    algorithm finishes (drives the Model Execution screen's progress bar/log tail).

    algorithms: restrict execution to this subset of registered plugin names (the Algorithm
    Recommendation HITL shortlist) — None/empty runs every registered plugin.
    on_algorithm_status(name, status): "running"|"completed"|"failed", for live progress."""
    config = config or {}
    names = [n for n in PLUGIN_REGISTRY if not algorithms or n in algorithms]
    merged = {name: {**DEFAULT_HYPERPARAMETER_CONFIG.get(name, {}), **config.get(name, {})} for name in names}

    all_runs: list[PluginRun] = []
    total = len(names) or 1
    for i, name in enumerate(names, start=1):
        plugin = PLUGIN_REGISTRY[name]
        if progress_cb:
            progress_cb((i - 1) / total * 90, f"Running {name}...")
        if on_algorithm_status:
            on_algorithm_status(name, "running")
        error = None
        try:
            runs = plugin.run(X, merged[name])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Clustering plugin %s crashed", name)
            runs, error = [], f"{type(exc).__name__}: {exc}"
        all_runs.extend(runs)
        if on_algorithm_status:
            on_algorithm_status(name, "completed" if runs else "failed")
        if progress_cb:
            if error:
                progress_cb(i / total * 90, f"{name} failed: {error}")
            elif not runs:
                progress_cb(i / total * 90, f"{name} produced no usable run (every setting failed or found <2 clusters).")
            else:
                progress_cb(i / total * 90, f"{name} completed: {len(runs)} usable run(s).")

    return all_runs
