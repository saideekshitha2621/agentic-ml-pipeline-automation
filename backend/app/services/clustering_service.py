"""Runs every registered clustering plugin over a config-driven hyperparameter grid."""
from __future__ import annotations

import numpy as np

from app.core.config import DEFAULT_HYPERPARAMETER_CONFIG
from app.plugins import PLUGIN_REGISTRY
from app.plugins.base import PluginRun


def run_all(X: np.ndarray, config: dict | None = None, progress_cb=None) -> list[PluginRun]:
    """config: {"kmeans": {...}, "dbscan": {...}, ...} overriding DEFAULT_HYPERPARAMETER_CONFIG
    per algorithm. progress_cb(pct: float, message: str), if given, is called after each
    algorithm finishes (drives the Model Execution screen's progress bar/log tail)."""
    config = config or {}
    merged = {
        name: {**DEFAULT_HYPERPARAMETER_CONFIG.get(name, {}), **config.get(name, {})}
        for name in PLUGIN_REGISTRY
    }

    all_runs: list[PluginRun] = []
    total = len(PLUGIN_REGISTRY)
    for i, (name, plugin) in enumerate(PLUGIN_REGISTRY.items(), start=1):
        if progress_cb:
            progress_cb((i - 1) / total * 90, f"Running {name}...")
        runs = plugin.run(X, merged[name])
        all_runs.extend(runs)
        if progress_cb:
            progress_cb(i / total * 90, f"{name} completed: {len(runs)} usable run(s).")

    return all_runs
