"""Runs every registered classification plugin over a config-driven hyperparameter grid.

Mirrors clustering_service.run_all() exactly — same merge-config-over-defaults, same
per-algorithm progress callback — against the classification plugin registry instead.
"""
from __future__ import annotations

import numpy as np

from app.core.config import DEFAULT_CLASSIFICATION_HYPERPARAMETER_CONFIG
from app.plugins import CLASSIFICATION_PLUGIN_REGISTRY
from app.plugins.classification_base import ClassificationPluginRun


def run_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: dict | None = None,
    progress_cb=None,
    algorithms: list[str] | None = None,
    on_algorithm_status=None,
) -> list[ClassificationPluginRun]:
    """config: {"logistic_regression": {...}, "random_forest": {...}, ...} overriding
    DEFAULT_CLASSIFICATION_HYPERPARAMETER_CONFIG per algorithm.

    algorithms: restrict execution to this subset of registered plugin names (the
    Algorithm Recommendation HITL shortlist) — None/empty runs every registered plugin,
    same as before that stage existed.

    on_algorithm_status(name, status): "running"|"completed"|"failed" — lets the caller
    persist live per-algorithm progress (Training Progress stage) independent of the
    coarser progress_cb percentage.
    """
    config = config or {}
    names = [n for n in CLASSIFICATION_PLUGIN_REGISTRY if not algorithms or n in algorithms]
    merged = {name: {**DEFAULT_CLASSIFICATION_HYPERPARAMETER_CONFIG.get(name, {}), **config.get(name, {})} for name in names}

    all_runs: list[ClassificationPluginRun] = []
    total = len(names) or 1
    for i, name in enumerate(names, start=1):
        plugin = CLASSIFICATION_PLUGIN_REGISTRY[name]
        if progress_cb:
            progress_cb((i - 1) / total * 90, f"Running {name}...")
        if on_algorithm_status:
            on_algorithm_status(name, "running")
        try:
            runs = plugin.run(X_train, y_train, X_test, merged[name])
        except Exception:
            runs = []
        all_runs.extend(runs)
        if on_algorithm_status:
            on_algorithm_status(name, "completed" if runs else "failed")
        if progress_cb:
            progress_cb(i / total * 90, f"{name} completed: {len(runs)} usable run(s).")

    return all_runs
