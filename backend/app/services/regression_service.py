"""Runs every registered regression plugin over a config-driven hyperparameter grid.

Mirrors classification_service.run_all() exactly — same merge-config-over-defaults, same
per-algorithm progress callback and shortlist filter — against the regression plugin
registry instead.
"""
from __future__ import annotations

import numpy as np

from app.core.config import DEFAULT_REGRESSION_HYPERPARAMETER_CONFIG
from app.plugins import REGRESSION_PLUGIN_REGISTRY
from app.plugins.regression_base import RegressionPluginRun


def run_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: dict | None = None,
    progress_cb=None,
    algorithms: list[str] | None = None,
    on_algorithm_status=None,
) -> list[RegressionPluginRun]:
    """config: {"linear_regression": {...}, "random_forest": {...}, ...} overriding
    DEFAULT_REGRESSION_HYPERPARAMETER_CONFIG per algorithm.

    algorithms: restrict execution to this subset of registered plugin names (the
    Algorithm Recommendation HITL shortlist) — None/empty runs every registered plugin.
    on_algorithm_status(name, status): "running"|"completed"|"failed", for live progress.
    """
    config = config or {}
    names = [n for n in REGRESSION_PLUGIN_REGISTRY if not algorithms or n in algorithms]
    merged = {name: {**DEFAULT_REGRESSION_HYPERPARAMETER_CONFIG.get(name, {}), **config.get(name, {})} for name in names}

    all_runs: list[RegressionPluginRun] = []
    total = len(names) or 1
    for i, name in enumerate(names, start=1):
        plugin = REGRESSION_PLUGIN_REGISTRY[name]
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
