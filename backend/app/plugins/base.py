"""Contract every clustering algorithm plugin implements.

Adding a new algorithm to the platform means writing one file that implements this
ABC and self-registers via `@register_plugin` — no router or service code changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PluginRun:
    algorithm: str
    params: dict[str, Any]
    labels: np.ndarray
    n_clusters: int
    n_noise: int = 0
    extra: dict[str, Any] = None  # type: ignore[assignment]  # e.g. {"inertia": ...} for KMeans

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class ClusteringPlugin(ABC):
    name: str

    @abstractmethod
    def param_grid(self, config: dict) -> list[dict[str, Any]]:
        """Expand this algorithm's hyperparameter grid from a job's config-driven overrides."""

    @abstractmethod
    def fit_predict(self, X: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Fit the model and return integer cluster labels (-1 = noise, where applicable)."""

    def fit(self, X: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Like fit_predict, but can also return extra diagnostics (e.g. KMeans inertia for
        an elbow curve). Default just wraps fit_predict with no extras; override when a
        plugin has cheap extra info available from the fitted model."""
        return self.fit_predict(X, params), {}

    def run(self, X: np.ndarray, config: dict) -> list[PluginRun]:
        runs = []
        for params in self.param_grid(config):
            try:
                labels, extra = self.fit(X, params)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s failed for params %s: %s: %s", self.name, params, type(exc).__name__, exc)
                continue
            n_clusters = len(set(labels) - {-1})
            if n_clusters < 2:
                logger.info("%s skipped for params %s: only %d cluster(s) found", self.name, params, n_clusters)
                continue
            n_noise = int(np.sum(labels == -1))
            runs.append(PluginRun(self.name, params, labels, n_clusters, n_noise, extra))
        return runs
