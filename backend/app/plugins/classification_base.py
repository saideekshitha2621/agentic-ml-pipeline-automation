"""Contract every classification algorithm plugin implements.

Mirrors plugins/base.py's ClusteringPlugin shape (self-registering, one file per
algorithm, a param grid expanded from job config, failures on one param combination
skipped rather than fatal) — but fit/predict needs a train/test split and a target
vector, which fit_predict(X) has no room for, so this is a sibling contract rather than
a shared base class with clustering.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClassificationPluginRun:
    algorithm: str
    params: dict[str, Any]
    model: Any  # fitted estimator — kept so recommendation/report stages can reuse it
    y_pred: np.ndarray
    y_proba: np.ndarray | None = None  # shape (n_samples, n_classes) when the estimator supports it
    extra: dict[str, Any] = field(default_factory=dict)


class ClassificationPlugin(ABC):
    name: str

    @abstractmethod
    def param_grid(self, config: dict) -> list[dict[str, Any]]:
        """Expand this algorithm's hyperparameter grid from a job's config-driven overrides."""

    @abstractmethod
    def build_model(self, params: dict[str, Any]) -> Any:
        """Return an unfitted sklearn-compatible estimator for these params."""

    def run(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, config: dict
    ) -> list[ClassificationPluginRun]:
        runs = []
        for params in self.param_grid(config):
            try:
                model = self.build_model(params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            except Exception:
                continue
            runs.append(ClassificationPluginRun(self.name, params, model, y_pred, y_proba))
        return runs
