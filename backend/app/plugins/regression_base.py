"""Contract every regression algorithm plugin implements.

Mirrors `classification_base.py`'s shape exactly (self-registering, one file per
algorithm, a param grid expanded from job config, failures on one param combination
skipped rather than fatal) — the only difference is no `y_proba`/`predict_proba` concept,
since a regressor predicts a number, not a class probability.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RegressionPluginRun:
    algorithm: str
    params: dict[str, Any]
    model: Any  # fitted estimator — kept so recommendation/report stages can reuse it
    y_pred: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)


class RegressionPlugin(ABC):
    name: str

    @abstractmethod
    def param_grid(self, config: dict) -> list[dict[str, Any]]:
        """Expand this algorithm's hyperparameter grid from a job's config-driven overrides."""

    @abstractmethod
    def build_model(self, params: dict[str, Any]) -> Any:
        """Return an unfitted sklearn-compatible estimator for these params."""

    def run(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, config: dict) -> list[RegressionPluginRun]:
        runs = []
        for params in self.param_grid(config):
            try:
                model = self.build_model(params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            except Exception:
                continue
            runs.append(RegressionPluginRun(self.name, params, model, y_pred))
        return runs
