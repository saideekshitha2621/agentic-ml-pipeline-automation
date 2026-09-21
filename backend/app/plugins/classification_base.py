"""Contract every classification algorithm plugin implements.

Mirrors plugins/base.py's ClusteringPlugin shape (self-registering, one file per
algorithm, a param grid expanded from job config, failures on one param combination
skipped rather than fatal) — but fit/predict needs a train/test split and a target
vector, which fit_predict(X) has no room for, so this is a sibling contract rather than
a shared base class with clustering.
"""
from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Class-imbalance handling (Phase 4): "balanced" re-weights classes inversely to their
# frequency. It is a *context* value rather than a parameter so it reaches every place a
# classifier is built — training, hyperparameter search, the final champion refit — without
# threading a new argument through each of them. ContextVars are per-thread/task safe.
_CLASS_WEIGHT: contextvars.ContextVar[str | None] = contextvars.ContextVar("class_weight", default=None)


@contextmanager
def class_weight_context(class_weight: str | None):
    token = _CLASS_WEIGHT.set(class_weight)
    try:
        yield
    finally:
        _CLASS_WEIGHT.reset(token)


def apply_class_weight(model: Any) -> bool:
    """Sets class_weight on estimators that support it (logistic regression, random forest,
    SVM); estimators without the parameter (gradient boosting, kNN) are left unchanged.
    Returns whether it was applied."""
    weight = _CLASS_WEIGHT.get()
    if weight and hasattr(model, "get_params") and "class_weight" in model.get_params():
        model.set_params(class_weight=weight)
        return True
    return False


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

    def build_configured(self, params: dict[str, Any]) -> Any:
        """`build_model` plus any active context options (currently class_weight)."""
        model = self.build_model(params)
        apply_class_weight(model)
        return model

    def run(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, config: dict
    ) -> list[ClassificationPluginRun]:
        runs = []
        for params in self.param_grid(config):
            try:
                model = self.build_configured(params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            except Exception:
                continue
            runs.append(ClassificationPluginRun(self.name, params, model, y_pred, y_proba))
        return runs
