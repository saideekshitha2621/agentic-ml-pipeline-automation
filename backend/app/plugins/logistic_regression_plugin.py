from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from app.core.config import RANDOM_STATE
from app.plugins.classification_base import ClassificationPlugin
from app.plugins.registry import register_classification_plugin


@register_classification_plugin
class LogisticRegressionPlugin(ClassificationPlugin):
    name = "logistic_regression"

    def param_grid(self, config: dict) -> list[dict]:
        return [{"C": c} for c in config.get("C", [1.0])]

    def build_model(self, params: dict):
        return LogisticRegression(C=params["C"], max_iter=1000, random_state=RANDOM_STATE)
