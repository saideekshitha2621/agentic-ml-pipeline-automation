from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier

from app.core.config import RANDOM_STATE
from app.plugins.classification_base import ClassificationPlugin
from app.plugins.registry import register_classification_plugin


@register_classification_plugin
class GradientBoostingClassifierPlugin(ClassificationPlugin):
    name = "gradient_boosting"

    def param_grid(self, config: dict) -> list[dict]:
        return [
            {"n_estimators": n, "learning_rate": lr}
            for n in config.get("n_estimators", [100])
            for lr in config.get("learning_rate", [0.1])
        ]

    def build_model(self, params: dict):
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=RANDOM_STATE,
        )
