from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from app.core.config import RANDOM_STATE
from app.plugins.classification_base import ClassificationPlugin
from app.plugins.registry import register_classification_plugin


@register_classification_plugin
class RandomForestClassifierPlugin(ClassificationPlugin):
    name = "random_forest"

    def param_grid(self, config: dict) -> list[dict]:
        return [
            {"n_estimators": n, "max_depth": d}
            for n in config.get("n_estimators", [100])
            for d in config.get("max_depth", [None])
        ]

    def build_model(self, params: dict):
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=RANDOM_STATE,
        )
