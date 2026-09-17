from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from app.plugins.classification_base import ClassificationPlugin
from app.plugins.registry import register_classification_plugin


@register_classification_plugin
class KNNClassifierPlugin(ClassificationPlugin):
    name = "knn"

    def param_grid(self, config: dict) -> list[dict]:
        return [{"n_neighbors": k} for k in config.get("n_neighbors", [5])]

    def build_model(self, params: dict):
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"])
