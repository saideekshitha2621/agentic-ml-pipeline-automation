from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor

from app.plugins.regression_base import RegressionPlugin
from app.plugins.registry import register_regression_plugin


@register_regression_plugin
class KNNRegressorPlugin(RegressionPlugin):
    name = "knn"

    def param_grid(self, config: dict) -> list[dict]:
        return [{"n_neighbors": k} for k in config.get("n_neighbors", [5])]

    def build_model(self, params: dict):
        return KNeighborsRegressor(n_neighbors=params["n_neighbors"])
