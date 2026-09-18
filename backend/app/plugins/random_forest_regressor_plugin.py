from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from app.core.config import RANDOM_STATE
from app.plugins.regression_base import RegressionPlugin
from app.plugins.registry import register_regression_plugin


@register_regression_plugin
class RandomForestRegressorPlugin(RegressionPlugin):
    name = "random_forest"

    def param_grid(self, config: dict) -> list[dict]:
        return [
            {"n_estimators": n, "max_depth": d}
            for n in config.get("n_estimators", [100])
            for d in config.get("max_depth", [None])
        ]

    def build_model(self, params: dict):
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=RANDOM_STATE,
        )
