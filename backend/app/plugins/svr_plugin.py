from __future__ import annotations

from sklearn.svm import SVR

from app.plugins.regression_base import RegressionPlugin
from app.plugins.registry import register_regression_plugin


@register_regression_plugin
class SVRPlugin(RegressionPlugin):
    name = "svr"

    def param_grid(self, config: dict) -> list[dict]:
        return [{"C": c} for c in config.get("C", [1.0])]

    def build_model(self, params: dict):
        return SVR(kernel="linear", C=params["C"])
