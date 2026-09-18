from __future__ import annotations

from sklearn.linear_model import Ridge

from app.plugins.regression_base import RegressionPlugin
from app.plugins.registry import register_regression_plugin


@register_regression_plugin
class LinearRegressionPlugin(RegressionPlugin):
    # Ridge rather than plain LinearRegression — a small L2 penalty keeps coefficients
    # stable when features are correlated, at negligible cost when they aren't.
    name = "linear_regression"

    def param_grid(self, config: dict) -> list[dict]:
        return [{"alpha": a} for a in config.get("alpha", [1.0])]

    def build_model(self, params: dict):
        return Ridge(alpha=params["alpha"])
