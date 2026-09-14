from __future__ import annotations

import numpy as np
from sklearn.cluster import OPTICS

from app.plugins.base import ClusteringPlugin
from app.plugins.registry import register_plugin


@register_plugin
class OPTICSPlugin(ClusteringPlugin):
    name = "optics"

    def param_grid(self, config: dict) -> list[dict]:
        min_samples_values = config.get("min_samples", [3, 5, 10])
        xi_values = config.get("xi", [0.05])
        return [{"min_samples": m, "xi": x} for m in min_samples_values for x in xi_values]

    def fit_predict(self, X: np.ndarray, params: dict) -> np.ndarray:
        model = OPTICS(min_samples=params["min_samples"], xi=params["xi"])
        return model.fit_predict(X)
