from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from app.core.config import RANDOM_STATE
from app.plugins.base import ClusteringPlugin
from app.plugins.registry import register_plugin


@register_plugin
class GMMPlugin(ClusteringPlugin):
    name = "gmm"

    def param_grid(self, config: dict) -> list[dict]:
        lo, hi = config.get("component_range", [2, 8])
        cov_types = config.get("covariance_types", ["full"])
        return [{"n_components": n, "covariance_type": c} for c in cov_types for n in range(lo, hi + 1)]

    def fit_predict(self, X: np.ndarray, params: dict) -> np.ndarray:
        if params["n_components"] >= X.shape[0]:
            raise ValueError("n_components must be < n_samples")
        model = GaussianMixture(
            n_components=params["n_components"],
            covariance_type=params["covariance_type"],
            random_state=RANDOM_STATE,
        )
        return model.fit_predict(X)
