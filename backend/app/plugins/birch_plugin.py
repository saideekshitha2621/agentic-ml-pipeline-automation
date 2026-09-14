from __future__ import annotations

import numpy as np
from sklearn.cluster import Birch

from app.plugins.base import ClusteringPlugin
from app.plugins.registry import register_plugin


@register_plugin
class BirchPlugin(ClusteringPlugin):
    name = "birch"

    def param_grid(self, config: dict) -> list[dict]:
        lo, hi = config.get("k_range", [2, 8])
        threshold = config.get("threshold", 0.5)
        thresholds = threshold if isinstance(threshold, list) else [threshold]
        return [{"n_clusters": k, "threshold": t} for t in thresholds for k in range(lo, hi + 1)]

    def fit_predict(self, X: np.ndarray, params: dict) -> np.ndarray:
        if params["n_clusters"] >= X.shape[0]:
            raise ValueError("k must be < n_samples")
        model = Birch(n_clusters=params["n_clusters"], threshold=params["threshold"])
        return model.fit_predict(X)
