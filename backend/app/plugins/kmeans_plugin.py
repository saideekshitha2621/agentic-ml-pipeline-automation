from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from app.core.config import RANDOM_STATE
from app.plugins.base import ClusteringPlugin
from app.plugins.registry import register_plugin


@register_plugin
class KMeansPlugin(ClusteringPlugin):
    name = "kmeans"

    def param_grid(self, config: dict) -> list[dict]:
        lo, hi = config.get("k_range", [2, 8])
        return [{"n_clusters": k} for k in range(lo, hi + 1)]

    def fit_predict(self, X: np.ndarray, params: dict) -> np.ndarray:
        labels, _extra = self.fit(X, params)
        return labels

    def fit(self, X: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        if params["n_clusters"] >= X.shape[0]:
            raise ValueError("k must be < n_samples")
        model = KMeans(n_clusters=params["n_clusters"], n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        # inertia_ (within-cluster sum of squares) is what a classic elbow curve plots vs K
        return labels, {"inertia": round(float(model.inertia_), 4)}
