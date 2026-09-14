from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering

from app.core.config import RANDOM_STATE
from app.plugins.base import ClusteringPlugin
from app.plugins.registry import register_plugin


@register_plugin
class SpectralPlugin(ClusteringPlugin):
    name = "spectral"

    def param_grid(self, config: dict) -> list[dict]:
        lo, hi = config.get("k_range", [2, 8])
        return [{"n_clusters": k} for k in range(lo, hi + 1)]

    def fit_predict(self, X: np.ndarray, params: dict) -> np.ndarray:
        if params["n_clusters"] >= X.shape[0]:
            raise ValueError("k must be < n_samples")
        model = SpectralClustering(
            n_clusters=params["n_clusters"],
            affinity="nearest_neighbors",
            random_state=RANDOM_STATE,
            assign_labels="kmeans",
        )
        return model.fit_predict(X)
