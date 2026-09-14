from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from app.plugins.base import ClusteringPlugin
from app.plugins.registry import register_plugin


@register_plugin
class HierarchicalPlugin(ClusteringPlugin):
    name = "hierarchical"

    def param_grid(self, config: dict) -> list[dict]:
        lo, hi = config.get("k_range", [2, 8])
        linkages = config.get("linkages", ["ward", "complete", "average", "single"])
        return [{"n_clusters": k, "linkage": link} for link in linkages for k in range(lo, hi + 1)]

    def fit_predict(self, X: np.ndarray, params: dict) -> np.ndarray:
        if params["n_clusters"] >= X.shape[0]:
            raise ValueError("k must be < n_samples")
        kwargs = {"n_clusters": params["n_clusters"], "linkage": params["linkage"]}
        if params["linkage"] != "ward":
            kwargs["metric"] = "euclidean"
        return AgglomerativeClustering(**kwargs).fit_predict(X)
