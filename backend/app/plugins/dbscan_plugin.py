from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from app.plugins.base import ClusteringPlugin
from app.plugins.registry import register_plugin


def _suggest_eps(X: np.ndarray, k: int = 5) -> list[float]:
    n = min(k, max(2, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=n).fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    elbow = float(np.percentile(k_distances, 90)) or float(np.mean(k_distances)) or 0.5
    return [round(elbow * m, 4) for m in (0.5, 0.75, 1.0, 1.25, 1.5)]


@register_plugin
class DBSCANPlugin(ClusteringPlugin):
    name = "dbscan"

    def param_grid(self, config: dict) -> list[dict]:
        # `run()` below always resolves eps_values (auto-suggested from the data if the
        # job config didn't override it) before this is called.
        eps_values = config["eps_values"]
        min_samples_values = config.get("min_samples", [3, 5, 10])
        return [{"min_samples": m, "eps": e} for m in min_samples_values for e in eps_values]

    def fit_predict(self, X: np.ndarray, params: dict) -> np.ndarray:
        return DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit_predict(X)

    def run(self, X: np.ndarray, config: dict):
        eps_values = config.get("eps_values") or _suggest_eps(X)
        min_samples_values = config.get("min_samples", [3, 5, 10])
        resolved_config = {"eps_values": eps_values, "min_samples": min_samples_values}
        return super().run(X, resolved_config)
