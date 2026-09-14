"""Runs K-Means, Hierarchical, DBSCAN and GMM across hyperparameter grids."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


@dataclass
class ClusterRun:
    algorithm: str
    params: dict[str, Any]
    labels: np.ndarray
    n_clusters: int
    n_noise: int = 0
    model: Any = None


@dataclass
class HyperparameterGrid:
    kmeans_k: list[int] = field(default_factory=lambda: list(range(2, 9)))
    hierarchical_k: list[int] = field(default_factory=lambda: list(range(2, 9)))
    hierarchical_linkages: list[str] = field(
        default_factory=lambda: ["ward", "complete", "average", "single"]
    )
    dbscan_eps: list[float] | None = None  # auto-derived from data if None
    dbscan_min_samples: list[int] = field(default_factory=lambda: [3, 5, 10])
    gmm_components: list[int] = field(default_factory=lambda: list(range(2, 9)))
    gmm_covariance_types: list[str] = field(default_factory=lambda: ["full"])


def _suggest_dbscan_eps(X: np.ndarray, k: int = 5) -> list[float]:
    """Use a k-distance graph to suggest a small spread of eps values around the elbow."""
    n = min(k, max(2, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=n)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    elbow = float(np.percentile(k_distances, 90))
    if elbow <= 0:
        elbow = float(np.mean(k_distances)) or 0.5
    return [round(elbow * m, 4) for m in (0.5, 0.75, 1.0, 1.25, 1.5)]


def run_kmeans(X: np.ndarray, grid: HyperparameterGrid, random_state: int = 42) -> list[ClusterRun]:
    runs = []
    for k in grid.kmeans_k:
        if k >= X.shape[0]:
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
        runs.append(
            ClusterRun("K-Means", {"n_clusters": k}, labels, len(set(labels)), model=model)
        )
    return runs


def run_hierarchical(X: np.ndarray, grid: HyperparameterGrid) -> list[ClusterRun]:
    runs = []
    for linkage in grid.hierarchical_linkages:
        for k in grid.hierarchical_k:
            if k >= X.shape[0]:
                continue
            kwargs = {"n_clusters": k, "linkage": linkage}
            if linkage != "ward":
                kwargs["metric"] = "euclidean"
            try:
                model = AgglomerativeClustering(**kwargs)
                labels = model.fit_predict(X)
            except Exception:
                continue
            runs.append(
                ClusterRun(
                    "Hierarchical",
                    {"n_clusters": k, "linkage": linkage},
                    labels,
                    len(set(labels)),
                    model=model,
                )
            )
    return runs


def run_dbscan(X: np.ndarray, grid: HyperparameterGrid) -> list[ClusterRun]:
    eps_values = grid.dbscan_eps or _suggest_dbscan_eps(X)
    runs = []
    for eps in eps_values:
        for min_samples in grid.dbscan_min_samples:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            n_clusters = len(set(labels) - {-1})
            n_noise = int(np.sum(labels == -1))
            if n_clusters < 2:
                continue  # not a usable clustering
            runs.append(
                ClusterRun(
                    "DBSCAN",
                    {"eps": eps, "min_samples": min_samples},
                    labels,
                    n_clusters,
                    n_noise=n_noise,
                    model=model,
                )
            )
    return runs


def run_gmm(X: np.ndarray, grid: HyperparameterGrid, random_state: int = 42) -> list[ClusterRun]:
    runs = []
    for cov_type in grid.gmm_covariance_types:
        for n in grid.gmm_components:
            if n >= X.shape[0]:
                continue
            model = GaussianMixture(n_components=n, covariance_type=cov_type, random_state=random_state)
            labels = model.fit_predict(X)
            runs.append(
                ClusterRun(
                    "GMM",
                    {"n_components": n, "covariance_type": cov_type},
                    labels,
                    len(set(labels)),
                    model=model,
                )
            )
    return runs


def run_all_algorithms(
    X: np.ndarray, grid: HyperparameterGrid | None = None, random_state: int = 42
) -> list[ClusterRun]:
    grid = grid or HyperparameterGrid()
    runs: list[ClusterRun] = []
    runs += run_kmeans(X, grid, random_state)
    runs += run_hierarchical(X, grid)
    runs += run_dbscan(X, grid)
    runs += run_gmm(X, grid, random_state)
    return runs
