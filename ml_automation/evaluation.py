"""Evaluation metrics for clustering runs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .clustering import ClusterRun


def evaluate_run(X: np.ndarray, run: ClusterRun, sample_size: int | None = 5000) -> dict:
    """Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz for one clustering run.

    DBSCAN noise points (-1) are excluded from silhouette/CH/DB computation since those
    metrics require a real cluster assignment.
    """
    labels = run.labels
    mask = labels != -1
    X_eval, labels_eval = X[mask], labels[mask]

    n_labels = len(set(labels_eval))
    if n_labels < 2 or n_labels >= X_eval.shape[0]:
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "n_clusters_evaluated": n_labels,
            "n_samples_evaluated": int(X_eval.shape[0]),
        }

    if sample_size and X_eval.shape[0] > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_eval.shape[0], size=sample_size, replace=False)
        sil = silhouette_score(X_eval[idx], labels_eval[idx])
    else:
        sil = silhouette_score(X_eval, labels_eval)

    db = davies_bouldin_score(X_eval, labels_eval)
    ch = calinski_harabasz_score(X_eval, labels_eval)

    return {
        "silhouette_score": round(float(sil), 4),
        "davies_bouldin_score": round(float(db), 4),
        "calinski_harabasz_score": round(float(ch), 2),
        "n_clusters_evaluated": n_labels,
        "n_samples_evaluated": int(X_eval.shape[0]),
    }
