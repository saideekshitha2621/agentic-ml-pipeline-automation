"""HITL Approval screen: top-3 models with plain-language strengths/weaknesses."""
from __future__ import annotations

import numpy as np


def strengths_and_weaknesses(run: dict, problem_type: str = "clustering") -> tuple[list[str], list[str]]:
    if problem_type == "classification":
        return _classification_strengths_and_weaknesses(run)
    if problem_type == "regression":
        return _regression_strengths_and_weaknesses(run)
    return _clustering_strengths_and_weaknesses(run)


def _regression_strengths_and_weaknesses(run: dict) -> tuple[list[str], list[str]]:
    strengths, weaknesses = [], []
    metrics = run.get("metrics", run)
    r2 = metrics.get("r2")
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")

    if r2 is not None:
        if r2 > 0.8:
            strengths.append(f"Explains most of the variation in the target (R² {r2:.3f}).")
        elif r2 < 0.3:
            weaknesses.append(f"Explains little of the variation in the target (R² {r2:.3f}).")

    if rmse is not None:
        strengths.append(f"Typical prediction error (RMSE) of {rmse:.3f}.")
    if mae is not None:
        strengths.append(f"Average absolute error (MAE) of {mae:.3f}.")

    if not strengths:
        strengths.append("Completed successfully with valid predictions on the held-out test set.")
    return strengths, weaknesses


def prediction_stats_breakdown(y_pred: np.ndarray) -> dict[str, float]:
    """Regression's counterpart to `prediction_class_breakdown` — a continuous prediction
    array has no discrete "classes" to count, so summarize its distribution instead."""
    values = np.asarray(y_pred, dtype=float)
    return {
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
    }


def _classification_strengths_and_weaknesses(run: dict) -> tuple[list[str], list[str]]:
    strengths, weaknesses = [], []
    metrics = run.get("metrics", run)  # accept either a flat run_dict or {"metrics": {...}}
    acc = metrics.get("accuracy")
    f1 = metrics.get("f1_macro")
    roc = metrics.get("roc_auc")

    if f1 is not None:
        if f1 > 0.85:
            strengths.append(f"Strong, balanced performance across classes (F1 {f1:.3f}).")
        elif f1 < 0.5:
            weaknesses.append(f"Weak, imbalanced performance across classes (F1 {f1:.3f}).")

    if acc is not None:
        if acc > 0.8:
            strengths.append(f"{acc:.1%} accuracy on the held-out test set.")
        elif acc < 0.6:
            weaknesses.append(f"Only {acc:.1%} accuracy on the held-out test set.")

    if roc is not None:
        if roc > 0.85:
            strengths.append(f"Strong class separability (ROC-AUC {roc:.3f}).")
        elif roc < 0.6:
            weaknesses.append(f"Poor class separability (ROC-AUC {roc:.3f}).")
    else:
        weaknesses.append("ROC-AUC unavailable — the test split didn't cover every class.")

    if not strengths:
        strengths.append("Completed successfully with valid predictions on the held-out test set.")
    return strengths, weaknesses


def prediction_class_breakdown(y_pred: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(y_pred, return_counts=True)
    return {str(u): int(c) for u, c in zip(unique, counts)}


def _clustering_strengths_and_weaknesses(run: dict) -> tuple[list[str], list[str]]:
    strengths, weaknesses = [], []
    sil = run.get("silhouette_score")
    db = run.get("davies_bouldin_score")
    ch = run.get("calinski_harabasz_score")

    if sil is not None:
        if sil > 0.5:
            strengths.append(f"Strong cluster separation (silhouette {sil:.3f}).")
        elif sil < 0.25:
            weaknesses.append(f"Weak cluster separation (silhouette {sil:.3f}).")

    if db is not None:
        if db < 0.7:
            strengths.append(f"Compact, well-separated clusters (Davies-Bouldin {db:.3f}).")
        elif db > 1.2:
            weaknesses.append(f"Clusters overlap or are dispersed (Davies-Bouldin {db:.3f}).")

    if ch is not None:
        strengths.append(f"Calinski-Harabasz of {ch:,.1f} indicates dense, well-separated groupings.")

    noise_pct = run.get("noise_pct")
    if run.get("n_noise", 0) > 0:
        if noise_pct is not None and noise_pct >= 30:
            weaknesses.append(
                f"⚠️ {run['n_noise']} points ({noise_pct:.0f}% of the dataset) classified as noise / "
                "unassigned — the metrics above only reflect the remaining points, not the full data."
            )
        else:
            weaknesses.append(f"{run['n_noise']} points classified as noise / unassigned.")

    n_clusters = run.get("n_clusters", 0)
    if n_clusters <= 1:
        weaknesses.append("Produced too few clusters to be actionable.")
    elif n_clusters > 15:
        weaknesses.append(f"Produced {n_clusters} clusters, which may be too granular for business use.")
    else:
        strengths.append(f"Produced {n_clusters} clusters — a manageable number for segmentation.")

    if not strengths:
        strengths.append("Completed successfully with valid cluster assignments.")
    return strengths, weaknesses


def cluster_size_breakdown(labels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {("noise" if u == -1 else str(int(u))): int(c) for u, c in zip(unique, counts)}
