"""Static plain-English explanations for evaluation metrics — no LLM needed, since the
meaning of "accuracy" or "silhouette score" doesn't change per run, only the number does."""
from __future__ import annotations

_GLOSSARY = {
    "accuracy": "The share of predictions that were exactly correct.",
    "f1_macro": "Balances precision and recall equally across every class — a single number that "
    "penalizes models which ignore rare classes.",
    "precision_macro": "Of everything the model labeled as a given class, the share that was actually right.",
    "recall_macro": "Of everything that truly belonged to a given class, the share the model found.",
    "roc_auc": "How well the model separates classes across every possible decision threshold — "
    "1.0 is perfect separation, 0.5 is a coin flip.",
    "silhouette_score": "How well-separated the clusters are, from -1 (overlapping) to 1 (cleanly separated).",
    "davies_bouldin_score": "Average similarity between each cluster and its most similar neighbor — lower is better.",
    "calinski_harabasz_score": "Ratio of between-cluster to within-cluster dispersion — higher means denser, "
    "better-separated clusters.",
    "rmse": "Root mean squared error — the typical size of a prediction's miss, in the same units as the target, "
    "with larger misses penalized more heavily.",
    "mae": "Mean absolute error — the average size of a prediction's miss, in the same units as the target.",
    "mape": "Mean absolute percentage error — the average miss size as a percentage of the actual value.",
    "r2": "R² (coefficient of determination) — the share of the target's variation the model explains, from 0 "
    "(none) to 1 (all of it).",
}


def explain(problem_type: str, metrics: dict) -> dict[str, str]:
    return {key: _GLOSSARY.get(key, "") for key in metrics if metrics.get(key) is not None}
