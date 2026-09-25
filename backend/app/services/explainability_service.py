"""Explainability service — classification only (clustering already has
`explanation_service.py`/`ClusterInterpretation`).

Global feature importance: native `feature_importances_`/`coef_` when the estimator
exposes them, else a `permutation_importance` fallback so every estimator type is covered.

Per-prediction explanation: SHAP when the estimator type is supported, else a heuristic
"how far this value sits from the training mean, weighted by that feature's global
importance" fallback — kept deliberately simple so a missing/incompatible SHAP explainer
never breaks the Prediction Playground.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.services import llm_service


PERMUTATION_MAX_ROWS = 300
PERMUTATION_MAX_FEATURES = 30
PERMUTATION_REPEATS = 2


def _bounded_permutation_importance(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Permutation importance whose cost does not grow with the data.

    sklearn's `permutation_importance` re-scores the model for *every* column x repeat over *every*
    row. A one-hot encoded high-cardinality column (e.g. a date with ~1,250 values) makes that
    thousands of full-dataset predictions — hours for KNN — and stalls the finalize stage. Here rows
    are subsampled, and only the columns most correlated with the target (a cheap filter) are
    permuted; the rest score 0."""
    rng = np.random.default_rng(42)
    if len(X) > PERMUTATION_MAX_ROWS:
        idx = rng.choice(len(X), PERMUTATION_MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]

    n_features = X.shape[1]
    if n_features > PERMUTATION_MAX_FEATURES:
        y_num = y if np.issubdtype(np.asarray(y).dtype, np.number) else LabelEncoder().fit_transform(y)
        Xc = X - X.mean(axis=0)
        yc = np.asarray(y_num, dtype=float) - float(np.mean(y_num))
        denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
        denom = np.where(denom == 0, 1.0, denom)
        corr = np.abs(np.nan_to_num((Xc * yc[:, None]).sum(axis=0) / denom))
        candidates = np.argsort(corr)[::-1][:PERMUTATION_MAX_FEATURES]
    else:
        candidates = np.arange(n_features)

    baseline = model.score(X, y)
    importances = np.zeros(n_features)
    work = X.copy()
    for col in candidates:
        original = work[:, col].copy()
        drops = []
        for _ in range(PERMUTATION_REPEATS):
            work[:, col] = rng.permutation(original)
            drops.append(baseline - model.score(work, y))
        work[:, col] = original
        importances[col] = float(np.mean(drops))
    return importances


def global_feature_importance(model, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> list[dict]:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        try:
            importances = _bounded_permutation_importance(model, X, y)
        except Exception:
            importances = np.zeros(len(feature_names))

    total = float(np.sum(np.abs(importances))) or 1.0
    ranked = sorted(
        ({"feature": f, "importance": float(v), "importance_pct": round(float(abs(v)) / total * 100, 1)}
         for f, v in zip(feature_names, importances)),
        key=lambda r: -abs(r["importance"]),
    )
    return ranked


def explain_prediction(model, X_train: np.ndarray, x_row: np.ndarray, feature_names: list[str], top_n: int = 3) -> dict:
    contributions = None
    try:
        import shap

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_train)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, min(50, len(X_train))))
        shap_values = explainer.shap_values(x_row.reshape(1, -1))
        values = shap_values[0] if isinstance(shap_values, list) else shap_values
        contributions = np.asarray(values).reshape(-1)[: len(feature_names)]
    except Exception:
        pass

    if contributions is None:
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0) + 1e-9
        contributions = (x_row - means) / stds

    ranked = sorted(
        ({"feature": f, "contribution": float(c)} for f, c in zip(feature_names, contributions)),
        key=lambda r: -abs(r["contribution"]),
    )[:top_n]

    template = "Top contributing factors: " + "; ".join(
        f"{r['feature']} ({'pushed toward this prediction' if r['contribution'] > 0 else 'pushed against this prediction'})"
        for r in ranked
    )
    narrative = llm_service.explain(
        "prediction_explanation",
        {"top_features": ranked},
        fallback=template,
    )
    return {"top_features": ranked, "narrative": narrative}
