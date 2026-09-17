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
from sklearn.inspection import permutation_importance

from app.services import llm_service


def global_feature_importance(model, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> list[dict]:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        try:
            result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=1)
            importances = result.importances_mean
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
