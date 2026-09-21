"""Transformation Agent.

Explains the scaling/encoding choices the (approved) cleaning plan implies, plus any
engineered features — decoupled from cleaning_plan_agent so the HITL reviewer confirms
"what will be done to the data" (this stage) separately from "which values need fixing"
(the previous stage), even though both end up on the same PreprocessingPlanORM.
"""
from __future__ import annotations

import pandas as pd

from app.agents import feedback_utils

SCALING_ALTERNATIVES = ["standard", "robust", "minmax"]
_SCALING_LABELS = {
    "standard": "Standard scaling (zero mean, unit variance)",
    "robust": "Robust scaling (median/IQR-based)",
    "minmax": "Min-max scaling (rescales each column to 0-1)",
}


def _rejected_scalings(feedback: list[dict] | None) -> set[str]:
    return {f["proposal"].get("scaling_method") for f in feedback or [] if f.get("proposal")} - {None}


def _named_in(text: str, candidates: list[str], excluded: set[str]) -> str | None:
    lowered = text.lower().replace("-", "").replace(" ", "")
    return next((n for n in candidates if n not in excluded and n in lowered), None)


def pick_alternative_scaling(previous: set[str], hint_text: str = "") -> str | None:
    """Reviewer-named method first (e.g. 'use minmax'), otherwise the next unrejected one."""
    return _named_in(hint_text, SCALING_ALTERNATIVES, previous) or next(
        (n for n in SCALING_ALTERNATIVES if n not in previous), None
    )


def propose(df: pd.DataFrame, plan_fields: dict, feedback: list[dict] | None = None) -> dict:
    numerical = plan_fields["numerical_columns"]
    categorical = plan_fields["categorical_columns"]

    has_outliers = False
    max_outlier_frac = 0.0
    for col in numerical:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr:
            frac = float(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean())
            max_outlier_frac = max(max_outlier_frac, frac)
            if frac > 0.05:
                has_outliers = True

    scaling_method = "robust" if has_outliers else "standard"
    scaling_reason = (
        "Robust scaling (median/IQR-based) — numeric columns show meaningful outliers that would "
        "distort a mean/variance-based scaler."
        if has_outliers
        else "Standard scaling (zero mean, unit variance) — numeric columns look roughly outlier-free."
    )
    # Phase 1 revision loop: never re-propose a rejected scaler; honour one the reviewer named.
    previous = _rejected_scalings(feedback)
    feedback_text = " ".join(feedback_utils.reasons(feedback))
    requested = _named_in(feedback_text, SCALING_ALTERNATIVES, previous)
    if requested:
        scaling_method, scaling_reason = requested, f"{_SCALING_LABELS[requested]} — as requested in your feedback."
    elif scaling_method in previous:
        alternative = pick_alternative_scaling(previous)
        if alternative:
            scaling_method = alternative
            scaling_reason = (
                f"{_SCALING_LABELS[alternative]} — chosen because you rejected {sorted(previous)} in the previous proposal."
            )

    max_cardinality = max((df[c].nunique(dropna=True) for c in categorical if c in df.columns), default=0)
    encoding_method = "one_hot"
    encoding_reason = (
        f"One-hot encoding for {len(categorical)} categorical column(s) — cardinality is low enough "
        f"(max {max_cardinality} distinct values) that this won't over-expand the feature space."
    )

    engineered_features = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            engineered_features.append(
                {"column": col, "new_features": [f"{col}_year", f"{col}_month", f"{col}_dayofweek"],
                 "reason": "Datetime column — calendar parts are more predictive than the raw timestamp."}
            )

    confidence, factors = estimate_confidence(len(df), max_outlier_frac, max_cardinality, revised=bool(feedback))
    return {
        "confidence": confidence,
        "confidence_factors": factors,
        "revision": len(feedback) if feedback else 0,
        "scaling_method": scaling_method,
        "scaling_reason": scaling_reason,
        "encoding_method": encoding_method,
        "encoding_reason": encoding_reason,
        "engineered_features": engineered_features,
        "numerical_columns": numerical,
        "categorical_columns": categorical,
    }


def estimate_confidence(
    n_rows: int, max_outlier_frac: float, max_cardinality: int, revised: bool = False
) -> tuple[float, list[str]]:
    score, factors = 0.9, []
    if 0.02 <= max_outlier_frac <= 0.10:
        score -= 0.1
        factors.append(f"-0.10: outlier share ({max_outlier_frac:.1%}) sits near the robust-vs-standard decision boundary")
    if max_cardinality > 15:
        score -= 0.15
        factors.append(f"-0.15: one-hot encoding a {max_cardinality}-category column greatly expands the feature space")
    if n_rows < 200:
        score -= 0.1
        factors.append(f"-0.10: only {n_rows} rows to judge distributions from")
    if revised:
        score -= 0.05
        factors.append("-0.05: re-proposed after a rejection")
    return round(max(score, 0.3), 2), factors


def summarize(result: dict) -> str:
    parts = [result["scaling_reason"], result["encoding_reason"]]
    if result["engineered_features"]:
        parts.append(f"Proposing {len(result['engineered_features'])} engineered feature group(s) from datetime columns.")
    return " ".join(parts)
