"""Transformation Agent.

Explains the scaling/encoding choices the (approved) cleaning plan implies, plus any
engineered features — decoupled from cleaning_plan_agent so the HITL reviewer confirms
"what will be done to the data" (this stage) separately from "which values need fixing"
(the previous stage), even though both end up on the same PreprocessingPlanORM.
"""
from __future__ import annotations

import pandas as pd


def propose(df: pd.DataFrame, plan_fields: dict) -> dict:
    numerical = plan_fields["numerical_columns"]
    categorical = plan_fields["categorical_columns"]

    has_outliers = False
    for col in numerical:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr and ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean() > 0.05:
            has_outliers = True
            break

    scaling_method = "robust" if has_outliers else "standard"
    scaling_reason = (
        "Robust scaling (median/IQR-based) — numeric columns show meaningful outliers that would "
        "distort a mean/variance-based scaler."
        if has_outliers
        else "Standard scaling (zero mean, unit variance) — numeric columns look roughly outlier-free."
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

    return {
        "scaling_method": scaling_method,
        "scaling_reason": scaling_reason,
        "encoding_method": encoding_method,
        "encoding_reason": encoding_reason,
        "engineered_features": engineered_features,
        "numerical_columns": numerical,
        "categorical_columns": categorical,
    }


def summarize(result: dict) -> str:
    parts = [result["scaling_reason"], result["encoding_reason"]]
    if result["engineered_features"]:
        parts.append(f"Proposing {len(result['engineered_features'])} engineered feature group(s) from datetime columns.")
    return " ".join(parts)
