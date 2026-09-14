"""Preprocessing Review screen: detect defaults, apply a (possibly human-edited) plan.

Mirrors `ml_automation.preprocessing.DataPreprocessor` (median/mode imputation, dedup,
scaling) but adds the scaling-method choice (standard/minmax/robust/none) the platform's
Preprocessing Review screen exposes, which the CLI/Streamlit version didn't need.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def detect_default_plan(
    df: pd.DataFrame, id_columns: list[str] | None = None, id_like_threshold: float = 0.95
) -> dict:
    """id_like_threshold: columns where more than this fraction of values are unique
    (e.g. customer_id, order_id) are auto-excluded by default rather than treated as
    categorical — one-hot encoding an identifier column silently explodes dimensionality
    with near-meaningless features. The human can always move it back on the
    Preprocessing Review screen."""
    id_columns = id_columns or []
    numerical, categorical, likely_id = [], [], []
    n_rows = len(df) or 1
    for col in df.columns:
        if col in id_columns:
            continue
        if df[col].nunique(dropna=True) / n_rows > id_like_threshold:
            likely_id.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical.append(col)
        else:
            categorical.append(col)
    return {
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "dropped_columns": id_columns + likely_id,
        "numerical_impute_strategy": "median",
        "categorical_impute_strategy": "mode",
        "scaling_method": "standard",
        "drop_duplicates": True,
        "pca_enabled": False,
        "pca_variance_target": 0.95,
    }


def apply_plan(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (cleaned_df, encoded_df, report)."""
    report: dict = {"initial_shape": list(df.shape)}

    working = df.drop(columns=plan["dropped_columns"], errors="ignore").copy()
    keep_cols = plan["numerical_columns"] + plan["categorical_columns"]
    working = working[[c for c in keep_cols if c in working.columns]]

    n_dupes = int(working.duplicated().sum())
    if plan["drop_duplicates"] and n_dupes:
        working = working.drop_duplicates().reset_index(drop=True)
    report["duplicates_found"] = n_dupes
    report["duplicates_removed"] = n_dupes if plan["drop_duplicates"] else 0

    imputation: dict = {}
    for col in plan["numerical_columns"]:
        if col not in working.columns:
            continue
        n_missing = int(working[col].isna().sum())
        if n_missing == 0:
            continue
        strategy = plan["numerical_impute_strategy"]
        fill_value = working[col].median() if strategy == "median" else working[col].mean()
        working[col] = working[col].fillna(fill_value)
        imputation[col] = {"strategy": strategy, "fill_value": float(fill_value), "n_missing": n_missing}

    for col in plan["categorical_columns"]:
        if col not in working.columns:
            continue
        n_missing = int(working[col].isna().sum())
        if n_missing == 0:
            continue
        mode_vals = working[col].mode(dropna=True)
        fill_value = mode_vals.iloc[0] if not mode_vals.empty else "missing"
        working[col] = working[col].fillna(fill_value)
        imputation[col] = {"strategy": "mode", "fill_value": str(fill_value), "n_missing": n_missing}

    report["imputation"] = imputation
    cleaned_df = working.copy()

    encoded_df = working.copy()
    present_cats = [c for c in plan["categorical_columns"] if c in encoded_df.columns]
    if present_cats:
        encoded_df = pd.get_dummies(encoded_df, columns=present_cats, drop_first=False)

    scaling_method = plan["scaling_method"]
    if scaling_method != "none":
        scaler_cls = SCALERS.get(scaling_method, StandardScaler)
        scaler = scaler_cls()
        cols = list(encoded_df.columns)
        encoded_df = pd.DataFrame(scaler.fit_transform(encoded_df[cols]), columns=cols, index=encoded_df.index)
        report["scaling"] = {"method": scaling_method, "n_features": len(cols)}
    else:
        report["scaling"] = {"method": "none"}

    report["final_shape"] = list(encoded_df.shape)
    return cleaned_df, encoded_df, report
