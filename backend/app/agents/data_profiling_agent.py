"""Data Profiling Agent.

Wraps the existing `profiling_service.build_profile()` unchanged and adds the one thing
it doesn't do today: semantic role tagging per column (id-like / datetime / numeric /
categorical / free-text). Downstream agents (Problem Detection, Preprocessing) reason over
these roles instead of raw dtypes.
"""
from __future__ import annotations

import pandas as pd

ID_LIKE_UNIQUENESS_THRESHOLD = 0.95
CATEGORICAL_CARDINALITY_CAP_RATIO = 0.05
CATEGORICAL_CARDINALITY_CAP_ABS = 20


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(25)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def _column_role(series: pd.Series, n_rows: int) -> dict:
    nunique = int(series.nunique(dropna=True))
    uniqueness_ratio = nunique / n_rows if n_rows else 0.0

    if uniqueness_ratio > ID_LIKE_UNIQUENESS_THRESHOLD:
        role = "id_like"
    elif _looks_like_datetime(series):
        role = "datetime"
    elif pd.api.types.is_numeric_dtype(series):
        role = "numeric"
    elif nunique <= max(CATEGORICAL_CARDINALITY_CAP_ABS, int(n_rows * CATEGORICAL_CARDINALITY_CAP_RATIO)):
        role = "categorical"
    else:
        role = "free_text"

    return {"role": role, "dtype": str(series.dtype), "n_unique": nunique}


def analyze(df: pd.DataFrame, base_profile: dict) -> dict:
    """Returns base_profile extended with `column_roles` and a `dataset_understanding`
    block (req 1: rows/columns/numeric/categorical/missing/duplicates/potential targets/
    quality summary) for the Dataset Understanding stage card."""
    n_rows = len(df)
    column_roles = {col: _column_role(df[col], n_rows) for col in df.columns}

    numerical_columns = [c for c, m in column_roles.items() if m["role"] == "numeric"]
    categorical_columns = [c for c, m in column_roles.items() if m["role"] == "categorical"]
    n_duplicates = int(df.duplicated().sum())
    missing_by_column = {
        col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()
    }
    potential_targets = [
        c for c, m in column_roles.items() if m["role"] in ("numeric", "categorical")
    ]

    quality_score = base_profile.get("data_quality_score", 0)
    if quality_score >= 85:
        quality_summary = f"High data quality ({quality_score}/100) — this dataset is ready to model with minimal cleanup."
    elif quality_score >= 60:
        quality_summary = f"Moderate data quality ({quality_score}/100) — expect some cleaning work before modeling."
    else:
        quality_summary = f"Low data quality ({quality_score}/100) — significant cleaning is likely needed before modeling."

    dataset_understanding = {
        "n_rows": n_rows,
        "n_columns": int(df.shape[1]),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "n_missing_values": int(sum(missing_by_column.values())),
        "missing_by_column": missing_by_column,
        "n_duplicate_rows": n_duplicates,
        "potential_target_columns": potential_targets,
        "data_quality_summary": quality_summary,
    }
    return {**base_profile, "column_roles": column_roles, "dataset_understanding": dataset_understanding}


def summarize(profile: dict) -> str:
    roles = profile.get("column_roles", {})
    counts: dict[str, int] = {}
    for meta in roles.values():
        counts[meta["role"]] = counts.get(meta["role"], 0) + 1
    parts = [f"{v} {k.replace('_', ' ')}" for k, v in counts.items()]
    du = profile.get("dataset_understanding", {})
    return (
        f"{profile.get('n_rows', 0)} rows, {profile.get('n_columns', 0)} columns "
        f"(data quality score {profile.get('data_quality_score', 0)}/100), "
        f"{du.get('n_duplicate_rows', 0)} duplicate row(s), {du.get('n_missing_values', 0)} missing value(s) total. "
        f"Column roles: {', '.join(parts)}."
    )
