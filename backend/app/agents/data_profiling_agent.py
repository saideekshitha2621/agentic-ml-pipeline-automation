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
    """Returns base_profile extended with `column_roles: {column: {role, dtype, n_unique}}`."""
    n_rows = len(df)
    column_roles = {col: _column_role(df[col], n_rows) for col in df.columns}
    return {**base_profile, "column_roles": column_roles}


def summarize(profile: dict) -> str:
    roles = profile.get("column_roles", {})
    counts: dict[str, int] = {}
    for meta in roles.values():
        counts[meta["role"]] = counts.get(meta["role"], 0) + 1
    parts = [f"{v} {k.replace('_', ' ')}" for k, v in counts.items()]
    return (
        f"{profile.get('n_rows', 0)} rows, {profile.get('n_columns', 0)} columns "
        f"(data quality score {profile.get('data_quality_score', 0)}/100). "
        f"Column roles: {', '.join(parts)}."
    )
