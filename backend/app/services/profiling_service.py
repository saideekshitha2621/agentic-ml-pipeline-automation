"""Data Quality screen: missing values, duplicates, outliers, category standardization."""
from __future__ import annotations

import pandas as pd


def missing_value_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 2),
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )
    return sorted(rows, key=lambda r: r["missing_pct"], reverse=True)


def detect_outliers(df: pd.DataFrame) -> list[dict]:
    """IQR-based outlier detection for numeric columns."""
    rows = []
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        if n_outliers == 0:
            continue
        rows.append(
            {
                "column": col,
                "method": "IQR",
                "n_outliers": n_outliers,
                "pct_outliers": round(n_outliers / len(series) * 100, 2),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }
        )
    return sorted(rows, key=lambda r: r["pct_outliers"], reverse=True)


def category_standardization_suggestions(df: pd.DataFrame) -> list[dict]:
    """Flags categorical columns with casing/whitespace inconsistencies that likely
    represent the same category (e.g. "North", "north ", "NORTH")."""
    suggestions = []
    for col in df.select_dtypes(exclude="number").columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        normalized_groups: dict[str, set[str]] = {}
        for val in series.unique():
            key = val.strip().lower()
            normalized_groups.setdefault(key, set()).add(val)

        inconsistent = {k: v for k, v in normalized_groups.items() if len(v) > 1}
        if inconsistent:
            examples = sorted({v for group in inconsistent.values() for v in group})[:8]
            suggestions.append(
                {
                    "column": col,
                    "issue": f"{len(inconsistent)} value(s) have inconsistent casing/whitespace",
                    "examples": examples,
                    "suggestion": "Normalize to a single canonical casing (e.g. strip + title-case) "
                    "before encoding.",
                }
            )

        high_cardinality_threshold = max(20, int(len(series) * 0.5))
        if series.nunique() > high_cardinality_threshold:
            suggestions.append(
                {
                    "column": col,
                    "issue": f"high cardinality ({series.nunique()} unique values)",
                    "examples": sorted(series.unique())[:8],
                    "suggestion": "Consider grouping rare categories into 'Other' or excluding "
                    "this column from clustering.",
                }
            )
    return suggestions


def data_quality_score(df: pd.DataFrame, n_duplicates: int, outliers: list[dict]) -> float:
    """0-100 composite score: penalizes missing data, duplicates, and outlier-heavy columns."""
    if df.empty:
        return 0.0
    missing_penalty = float(df.isna().mean().mean()) * 100
    dup_penalty = (n_duplicates / len(df)) * 100 if len(df) else 0
    outlier_penalty = sum(o["pct_outliers"] for o in outliers) / max(len(df.columns), 1)

    score = 100 - (missing_penalty * 0.5 + dup_penalty * 0.3 + outlier_penalty * 0.2)
    return round(max(0.0, min(100.0, score)), 2)


def build_profile(df: pd.DataFrame) -> dict:
    n_duplicates = int(df.duplicated().sum())
    outliers = detect_outliers(df)
    missing = missing_value_rows(df)
    suggestions = category_standardization_suggestions(df)
    score = data_quality_score(df, n_duplicates, outliers)

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_duplicates": n_duplicates,
        "data_quality_score": score,
        "missing_values": missing,
        "outliers": outliers,
        "category_suggestions": suggestions,
    }
