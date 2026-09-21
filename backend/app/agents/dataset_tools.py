"""Read-only dataset investigation tools an LLM agent can call mid-reasoning (Phase 3).

Every tool is a pure function over the in-memory DataFrame — none writes anything, touches
the DB, or trains a persistent model — so handing them to a model is safe. Results are small
JSON-able dicts (the loop truncates anything larger).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.llm_service import Tool

_COLUMN_PARAM = {
    "type": "object",
    "properties": {"column": {"type": "string", "description": "Exact column name."}},
    "required": ["column"],
}


def build_tools(df: pd.DataFrame, target_column: str | None = None) -> list[Tool]:
    def _col(column: str) -> pd.Series:
        if column not in df.columns:
            raise ValueError(f"unknown column '{column}'; columns are {list(df.columns)[:40]}")
        return df[column]

    def column_stats(column: str) -> dict:
        s = _col(column)
        out = {
            "dtype": str(s.dtype), "n_unique": int(s.nunique(dropna=True)),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
        }
        if pd.api.types.is_numeric_dtype(s):
            d = s.describe()
            out.update({k: round(float(d[k]), 4) for k in ("min", "mean", "50%", "max", "std") if k in d})
            out["skew"] = round(float(s.skew()), 3) if s.notna().sum() > 2 else None
        return out

    def value_counts(column: str, top_n: int = 8) -> dict:
        return {str(k): int(v) for k, v in _col(column).value_counts(dropna=True).head(min(int(top_n), 20)).items()}

    def outlier_report() -> dict:
        report = {}
        for c in df.select_dtypes("number").columns:
            s = df[c].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            report[c] = round(float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()), 4) if iqr else 0.0
        return {"iqr_outlier_fraction_by_numeric_column": report}

    def leakage_probe(column: str) -> dict:
        """How well ONE column alone predicts the target (shallow tree, 3-fold CV)."""
        if not target_column or target_column not in df.columns:
            raise ValueError("no target column is set for this run")
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        data = df[[column, target_column]].dropna()
        if len(data) < 20:
            return {"single_feature_score": None, "note": "too few complete rows"}
        x = data[[column]]
        x = pd.get_dummies(x) if not pd.api.types.is_numeric_dtype(data[column]) else x
        y = data[target_column]
        regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
        model = DecisionTreeRegressor(max_depth=3) if regression else DecisionTreeClassifier(max_depth=3)
        score = float(np.mean(cross_val_score(model, x, y, cv=3, scoring="r2" if regression else "accuracy")))
        return {
            "single_feature_score": round(score, 3), "metric": "r2" if regression else "accuracy",
            "verdict": "possible_leakage" if score > 0.95 else "ok",
        }

    return [
        Tool("column_stats", "Dtype, cardinality, missing % and (numeric) distribution summary for one column.", _COLUMN_PARAM, column_stats),
        Tool("value_counts", "Most frequent values of one column.",
             {"type": "object", "properties": {"column": {"type": "string"}, "top_n": {"type": "integer"}}, "required": ["column"]},
             value_counts),
        Tool("outlier_report", "Share of IQR outliers in every numeric column.", {"type": "object", "properties": {}}, outlier_report),
        Tool("leakage_probe", "Score of a tiny model using ONLY this column to predict the target — near 1.0 signals leakage.",
             _COLUMN_PARAM, leakage_probe),
    ]
