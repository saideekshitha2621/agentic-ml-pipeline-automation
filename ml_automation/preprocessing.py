"""Data preprocessing with an optional human-in-the-loop review step."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessingPlan:
    """Decisions the pipeline made (or the human overrode) about how to preprocess data."""

    numerical_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    numerical_impute_strategy: str = "median"
    categorical_impute_strategy: str = "mode"
    drop_duplicates: bool = True
    scale_features: bool = True

    def to_dict(self) -> dict:
        return {
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "dropped_columns": self.dropped_columns,
            "numerical_impute_strategy": self.numerical_impute_strategy,
            "categorical_impute_strategy": self.categorical_impute_strategy,
            "drop_duplicates": self.drop_duplicates,
            "scale_features": self.scale_features,
        }


class DataPreprocessor:
    """Detects column types, imputes missing values, dedupes, and scales features.

    Usage:
        pre = DataPreprocessor(df)
        plan = pre.build_default_plan()   # inspect / let a human edit this
        processed_df, encoded_df, report = pre.run(plan)
    """

    def __init__(self, df: pd.DataFrame, id_columns: Optional[list[str]] = None):
        self.raw_df = df.copy()
        self.id_columns = id_columns or []
        self.scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------ #
    # Step 1: detect column types and build a default plan
    # ------------------------------------------------------------------ #
    def detect_column_types(
        self, id_like_threshold: float = 0.95
    ) -> tuple[list[str], list[str], list[str]]:
        """Splits columns into (numerical, categorical, likely_id) based on dtype and
        uniqueness. A column where more than `id_like_threshold` of values are unique
        (e.g. a customer_id, order_id, or free-text name) behaves like an identifier,
        not a clustering feature — one-hot encoding it would silently explode
        dimensionality with near-meaningless columns. It's excluded by default but the
        human can always move it back via the Preprocessing Review screen."""
        df = self.raw_df.drop(columns=self.id_columns, errors="ignore")
        numerical, categorical, likely_id = [], [], []
        n_rows = len(df) or 1
        for col in df.columns:
            if df[col].nunique(dropna=True) / n_rows > id_like_threshold:
                likely_id.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Low-cardinality integer columns often behave like categories,
                # but we still default to numerical unless the human overrides it.
                numerical.append(col)
            else:
                categorical.append(col)
        return numerical, categorical, likely_id

    def build_default_plan(self) -> PreprocessingPlan:
        numerical, categorical, likely_id = self.detect_column_types()
        return PreprocessingPlan(
            numerical_columns=numerical,
            categorical_columns=categorical,
            dropped_columns=list(self.id_columns) + likely_id,
        )

    def missing_value_summary(self) -> pd.DataFrame:
        df = self.raw_df
        summary = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(t) for t in df.dtypes],
                "missing_count": df.isna().sum().values,
                "missing_pct": (df.isna().mean() * 100).round(2).values,
                "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
            }
        )
        return summary.sort_values("missing_pct", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Step 2: run preprocessing according to a (possibly human-edited) plan
    # ------------------------------------------------------------------ #
    def run(self, plan: PreprocessingPlan) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        report: dict = {"plan": plan.to_dict()}

        df = self.raw_df.drop(columns=plan.dropped_columns, errors="ignore").copy()
        keep_cols = plan.numerical_columns + plan.categorical_columns
        df = df[[c for c in keep_cols if c in df.columns]]

        report["initial_shape"] = list(self.raw_df.shape)
        report["shape_after_column_selection"] = list(df.shape)

        # --- duplicates ---
        n_dupes = int(df.duplicated().sum())
        if plan.drop_duplicates and n_dupes:
            df = df.drop_duplicates().reset_index(drop=True)
        report["duplicates_found"] = n_dupes
        report["duplicates_removed"] = n_dupes if plan.drop_duplicates else 0

        # --- missing values ---
        missing_before = df.isna().sum().to_dict()
        imputed_cols = {}
        for col in plan.numerical_columns:
            if col not in df.columns:
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            if plan.numerical_impute_strategy == "median":
                fill_value = df[col].median()
            elif plan.numerical_impute_strategy == "mean":
                fill_value = df[col].mean()
            else:
                fill_value = 0
            df[col] = df[col].fillna(fill_value)
            imputed_cols[col] = {
                "strategy": plan.numerical_impute_strategy,
                "fill_value": float(fill_value),
                "n_missing": n_missing,
            }

        for col in plan.categorical_columns:
            if col not in df.columns:
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            if plan.categorical_impute_strategy == "mode":
                mode_vals = df[col].mode(dropna=True)
                fill_value = mode_vals.iloc[0] if not mode_vals.empty else "missing"
            else:
                fill_value = "missing"
            df[col] = df[col].fillna(fill_value)
            imputed_cols[col] = {
                "strategy": plan.categorical_impute_strategy,
                "fill_value": str(fill_value),
                "n_missing": n_missing,
            }

        report["missing_before"] = {k: int(v) for k, v in missing_before.items()}
        report["imputation"] = imputed_cols

        cleaned_df = df.copy()

        # --- encode categoricals (one-hot) for modeling, keep cleaned_df human-readable ---
        encoded_df = df.copy()
        if plan.categorical_columns:
            present_cats = [c for c in plan.categorical_columns if c in encoded_df.columns]
            encoded_df = pd.get_dummies(encoded_df, columns=present_cats, drop_first=False)

        # --- scale numerical (+ one-hot) features ---
        if plan.scale_features:
            self.scaler = StandardScaler()
            feature_cols = [c for c in encoded_df.columns]
            scaled_values = self.scaler.fit_transform(encoded_df[feature_cols])
            encoded_df = pd.DataFrame(scaled_values, columns=feature_cols, index=encoded_df.index)
            report["scaling"] = {"method": "StandardScaler", "n_features": len(feature_cols)}
        else:
            report["scaling"] = {"method": "none"}

        report["final_shape"] = list(encoded_df.shape)
        report["numerical_columns"] = plan.numerical_columns
        report["categorical_columns"] = plan.categorical_columns

        return cleaned_df, encoded_df, report
