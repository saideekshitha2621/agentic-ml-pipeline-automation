"""Persists the human-approved champion model as a self-contained, reloadable prediction
pipeline — refit on the *full* dataset (not just the training split) since this is now the
production artifact, not an evaluation run. Powers the Prediction Playground (req 12) and
Explainability (req 13), which both need a live model plus the exact preprocessing it
expects, not just the metrics that were logged during model_runs.

Deliberately reimplements the impute/encode/scale steps rather than reusing
`preprocessing_service.apply_plan`/`split_target` — this needs a *reusable, single-row*
transform at prediction time, which those two (whole-dataframe) functions don't offer.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from app.core.config import MODELS_DIR
from app.plugins.registry import get_classification_plugin, get_regression_plugin
from app.services import explainability_service, feature_engineering_service, preprocessing_service

_PLUGIN_LOOKUP = {"classification": get_classification_plugin, "regression": get_regression_plugin}

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


@dataclass
class ChampionPipeline:
    """Joblib-serializable: raw feature dict -> prediction. Everything it needs (impute
    values, one-hot columns, fitted scaler, fitted estimator) is fit once at build time and
    baked in, so predict() has no dependency on the original dataframe."""

    numerical_columns: list[str]
    categorical_columns: list[str]
    impute_values: dict
    encoded_columns: list[str]
    scaler: object | None
    estimator: object
    target_classes: list
    feature_transforms: list = field(default_factory=list)  # Phase 4; older pickles lack it

    def transform(self, raw: dict) -> np.ndarray:
        row = {c: raw.get(c) for c in self.numerical_columns + self.categorical_columns}
        df = pd.DataFrame([row])
        for col in self.numerical_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # same stateless transform training used, applied to the *raw* entered value first
        df = feature_engineering_service.apply(df, getattr(self, "feature_transforms", []))
        for col in self.numerical_columns:
            if pd.isna(df[col]).any():
                df[col] = df[col].fillna(self.impute_values.get(col, 0))
        for col in self.categorical_columns:
            if df[col].isna().any() or df[col].iloc[0] in (None, ""):
                df[col] = df[col].fillna(self.impute_values.get(col, ""))
        encoded = pd.get_dummies(df, columns=self.categorical_columns) if self.categorical_columns else df
        encoded = encoded.reindex(columns=self.encoded_columns, fill_value=0)
        X = encoded.values.astype(float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def predict(self, raw: dict) -> dict:
        X = self.transform(raw)
        prediction = self.estimator.predict(X)[0]
        result = {"prediction": prediction.item() if hasattr(prediction, "item") else prediction}
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)[0]
            result["probabilities"] = {
                str(cls): round(float(p), 4) for cls, p in zip(self.target_classes, proba)
            }
            result["confidence"] = round(float(max(proba)), 4)
        return result


def build_and_persist(
    df: pd.DataFrame,
    plan: dict,
    target_column: str,
    algorithm: str,
    params: dict,
    pipeline_run_id: str,
    problem_type: str = "classification",
) -> dict:
    """Refits `algorithm`(`params`) on the full dataset and persists a ChampionPipeline.
    Returns {model_path, feature_schema, feature_importance}."""
    plugin = _PLUGIN_LOOKUP[problem_type](algorithm)
    column_actions: dict = plan.get("column_actions") or {}

    # Same as preprocessing_service.split_target: a missing target can't be filled or
    # guessed, so rows without a label are excluded before refitting the champion model.
    if df[target_column].isna().any():
        df = df.dropna(subset=[target_column]).reset_index(drop=True)

    drop_rows_cols = [c for c, a in column_actions.items() if a.get("action") == "drop_rows" and c in df.columns and c != target_column]
    if drop_rows_cols:
        df = df.dropna(subset=drop_rows_cols).reset_index(drop=True)

    feature_transforms = plan.get("feature_transforms") or []
    raw_df = df  # the Prediction Playground schema must describe RAW values, not transformed ones
    df = feature_engineering_service.apply(df, feature_transforms)

    y = df[target_column]
    working = df.drop(columns=[target_column]).drop(columns=plan["dropped_columns"], errors="ignore").copy()
    numerical_columns = [c for c in plan["numerical_columns"] if c in working.columns]
    categorical_columns = [c for c in plan["categorical_columns"] if c in working.columns]
    working = working[numerical_columns + categorical_columns]

    # impute_values does double duty: it's both "what filled the training gaps" and "the
    # sensible default the Prediction Playground shows/falls back to for this feature" — so
    # every retained column gets a representative value here even when it had no missing
    # values to fill (action "keep"), rather than only the columns that needed a fill.
    impute_values: dict = {}
    for col in numerical_columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")
        col_action = column_actions.get(col, {})
        action = col_action.get("action", plan["numerical_impute_strategy"])
        n_missing = int(working[col].isna().sum())
        representative = working[col].median()
        impute_values[col] = 0.0 if pd.isna(representative) else float(representative)
        if n_missing:
            fill_value = preprocessing_service._numeric_fill_value(working[col], action, col_action.get("custom_value"))
            if fill_value is not None:
                impute_values[col] = fill_value
                working[col] = working[col].fillna(fill_value)
    for col in categorical_columns:
        col_action = column_actions.get(col, {})
        action = col_action.get("action", plan["categorical_impute_strategy"])
        n_missing = int(working[col].isna().sum())
        modes = working[col].mode(dropna=True)
        impute_values[col] = str(modes.iloc[0]) if not modes.empty else ""
        if n_missing:
            fill_value = preprocessing_service._categorical_fill_value(working[col], action, col_action.get("custom_value"))
            if fill_value is not None:
                impute_values[col] = fill_value
                working[col] = working[col].fillna(fill_value)

    preprocessing_service._assert_no_missing_values(working)

    encoded = pd.get_dummies(working, columns=categorical_columns) if categorical_columns else working
    encoded_columns = list(encoded.columns)

    scaler = None
    X = encoded.values.astype(float)
    if plan["scaling_method"] != "none":
        scaler = SCALERS.get(plan["scaling_method"], StandardScaler)()
        X = scaler.fit_transform(X)

    estimator = getattr(plugin, "build_configured", plugin.build_model)(params)
    estimator.fit(X, y.values)

    # Meaningless (and wastefully large) for a continuous regression target — a classifier's
    # small set of class labels drives the Prediction Playground's probability breakdown,
    # which doesn't apply to a numeric prediction.
    target_classes = sorted(y.dropna().unique().tolist(), key=str) if problem_type == "classification" else []

    pipeline = ChampionPipeline(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        impute_values=impute_values,
        encoded_columns=encoded_columns,
        scaler=scaler,
        estimator=estimator,
        target_classes=target_classes,
        feature_transforms=feature_transforms,
    )

    model_path = MODELS_DIR / f"{pipeline_run_id}-{uuid.uuid4().hex[:8]}.joblib"
    joblib.dump(pipeline, model_path)

    feature_importance = explainability_service.global_feature_importance(estimator, X, y.values, encoded_columns)

    feature_schema = {}
    for col in numerical_columns:
        series = pd.to_numeric(raw_df[col], errors="coerce").dropna()
        feature_schema[col] = {
            "type": "numeric",
            "min": float(series.min()) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
            "default": feature_engineering_service.inverse_value(impute_values.get(col), col, feature_transforms),
        }
    for col in categorical_columns:
        feature_schema[col] = {
            "type": "categorical",
            "options": sorted(raw_df[col].dropna().astype(str).unique().tolist()),
            "default": impute_values.get(col),
        }

    return {
        "model_path": str(model_path),
        "feature_schema": feature_schema,
        "feature_importance": feature_importance,
        "target_classes": [str(c) for c in target_classes],
    }


def load(model_path: str) -> ChampionPipeline:
    return joblib.load(model_path)
