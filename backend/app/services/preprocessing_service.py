"""Preprocessing Review screen: detect defaults, apply a (possibly human-edited) plan.

Mirrors `ml_automation.preprocessing.DataPreprocessor` (median/mode imputation, dedup,
scaling) but adds the scaling-method choice (standard/minmax/robust/none) the platform's
Preprocessing Review screen exposes, which the CLI/Streamlit version didn't need.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from app.core.config import RANDOM_STATE
from app.services import feature_engineering_service

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class MissingValueValidationError(ValueError):
    """Raised instead of silently filling with 0 when one or more columns still contain
    missing values after applying every configured imputation action — e.g. a column left
    as "keep" despite having missing values, or an explicit mean/median/mode that itself
    resolved to NaN (every remaining value in that column was already missing). Data
    quality and explainability take priority over forcing the pipeline to complete."""

    def __init__(self, columns_with_missing: dict[str, int]):
        self.columns_with_missing = columns_with_missing
        detail = ", ".join(f"{col} ({n} missing)" for col, n in columns_with_missing.items())
        super().__init__(
            f"Missing values remain after applying the cleaning plan: {detail}. Choose an "
            "explicit strategy (mean/median/mode/drop column/drop rows/custom value) for "
            "these column(s) at the Cleaning Plan stage before training."
        )


def _numeric_fill_value(series: pd.Series, action: str, custom_value) -> float | None:
    if action == "median":
        value = series.median()
    elif action == "mean":
        value = series.mean()
    elif action == "fill_zero":
        return 0.0
    elif action in ("fill_custom", "business_rule"):
        return float(custom_value) if custom_value is not None else None
    else:  # "keep" or an unrecognized action — no fill; surfaced by the NaN check below
        return None
    return None if pd.isna(value) else float(value)


def _categorical_fill_value(series: pd.Series, action: str, custom_value) -> str | None:
    if action == "mode":
        modes = series.mode(dropna=True)
        return str(modes.iloc[0]) if not modes.empty else None
    if action == "fill_zero":
        return "0"
    if action in ("fill_custom", "business_rule"):
        return str(custom_value) if custom_value is not None else None
    return None


def _assert_no_missing_values(df: pd.DataFrame) -> None:
    remaining = {col: int(n) for col, n in df.isna().sum().items() if n > 0}
    if remaining:
        raise MissingValueValidationError(remaining)


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
    """Returns (cleaned_df, encoded_df, report).

    `plan["column_actions"]`, if present (populated by the Cleaning Plan HITL stage — see
    `agents/cleaning_plan_agent.py`), gives each retained column its own imputation action
    (mean/median/mode/drop_rows/fill_zero/fill_custom/business_rule/keep) instead of the
    single dataset-wide `numerical_impute_strategy`/`categorical_impute_strategy` the old
    manual flow's `detect_default_plan()` still uses when this key is absent/empty.
    """
    report: dict = {"initial_shape": list(df.shape)}
    column_actions: dict = plan.get("column_actions") or {}

    working = df.drop(columns=plan["dropped_columns"], errors="ignore").copy()
    keep_cols = plan["numerical_columns"] + plan["categorical_columns"]
    working = working[[c for c in keep_cols if c in working.columns]]

    drop_rows_cols = [c for c, a in column_actions.items() if a.get("action") == "drop_rows" and c in working.columns]
    if drop_rows_cols:
        before = len(working)
        working = working.dropna(subset=drop_rows_cols).reset_index(drop=True)
        report["rows_dropped_for_missing"] = {"columns": drop_rows_cols, "n_rows_dropped": before - len(working)}

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
        col_action = column_actions.get(col, {})
        action = col_action.get("action", plan["numerical_impute_strategy"])
        fill_value = _numeric_fill_value(working[col], action, col_action.get("custom_value"))
        if fill_value is not None:
            working[col] = working[col].fillna(fill_value)
            imputation[col] = {"strategy": action, "fill_value": fill_value, "n_missing": n_missing}

    for col in plan["categorical_columns"]:
        if col not in working.columns:
            continue
        n_missing = int(working[col].isna().sum())
        if n_missing == 0:
            continue
        col_action = column_actions.get(col, {})
        action = col_action.get("action", plan["categorical_impute_strategy"])
        fill_value = _categorical_fill_value(working[col], action, col_action.get("custom_value"))
        if fill_value is not None:
            working[col] = working[col].fillna(fill_value)
            imputation[col] = {"strategy": action, "fill_value": fill_value, "n_missing": n_missing}

    report["imputation"] = imputation
    _assert_no_missing_values(working)
    cleaned_df = working.copy()

    encoded_df = working.copy()
    present_cats = [c for c in plan["categorical_columns"] if c in encoded_df.columns]
    if present_cats:
        encoded_df = pd.get_dummies(encoded_df, columns=present_cats, drop_first=False)

    scaling_method = plan["scaling_method"]
    if scaling_method != "none" and not encoded_df.empty and len(encoded_df.columns):
        scaler_cls = SCALERS.get(scaling_method, StandardScaler)
        scaler = scaler_cls()
        cols = list(encoded_df.columns)
        encoded_df = pd.DataFrame(scaler.fit_transform(encoded_df[cols]), columns=cols, index=encoded_df.index)
        report["scaling"] = {"method": scaling_method, "n_features": len(cols)}
    else:
        report["scaling"] = {"method": "none"}

    report["final_shape"] = list(encoded_df.shape)
    return cleaned_df, encoded_df, report


def split_target(
    df: pd.DataFrame,
    target_column: str,
    plan: dict,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Leakage-safe counterpart to apply_plan() for classification/regression.

    apply_plan() fits imputation/scaling on the whole dataframe — correct for clustering,
    which has no held-out set to leak into. Once a target column exists, fitting those same
    statistics on rows the model will later be tested against is leakage: this splits FIRST,
    fits every statistic (impute values, scaler) on the training rows only, then applies
    those same fitted values to the test rows untouched.

    Deliberately duplicates apply_plan()'s impute/encode/scale steps rather than
    refactoring it to share code, so the clustering path — which must keep fitting on the
    whole dataset — is not put at risk by a change made for the supervised path.

    Returns (X_train, X_test, y_train, y_test, report).
    """
    # A missing target can't be filled or guessed without inventing the answer — rows
    # without a label are dropped before anything else (train_test_split's stratify=y would
    # otherwise raise "Input contains NaN" on the label itself, a failure the Cleaning Plan
    # stage — which only ever touches feature columns — can never catch).
    n_missing_target = int(df[target_column].isna().sum())
    if n_missing_target:
        df = df.dropna(subset=[target_column]).reset_index(drop=True)

    # Stateless feature engineering (log1p on skewed columns) — safe before the split since
    # it has no fitted state to leak; mirrored in champion_service for the refit + prediction.
    df = feature_engineering_service.apply(df, plan.get("feature_transforms"))

    column_actions: dict = plan.get("column_actions") or {}
    drop_rows_cols = [c for c, a in column_actions.items() if a.get("action") == "drop_rows" and c in df.columns and c != target_column]
    if drop_rows_cols:
        df = df.dropna(subset=drop_rows_cols).reset_index(drop=True)

    y = df[target_column]
    feature_df = df.drop(columns=[target_column])
    train_raw, test_raw, y_train, y_test = train_test_split(
        feature_df, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )

    def _select(working: pd.DataFrame) -> pd.DataFrame:
        working = working.drop(columns=plan["dropped_columns"], errors="ignore").copy()
        keep_cols = plan["numerical_columns"] + plan["categorical_columns"]
        return working[[c for c in keep_cols if c in working.columns]]

    train_sel, test_sel = _select(train_raw), _select(test_raw)

    impute_values: dict = {}
    for col in plan["numerical_columns"]:
        if col not in train_sel.columns:
            continue
        col_action = column_actions.get(col, {})
        action = col_action.get("action", plan["numerical_impute_strategy"])
        fill_value = _numeric_fill_value(train_sel[col], action, col_action.get("custom_value"))
        if fill_value is not None:
            impute_values[col] = fill_value
            train_sel[col] = train_sel[col].fillna(fill_value)
            if col in test_sel.columns:
                test_sel[col] = test_sel[col].fillna(fill_value)

    for col in plan["categorical_columns"]:
        if col not in train_sel.columns:
            continue
        col_action = column_actions.get(col, {})
        action = col_action.get("action", plan["categorical_impute_strategy"])
        fill_value = _categorical_fill_value(train_sel[col], action, col_action.get("custom_value"))
        if fill_value is not None:
            impute_values[col] = fill_value
            train_sel[col] = train_sel[col].fillna(fill_value)
            if col in test_sel.columns:
                test_sel[col] = test_sel[col].fillna(fill_value)

    _assert_no_missing_values(train_sel)
    _assert_no_missing_values(test_sel)

    present_cats = [c for c in plan["categorical_columns"] if c in train_sel.columns]
    train_enc = pd.get_dummies(train_sel, columns=present_cats, drop_first=False) if present_cats else train_sel.copy()
    test_enc = pd.get_dummies(test_sel, columns=present_cats, drop_first=False) if present_cats else test_sel.copy()
    # Categories that only appear in the test split become all-zero columns rather than new
    # ones (the model was never trained on them); columns missing from the test split are
    # added back as all-zero — one-hot shape always matches what the model was fit on.
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

    scaling_method = plan["scaling_method"]
    report = {
        "target_column": target_column,
        "rows_dropped_for_missing_target": n_missing_target,
        "train_shape": list(train_enc.shape),
        "test_shape": list(test_enc.shape),
        "imputation": impute_values,
        "scaling": {"method": scaling_method},
    }
    if scaling_method != "none":
        scaler_cls = SCALERS.get(scaling_method, StandardScaler)
        scaler = scaler_cls().fit(train_enc.values)
        cols = list(train_enc.columns)
        train_enc = pd.DataFrame(scaler.transform(train_enc[cols]), columns=cols, index=train_enc.index)
        test_enc = pd.DataFrame(scaler.transform(test_enc[cols]), columns=cols, index=test_enc.index)

    return (
        train_enc.reset_index(drop=True),
        test_enc.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        report,
    )
