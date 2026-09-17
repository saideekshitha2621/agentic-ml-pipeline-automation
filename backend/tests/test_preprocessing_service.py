import numpy as np
import pandas as pd
import pytest

from app.services import preprocessing_service


def _sample_df():
    return pd.DataFrame(
        {
            "age": [20, 30, np.nan, 40, 20],
            "income": [1000, 2000, 3000, np.nan, 1000],
            "region": ["N", "S", None, "E", "N"],
        }
    )


def test_detect_default_plan_splits_numeric_and_categorical():
    df = _sample_df()
    plan = preprocessing_service.detect_default_plan(df)
    assert set(plan["numerical_columns"]) == {"age", "income"}
    assert plan["categorical_columns"] == ["region"]


def test_detect_default_plan_auto_excludes_id_like_columns():
    """A column that's ~unique per row (like a customer_id) should default to excluded,
    not categorical — otherwise it gets one-hot encoded into as many columns as rows."""
    df = _sample_df()
    df["customer_id"] = [f"CUST{i:03d}" for i in range(len(df))]  # 100% unique
    plan = preprocessing_service.detect_default_plan(df)
    assert "customer_id" in plan["dropped_columns"]
    assert "customer_id" not in plan["categorical_columns"]
    assert "customer_id" not in plan["numerical_columns"]


def test_apply_plan_imputes_and_scales():
    df = _sample_df()
    plan = preprocessing_service.detect_default_plan(df)
    cleaned, encoded, report = preprocessing_service.apply_plan(df, plan)

    assert cleaned["age"].isna().sum() == 0
    assert cleaned["income"].isna().sum() == 0
    assert "age" in report["imputation"]
    assert "income" in report["imputation"]
    # encoded frame is one-hot + scaled -> no missing values, more columns than raw
    assert encoded.isna().sum().sum() == 0
    assert encoded.shape[1] >= 3


def test_apply_plan_drops_duplicates():
    df = _sample_df()
    plan = preprocessing_service.detect_default_plan(df)
    cleaned, _encoded, report = preprocessing_service.apply_plan(df, plan)
    assert report["duplicates_removed"] >= 0
    assert len(cleaned) <= len(df)


def test_apply_plan_scaling_none_keeps_original_scale():
    df = _sample_df()
    plan = preprocessing_service.detect_default_plan(df)
    plan["scaling_method"] = "none"
    _cleaned, encoded, report = preprocessing_service.apply_plan(df, plan)
    assert report["scaling"]["method"] == "none"
    assert encoded["age"].max() > 1  # unscaled values retain original magnitude


def test_apply_plan_all_missing_column_raises_validation_error_instead_of_zero_filling():
    """median()/mean() of an all-missing column is itself NaN — fillna(NaN) used to be a
    no-op, so training would fail downstream with 'Input contains NaN'. Silently defaulting
    that gap to 0 would hide a real data-quality problem, so this must instead raise a clear
    validation error identifying the column (data quality/explainability over silent success)."""
    df = pd.DataFrame({"age": [np.nan, np.nan, np.nan, np.nan], "region": ["N", "S", "E", "N"]})
    plan = preprocessing_service.detect_default_plan(df)
    with pytest.raises(preprocessing_service.MissingValueValidationError) as exc_info:
        preprocessing_service.apply_plan(df, plan)
    assert "age" in exc_info.value.columns_with_missing


def test_apply_plan_column_actions_drop_column_for_fully_missing_avoids_error():
    """The Cleaning Plan agent's default recommendation for a 100%-missing column is
    drop_column — once that's reflected in dropped_columns/column_actions, apply_plan
    should complete cleanly instead of raising."""
    df = pd.DataFrame({"age": [np.nan, np.nan, np.nan, np.nan], "region": ["N", "S", "E", "N"]})
    plan = preprocessing_service.detect_default_plan(df)
    plan["numerical_columns"] = []
    plan["dropped_columns"] = plan["dropped_columns"] + ["age"]
    cleaned, encoded, _report = preprocessing_service.apply_plan(df, plan)
    assert "age" not in cleaned.columns
    assert encoded.isna().sum().sum() == 0


def test_apply_plan_column_actions_fill_custom_resolves_missing_values():
    df = pd.DataFrame({"age": [np.nan, np.nan, np.nan, np.nan], "region": ["N", "S", "E", "N"]})
    plan = preprocessing_service.detect_default_plan(df)
    plan["column_actions"] = {"age": {"action": "fill_custom", "custom_value": "0"}}
    cleaned, encoded, report = preprocessing_service.apply_plan(df, plan)
    assert cleaned["age"].isna().sum() == 0
    assert report["imputation"]["age"]["strategy"] == "fill_custom"
    assert encoded.isna().sum().sum() == 0


def test_split_target_all_missing_training_column_raises_validation_error():
    df = pd.DataFrame(
        {
            "age": [np.nan] * 20,
            "income": list(range(20)),
            "region": (["N", "S"] * 10),
            "target": (["yes", "no"] * 10),
        }
    )
    plan = preprocessing_service.detect_default_plan(df)
    with pytest.raises(preprocessing_service.MissingValueValidationError) as exc_info:
        preprocessing_service.split_target(df, "target", plan, test_size=0.3, stratify=True)
    assert "age" in exc_info.value.columns_with_missing


def test_split_target_drops_rows_with_missing_target_instead_of_crashing():
    """A missing target value can't be filled or guessed — train_test_split(stratify=y)
    used to crash with a raw 'Input contains NaN' on the label itself, since the Cleaning
    Plan only ever touches feature columns, never the target."""
    df = pd.DataFrame(
        {
            "age": list(range(20)),
            "region": (["N", "S"] * 10),
            "target": (["yes", "no"] * 9) + [None, None],
        }
    )
    plan = preprocessing_service.detect_default_plan(df)
    X_train, X_test, y_train, y_test, report = preprocessing_service.split_target(
        df, "target", plan, test_size=0.3, stratify=True
    )
    assert report["rows_dropped_for_missing_target"] == 2
    assert len(X_train) + len(X_test) == 18
    assert y_train.isna().sum() == 0
    assert y_test.isna().sum() == 0


def test_split_target_column_actions_drop_rows_resolves_missing_values():
    df = pd.DataFrame(
        {
            "age": [np.nan if i < 5 else float(i) for i in range(20)],
            "income": list(range(20)),
            "region": (["N", "S"] * 10),
            "target": (["yes", "no"] * 10),
        }
    )
    plan = preprocessing_service.detect_default_plan(df)
    plan["column_actions"] = {"age": {"action": "drop_rows"}}
    X_train, X_test, _y_train, _y_test, _report = preprocessing_service.split_target(
        df, "target", plan, test_size=0.3, stratify=True
    )
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0
