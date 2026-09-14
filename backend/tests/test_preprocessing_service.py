import numpy as np
import pandas as pd

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
