from __future__ import annotations

import pandas as pd

from app.agents import (
    algorithm_shortlist_agent,
    cleaning_plan_agent,
    data_profiling_agent,
    data_validation_agent,
    split_agent,
    transformation_agent,
)


def _df():
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(50)],
            "age": [20 + (i % 40) for i in range(50)],
            "income": [30000.0 + i * 137.5 for i in range(50)],
            "region": (["north", "south", "east", "west"] * 13)[:50],
            "churn": (["yes", "no"] * 25),
            "constant_col": [1] * 50,
        }
    )


def test_dataset_understanding_reports_expected_fields():
    df = _df()
    profile = data_profiling_agent.analyze(df, {"data_quality_score": 90})
    du = profile["dataset_understanding"]
    assert du["n_rows"] == 50
    assert "age" in du["numerical_columns"]
    assert "region" in du["categorical_columns"]
    assert du["n_duplicate_rows"] == 0


def test_data_validation_flags_constant_column():
    df = _df()
    profile = data_profiling_agent.analyze(df, {"data_quality_score": 90})
    result = data_validation_agent.validate(df, profile, target_column="churn")
    constant_check = next(c for c in result["checks"] if c["name"] == "Constant columns")
    assert constant_check["status"] != "ok"
    assert "constant_col" in constant_check["affected_columns"]


def test_cleaning_plan_drops_id_like_and_constant_columns():
    df = _df()
    profile = data_profiling_agent.analyze(df, {"data_quality_score": 90})
    validation = data_validation_agent.validate(df, profile, target_column="churn")
    plan = cleaning_plan_agent.propose(df, validation)
    fields = cleaning_plan_agent.to_preprocessing_plan_fields(plan)
    assert "customer_id" in fields["dropped_columns"]
    assert "constant_col" in fields["dropped_columns"]
    assert "age" in fields["numerical_columns"]
    assert "income" in fields["numerical_columns"]  # float column not misclassified as id-like


def test_cleaning_plan_recommendations_include_missing_stats():
    df = _df()
    df.loc[0:5, "age"] = None
    profile = data_profiling_agent.analyze(df, {"data_quality_score": 90})
    validation = data_validation_agent.validate(df, profile, target_column="churn")
    plan = cleaning_plan_agent.propose(df, validation)
    age_rec = next(r for r in plan["recommendations"] if r["column"] == "age")
    assert age_rec["missing_count"] == 6
    assert age_rec["missing_pct"] > 0
    assert age_rec["action"] in ("mean", "median")


def test_cleaning_plan_recommends_dropping_fully_missing_column():
    df = _df()
    df["fully_missing"] = None
    profile = data_profiling_agent.analyze(df, {"data_quality_score": 90})
    validation = data_validation_agent.validate(df, profile, target_column="churn")
    plan = cleaning_plan_agent.propose(df, validation)
    rec = next(r for r in plan["recommendations"] if r["column"] == "fully_missing")
    assert rec["action"] == "drop_column"
    assert rec["no_information"] is True
    assert set(rec["options"]) == {"drop_column", "fill_zero", "fill_custom", "business_rule"}


def test_transformation_agent_proposes_scaling_and_encoding():
    df = _df()
    plan = cleaning_plan_agent.to_preprocessing_plan_fields(
        cleaning_plan_agent.propose(df, data_validation_agent.validate(df, data_profiling_agent.analyze(df, {}), "churn"))
    )
    result = transformation_agent.propose(df, plan)
    assert result["scaling_method"] in ("standard", "robust")
    assert result["encoding_method"] == "one_hot"


def test_split_agent_recommends_ratio_for_classification():
    df = _df()
    result = split_agent.recommend(df, "classification", target_column="churn")
    assert result["applicable"] is True
    assert 0 < result["test_size"] < 1


def test_split_agent_not_applicable_for_clustering():
    df = _df()
    result = split_agent.recommend(df, "clustering")
    assert result["applicable"] is False


def test_algorithm_shortlist_recommends_subset():
    result = algorithm_shortlist_agent.recommend({"n_rows": 50}, "classification")
    assert len(result["selected_algorithms"]) > 0
    assert all(isinstance(a, str) for a in result["selected_algorithms"])
