import numpy as np
import pandas as pd

from app.agents import problem_detection_agent
from app.services import clustering_service


def _blobs_df():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, (60, 3))
    b = rng.normal(5, 0.3, (60, 3))
    return pd.DataFrame(np.vstack([a, b]), columns=["f1", "f2", "f3"])


def _profile(df):
    return {"column_roles": {c: {"role": "numeric"} for c in df.columns}}


def test_unlabeled_dataset_proposes_clustering():
    df = _blobs_df()
    result = problem_detection_agent.detect(df, _profile(df))
    assert result["problem_type"] == "clustering"
    assert result["target_column"] is None
    assert result["requires_review"] is True


def test_outcome_named_column_still_supervised():
    df = _blobs_df()
    df["churn"] = ([0, 1] * 60)
    result = problem_detection_agent.detect(df, _profile(df))
    assert result["problem_type"] == "classification"
    assert result["target_column"] == "churn"


def test_run_all_reports_plugin_crash(monkeypatch):
    from app.plugins import PLUGIN_REGISTRY

    class Boom:
        def run(self, X, cfg):
            raise RuntimeError("boom")

    monkeypatch.setitem(PLUGIN_REGISTRY, "kmeans", Boom())
    msgs = []
    runs = clustering_service.run_all(
        _blobs_df().values, algorithms=["kmeans"], progress_cb=lambda p, m: msgs.append(m)
    )
    assert runs == []
    assert any("boom" in m for m in msgs)


def test_unsupervised_goal_skips_target_inference():
    df = _blobs_df()
    df["churn"] = [0, 1] * 60
    result = problem_detection_agent.detect(df, _profile(df), learning_type="unsupervised")
    assert result["problem_type"] == "clustering"
    assert result["target_column"] is None


def test_supervised_goal_never_falls_back_to_clustering_and_asks_for_target():
    df = _blobs_df()
    result = problem_detection_agent.detect(df, _profile(df), learning_type="supervised")
    assert result["problem_type"] != "clustering"
    assert result["requires_target_selection"] is True
    assert {c["column"] for c in result["target_candidates"]} == {"f1", "f2", "f3"}


def test_suggested_target_must_be_selected_by_the_user():
    df = pd.DataFrame({"YearsExperience": range(1, 51), "Salary": [30000 + i * 1500 for i in range(50)]})
    result = problem_detection_agent.detect(df, _profile(df))
    assert result["target_column"] == "Salary"
    assert result["requires_target_selection"] is True


def test_conventional_target_name_is_preselected():
    df = _blobs_df()
    df["label"] = [0, 1] * 60
    result = problem_detection_agent.detect(df, _profile(df))
    assert result["target_column"] == "label"
    assert result["requires_target_selection"] is False


def test_credit_card_style_columns_are_not_mistaken_for_targets():
    rng = np.random.default_rng(1)
    cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY"]
    df = pd.DataFrame(rng.gamma(2, 500, (500, len(cols))), columns=cols)
    result = problem_detection_agent.detect(df, _profile(df))
    assert result["problem_type"] == "clustering"
