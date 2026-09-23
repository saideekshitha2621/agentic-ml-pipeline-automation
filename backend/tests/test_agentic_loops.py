"""Phases 1-3 of the agentic upgrade: revision loop, self-correction loop + critic, and the
tool-calling LLM agents (exercised with a fake provider session — no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.agents import (
    algorithm_selection_llm, cleaning_plan_agent, critic_agent, dataset_tools, orchestrator,
    problem_detection_agent, problem_detection_llm, quality_check_agent, split_agent,
    transformation_agent, transformation_llm,
)
from app.db.database import SessionLocal
from app.db.models import AgentDecision as AgentDecisionORM
from app.db.models import PipelineRun as PipelineRunORM
from app.services import agent_decision_service, llm_service

from .test_orchestrator_graph import _classification_df, dataset_factory  # noqa: F401  (fixture)

DISPATCH = {
    "problem_detection": orchestrator.advance_after_problem_approval,
    "data_validation": orchestrator.advance_after_validation_approval,
    "cleaning_plan": orchestrator.advance_after_cleaning_approval,
    "transformation": orchestrator.advance_after_transformation_approval,
    "train_test_split": orchestrator.advance_after_split_approval,
    "algorithm_recommendation": orchestrator.advance_after_algorithm_approval,
    "recommendation": orchestrator.finalize_after_recommendation_approval,
}


def _drive_until(db, run_id, stop_status):
    for _ in range(12):
        db.expire_all()
        run = db.get(PipelineRunORM, run_id)
        if run.status in (stop_status, "completed", "failed"):
            return run
        pending = (
            db.query(AgentDecisionORM).filter_by(pipeline_run_id=run_id, status="proposed")
            .order_by(AgentDecisionORM.created_at.desc()).first()
        )
        pending.status = "approved"
        db.commit()
        DISPATCH[pending.agent_name](run_id)
    raise AssertionError("did not reach " + stop_status)


def _fb(proposal, reason=""):
    return [{"proposal": proposal, "reason": reason}]


# ---------------- Phase 1: revision loop ------------------------------------------------

def test_rejecting_problem_detection_re_proposes_a_different_target(dataset_factory):  # noqa: F811
    run = dataset_factory(_classification_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(run.id)
        first = db.query(AgentDecisionORM).filter_by(pipeline_run_id=run.id, agent_name="problem_detection").one()
        assert first.decision_json["target_column"] == "churn"

        # what the review endpoint does on reject
        first.status, first.override_reason = "rejected", "churn is not what we want to predict"
        db.commit()
        assert agent_decision_service.can_revise(db, run.id, "problem_detection")
        orchestrator.revise_after_rejection(run.id)

        db.expire_all()
        rows = agent_decision_service.all_decisions(db, run.id, "problem_detection")
        assert [r.status for r in rows] == ["rejected", "proposed"]
        assert rows[1].decision_json["target_column"] != "churn"
        assert rows[1].decision_json["revision"] == 1
        assert db.get(PipelineRunORM, run.id).status == "awaiting_problem_approval"
        # profiling must NOT be duplicated by the loop
        assert len(agent_decision_service.all_decisions(db, run.id, "data_profiling")) == 1
    finally:
        db.close()


def test_revisions_are_bounded(dataset_factory):  # noqa: F811
    run = dataset_factory(_classification_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(run.id)
        for i in range(agent_decision_service.MAX_REVISIONS + 1):
            latest = agent_decision_service.latest_decision(db, run.id, "problem_detection")
            latest.status, latest.override_reason = "rejected", f"no {i}"
            db.commit()
            if i < agent_decision_service.MAX_REVISIONS:
                assert agent_decision_service.can_revise(db, run.id, "problem_detection")
                orchestrator.revise_after_rejection(run.id)
        assert not agent_decision_service.can_revise(db, run.id, "problem_detection")  # router would now fail the run
        assert not agent_decision_service.can_revise(db, run.id, "data_validation")  # never revisable
    finally:
        db.close()


def test_transformation_never_repeats_a_rejected_scaler():
    df = _classification_df()
    fields = {"numerical_columns": ["age", "income"], "categorical_columns": ["region"]}
    first = transformation_agent.propose(df, fields)
    second = transformation_agent.propose(df, fields, _fb(first))
    assert second["scaling_method"] != first["scaling_method"]
    third = transformation_agent.propose(df, fields, _fb(first, "please use minmax"))
    assert third["scaling_method"] == "minmax"


def test_split_moves_to_new_ratio_or_honours_request():
    df = _classification_df(400)
    first = split_agent.recommend(df, "classification", "churn")
    second = split_agent.recommend(df, "classification", "churn", _fb(first))
    assert second["test_size"] != first["test_size"]
    assert split_agent.recommend(df, "classification", "churn", _fb(first, "use a 30% test set"))["test_size"] == 0.3


def test_cleaning_revision_keeps_named_and_leakage_columns():
    df = pd.DataFrame({"a": range(100), "leaky": list(range(100)), "y": [0, 1] * 50})
    validation = {"checks": [{"name": "Data leakage risk", "status": "critical", "affected_columns": ["leaky"]}]}
    first = cleaning_plan_agent.propose(df, validation, "y")
    assert {r["column"]: r["action"] for r in first["recommendations"]}["leaky"] == "drop_column"
    revised = cleaning_plan_agent.propose(df, validation, "y", _fb(first, "keep leaky please"))
    assert {r["column"]: r["action"] for r in revised["recommendations"]}["leaky"] != "drop_column"
    assert revised["revision"] == 1 and revised["revision_notes"]


def test_problem_detection_excludes_rejected_targets():
    df = _classification_df()
    profile = {"column_roles": {}}
    first = problem_detection_agent.detect(df, profile)
    second = problem_detection_agent.detect(df, profile, feedback=_fb(first))
    assert second["target_column"] != first["target_column"]


def test_algorithm_feedback_is_applied_even_without_llm(monkeypatch):
    monkeypatch.setattr(llm_service, "is_configured", lambda: False)
    profile = {"n_rows": 60, "column_roles": {}, "dataset_understanding": {}}
    df = _classification_df()
    first = algorithm_selection_llm.recommend(profile, None, "classification", df, "churn")
    revised = algorithm_selection_llm.recommend(
        profile, None, "classification", df, "churn", _fb(first, "exclude knn, include svm")
    )
    assert "knn" not in revised["selected_algorithms"] and "svm" in revised["selected_algorithms"]
    broadened = algorithm_selection_llm.recommend(profile, None, "classification", df, "churn", _fb(first, "not good"))
    assert set(broadened["selected_algorithms"]) == set(algorithm_selection_llm._REGISTRY_BY_PROBLEM_TYPE["classification"])


def test_confidence_is_signal_derived_not_constant():
    clean = split_agent.estimate_confidence(_classification_df(1000), "classification", "churn")[0]
    tiny_imbalanced = pd.DataFrame({"x": range(50), "y": [1] * 47 + [0] * 3})
    risky = split_agent.estimate_confidence(tiny_imbalanced, "classification", "y")[0]
    assert risky < 0.75 <= clean  # crosses the auto-approval threshold


# ---------------- Phase 2: quality loop + critic ----------------------------------------

def test_quality_check_verdicts():
    algos = ["a", "b", "c"]
    ok = quality_check_agent.assess("classification", 0.8, selected_algorithms=algos, all_algorithms=algos, tried_remedies=[])
    assert ok["verdict"] == "proceed" and not ok["flags"]
    leak = quality_check_agent.assess("classification", 0.999, selected_algorithms=algos, all_algorithms=algos, tried_remedies=[])
    assert leak["verdict"] == "proceed" and "suspected_leakage" in leak["flags"]
    r1 = quality_check_agent.assess("regression", 0.1, selected_algorithms=["a"], all_algorithms=algos, tried_remedies=[])
    assert (r1["verdict"], r1["remedy"]) == ("retry", "broaden_algorithms")
    r2 = quality_check_agent.assess("regression", 0.1, selected_algorithms=algos, all_algorithms=algos, tried_remedies=["broaden_algorithms"])
    assert (r2["verdict"], r2["remedy"]) == ("retry", "alternate_scaling")
    done = quality_check_agent.assess(
        "regression", 0.1, selected_algorithms=algos, all_algorithms=algos,
        tried_remedies=["broaden_algorithms", "alternate_scaling"],
    )
    assert done["verdict"] == "proceed" and "low_performance_unresolved" in done["flags"]
    # nothing to broaden -> skips straight to the scaler remedy
    skip = quality_check_agent.assess("regression", 0.1, selected_algorithms=algos, all_algorithms=algos, tried_remedies=[])
    assert skip["remedy"] == "alternate_scaling"


def test_critic_flags_leakage_and_weak_results():
    quality = {"flags": ["suspected_leakage"], "key_metric": "f1_macro", "value": 0.999, "threshold": 0.6}
    review = critic_agent.review(
        problem_type="classification", recommendation={"confidence": "low"}, quality=quality,
        validation=None, cleaning=None, n_rows=50, retry_remedies=["broaden_algorithms"],
    )
    codes = {f["code"] for f in review["findings"]}
    assert {"suspected_leakage", "near_tie", "small_dataset", "self_corrected"} <= codes
    assert review["verdict"] == "serious_concerns"
    clean = critic_agent.review(
        problem_type="classification", recommendation={"confidence": "high"}, quality={"flags": []},
        validation=None, cleaning=None, n_rows=5000, retry_remedies=[],
    )
    assert clean["verdict"] == "no_concerns"


def test_self_correction_loop_retries_then_escalates_to_human(dataset_factory, monkeypatch):  # noqa: F811
    # An unreachable bar forces the loop: retry (broaden), retry (scaler), then escalate.
    monkeypatch.setitem(quality_check_agent.QUALITY_TARGETS, "classification", ("f1_macro", 2.0, None))
    run = dataset_factory(_classification_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(run.id)
        run = _drive_until(db, run.id, "awaiting_recommendation_approval")
        assert run.status == "awaiting_recommendation_approval", run.error_message

        checks = agent_decision_service.all_decisions(db, run.id, "quality_check")
        verdicts = [c.decision_json["verdict"] for c in checks]
        assert verdicts[-1] == "proceed" and verdicts.count("retry") == len(checks) - 1
        assert 1 <= verdicts.count("retry") + 1 <= quality_check_agent.MAX_AUTO_RETRIES + 1
        assert len(agent_decision_service.all_decisions(db, run.id, "model_selection")) == len(checks)

        critic = agent_decision_service.latest_decision(db, run.id, "critic")
        rec = agent_decision_service.latest_decision(db, run.id, "recommendation")
        assert critic is not None and rec.decision_json["critic"]["verdict"] == critic.decision_json["verdict"]

        # human still has the final say: approve -> champion + report
        rec.status = "approved"
        db.commit()
        orchestrator.finalize_after_recommendation_approval(run.id)
        db.expire_all()
        assert db.get(PipelineRunORM, run.id).status == "completed"
    finally:
        db.close()


# ---------------- Phase 3: tool-calling loop ---------------------------------------------

class _FakeSession:
    """Scripted provider: asks for tools, then answers."""
    script: list = []
    last_results: list = []

    def __init__(self, config, system, tools, max_tokens):
        self.step = 0

    def send_user(self, text):
        self.user = text

    def next(self):
        self.step += 1
        return self.script[self.step - 1]

    def send_results(self, results):
        type(self).last_results = results


@pytest.fixture
def fake_llm(monkeypatch):
    llm_service.key_manager.configure_test_provider("fake", ["k"], model="m")
    monkeypatch.setitem(llm_service._TOOL_SESSIONS, "fake", _FakeSession)
    yield _FakeSession
    llm_service.key_manager.clear_test_provider("fake")


def test_tool_loop_executes_tools_and_returns_final_text(fake_llm):
    fake_llm.script = [("", [("c1", "column_stats", {"column": "age"}), ("c2", "nope", {})]), ('{"ok": 1}', [])]
    tools = dataset_tools.build_tools(_classification_df(), "churn")
    result = llm_service.run_tool_loop("sys", "usr", tools)
    assert result.text == '{"ok": 1}'
    assert [(c["tool"], c["ok"]) for c in result.calls] == [("column_stats", True), ("nope", False)]
    assert '"n_unique"' in fake_llm.last_results[0][1] and fake_llm.last_results[1][1].startswith("error")


def test_tool_loop_step_budget_and_unavailable(fake_llm):
    fake_llm.script = [("", [("c", "outlier_report", {})])] * 10
    with pytest.raises(RuntimeError):
        llm_service.run_tool_loop("s", "u", dataset_tools.build_tools(_classification_df()), max_steps=3)
    llm_service.key_manager.clear_test_provider("fake")  # simulate nothing configured at all
    with pytest.raises(llm_service.ToolLoopUnavailable):
        llm_service.run_tool_loop("s", "u", [])


def test_gemini_has_a_tool_adapter():
    assert "gemini" in llm_service._TOOL_SESSIONS
    schema = llm_service._gemini_schema(
        {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column"]}
    )
    assert schema["type"] == "OBJECT" and schema["properties"]["n"]["type"] == "INTEGER"


def test_problem_detection_llm_uses_tools_and_is_guarded(fake_llm):
    df = _classification_df()
    profile = {"column_roles": {}}
    fake_llm.script = [
        ("", [("c1", "leakage_probe", {"column": "age"})]),
        ('```json\n{"target_column": "income", "confidence": 0.99, "rationale": "continuous outcome"}\n```', []),
    ]
    out = problem_detection_llm.detect(df, profile)
    assert out["source"] == "llm" and out["target_column"] == "income"
    assert out["problem_type"] == "regression"  # from real dtype, not the model
    assert out["confidence"] <= 0.85 and out["tool_calls"][0]["tool"] == "leakage_probe"

    fake_llm.script = [('{"target_column": "not_a_column", "confidence": 0.9, "rationale": "x"}', [])]
    out = problem_detection_llm.detect(df, profile)
    assert out["source"] == "deterministic" and out["fallback_reason"].startswith("llm_chose_non_candidate")

    declared = problem_detection_llm.detect(df, profile, declared_target="churn")
    assert declared["source"] == "deterministic" and declared["target_column"] == "churn"  # user intent wins


def test_transformation_llm_guardrails(fake_llm):
    df = _classification_df()
    fields = {"numerical_columns": ["age", "income"], "categorical_columns": ["region"]}
    fake_llm.script = [('{"scaling_method": "minmax", "confidence": 0.95, "reason": "bounded ranges"}', [])]
    out = transformation_llm.propose(df, fields)
    assert out["source"] == "llm" and out["scaling_method"] == "minmax"
    assert out["confidence"] <= transformation_agent.propose(df, fields)["confidence"]  # LLM can only lower it

    first = transformation_agent.propose(df, fields)
    fake_llm.script = [('{"scaling_method": "%s", "confidence": 0.9, "reason": "r"}' % first["scaling_method"], [])]
    out = transformation_llm.propose(df, fields, _fb(first))
    assert out["source"] == "deterministic"  # model tried to repeat a rejected scaler -> guardrail
    assert out["scaling_method"] != first["scaling_method"]

    fake_llm.script = [("not json at all", [])]
    assert transformation_llm.propose(df, fields)["source"] == "deterministic"


def test_leakage_probe_detects_a_leaky_column():
    df = pd.DataFrame({"leak": [0, 1] * 50, "noise": range(100), "y": [0, 1] * 50})
    probe = {t.name: t for t in dataset_tools.build_tools(df, "y")}["leakage_probe"]
    assert probe.fn("leak")["verdict"] == "possible_leakage"


# ---------------- Rejected final recommendation ------------------------------------------

class _R:
    def __init__(self, algorithm, rank):
        self.algorithm, self.rank = algorithm, rank


def test_recommendation_selection_skips_rejected_and_honours_names():
    from app.agents import recommendation_agent as ra

    ranked = [_R("a", 1), _R("a", 2), _R("b", 3), _R("c", 4), _R("d", 5)]
    assert [r.rank for r in ra._select_top_runs(ranked, None)] == [1, 2, 3]
    rejected_a = _fb({"top_choice": {"algorithm": "a"}})
    assert [r.algorithm for r in ra._select_top_runs(ranked, rejected_a)] == ["b", "c", "d"]  # all of "a" skipped
    prefer_d = _fb({"top_choice": {"algorithm": "a"}}, "use d instead")
    assert ra._select_top_runs(ranked, prefer_d)[0].algorithm == "d"
    exclude_b = _fb({"top_choice": {"algorithm": "a"}}, "exclude b")
    assert "b" not in [r.algorithm for r in ra._select_top_runs(ranked, exclude_b)]
    assert ra._select_top_runs(ranked[:2], rejected_a) == []


def test_rejecting_final_recommendation_recommends_next_best_and_can_still_complete(dataset_factory):  # noqa: F811
    run = dataset_factory(_classification_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(run.id)
        run = _drive_until(db, run.id, "awaiting_recommendation_approval")
        assert run.status == "awaiting_recommendation_approval", run.error_message

        first = agent_decision_service.latest_decision(db, run.id, "recommendation")
        first_algo = first.decision_json["top_choice"]["algorithm"]

        # what the review endpoint does on reject
        assert agent_decision_service.can_revise(db, run.id, "recommendation")
        first.status, first.override_reason = "rejected", "I do not want this model"
        db.commit()
        orchestrator.revise_after_rejection(run.id)

        db.expire_all()
        run = db.get(PipelineRunORM, run.id)
        assert run.status == "awaiting_recommendation_approval", run.error_message  # NOT failed
        rows = agent_decision_service.all_decisions(db, run.id, "recommendation")
        assert [r.status for r in rows] == ["rejected", "proposed"]
        second = rows[1]
        assert second.decision_json["top_choice"]["algorithm"] != first_algo
        assert second.decision_json["revision"] == 1

        second.status = "approved"
        db.commit()
        orchestrator.finalize_after_recommendation_approval(run.id)
        db.expire_all()
        run = db.get(PipelineRunORM, run.id)
        assert run.status == "completed", run.error_message
        champion = db.get(__import__("app.db.models", fromlist=["ModelRun"]).ModelRun, run.champion_run_id)
        assert champion.algorithm == second.decision_json["top_choice"]["algorithm"]
    finally:
        db.close()


# ---------------- Phase 3 remainder: breaker, cleaning LLM, recommendation narrative -----

def _cleaning_df():
    import numpy as np

    rng = np.random.default_rng(1)
    n = 100
    skewed = rng.exponential(2.0, n)
    skewed[:10] = float("nan")
    return pd.DataFrame({
        "skewed_num": skewed,
        "cat": (["a", "b", "c", "d"] * 25),
        "empty": [float("nan")] * n,
        "y": [0, 1] * 50,
    }).assign(cat=lambda d: d["cat"].where(d.index >= 5))


def test_rate_limit_error_trips_breaker_and_skips_further_calls(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("429 You exceeded your current quota")

    monkeypatch.setitem(llm_service._DISPATCH, "fake", boom)
    with pytest.raises(RuntimeError):
        llm_service.call("fake", "k", "m", "s", [{"role": "user", "content": "x"}])
    assert llm_service.breaker_open()
    with pytest.raises(RuntimeError, match="temporarily disabled"):
        llm_service.call("fake", "k", "m", "s", [{"role": "user", "content": "x"}])
    assert len(calls) == 1  # second call never reached the provider
    # explain() degrades to its template instead of raising
    monkeypatch.setattr(llm_service, "_active_config", lambda: {"provider": "fake", "api_key": "k", "model": "m"})
    assert llm_service.explain("k", {}, "fallback text") == "fallback text"
    llm_service.reset_breaker()
    assert not llm_service.breaker_open()


def test_non_quota_errors_do_not_trip_breaker(monkeypatch):
    monkeypatch.setitem(llm_service._DISPATCH, "fake", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad json")))
    with pytest.raises(ValueError):
        llm_service.call("fake", "k", "m", "s", [{"role": "user", "content": "x"}])
    assert not llm_service.breaker_open()


def test_cleaning_llm_overrides_are_validated(monkeypatch):
    from app.agents import cleaning_plan_llm

    df = _cleaning_df()
    validation = {"checks": []}
    base = cleaning_plan_agent.propose(df, validation, "y")
    base_actions = {r["column"]: r["action"] for r in base["recommendations"]}
    assert base_actions["empty"] == "drop_column"

    reply = (
        '{"overrides": ['
        '{"column": "skewed_num", "action": "fill_zero", "reason": "missing means none"},'
        '{"column": "empty", "action": "mean", "reason": "trying to un-drop"},'          # not editable
        '{"column": "cat", "action": "median", "reason": "invalid for categorical"},'   # invalid action
        '{"column": "ghost", "action": "mean", "reason": "no such column"}'             # unknown column
        '], "confidence": 0.9}'
    )
    monkeypatch.setattr(llm_service, "complete", lambda system, user, max_tokens=0: reply)
    out = cleaning_plan_llm.propose(df, validation, "y")
    actions = {r["column"]: r["action"] for r in out["recommendations"]}
    assert out["source"] == "llm"
    assert actions["skewed_num"] == "fill_zero"          # valid override applied
    assert actions["empty"] == "drop_column"             # safety rule untouched
    assert actions["cat"] == base_actions["cat"]         # invalid action ignored
    assert [o["column"] for o in out["llm_overrides"]] == ["skewed_num"]


def test_cleaning_llm_falls_back_on_failure(monkeypatch):
    from app.agents import cleaning_plan_llm

    df, validation = _cleaning_df(), {"checks": []}
    monkeypatch.setattr(llm_service, "complete", lambda *a, **k: "garbage, not json")
    out = cleaning_plan_llm.propose(df, validation, "y")
    assert out["source"] == "deterministic" and out["fallback_reason"] == "invalid_json"

    def quota(*a, **k):
        raise RuntimeError("429 quota")

    monkeypatch.setattr(llm_service, "complete", quota)
    out = cleaning_plan_llm.propose(df, validation, "y")
    assert out["source"] == "deterministic" and out["fallback_reason"].startswith("llm_error")
    # without any provider configured the plan equals the rule-based one
    monkeypatch.undo()
    plain = cleaning_plan_agent.propose(df, validation, "y")
    assert cleaning_plan_llm.propose(df, validation, "y")["recommendations"] == plain["recommendations"]


def test_recommendation_carries_narrative_and_critic(dataset_factory):  # noqa: F811
    run = dataset_factory(_classification_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(run.id)
        run = _drive_until(db, run.id, "awaiting_recommendation_approval")
        rec = agent_decision_service.latest_decision(db, run.id, "recommendation").decision_json
        assert rec["narrative"] and "critic" in rec  # narrative falls back to the rationale when no LLM
    finally:
        db.close()


# ---------------- Phase 4: imbalance, feature engineering ---------------------------------

def _skewed_imbalanced_df(n=300):
    import numpy as np

    rng = np.random.default_rng(3)
    amount = rng.lognormal(mean=4.0, sigma=1.2, size=n)            # heavily right-skewed, non-negative
    age = rng.integers(20, 70, n)
    score = amount * 0.02 + rng.normal(0, 1.0, n)
    churn = ["yes" if s > 5.2 else "no" for s in score]            # minority class
    return pd.DataFrame({
        "amount": amount, "age": age, "region": (["n", "s", "e", "w"] * (n // 4 + 1))[:n], "churn": churn,
    })


def test_imbalance_analysis_and_feedback_opt_out():
    balanced = transformation_agent.analyze_imbalance(pd.Series(["a", "b"] * 50))
    assert balanced["strategy"] == "none"
    skewed = transformation_agent.analyze_imbalance(pd.Series(["a"] * 90 + ["b"] * 10))
    assert skewed["strategy"] == "class_weight_balanced" and skewed["ratio"] == 9.0
    df = _skewed_imbalanced_df()
    fields = {"numerical_columns": ["amount", "age"], "categorical_columns": ["region"]}
    proposal = transformation_agent.propose(df, fields, None, "classification", "churn")
    assert proposal["imbalance"]["strategy"] == "class_weight_balanced"
    assert [t["column"] for t in proposal["feature_transforms"]] == ["amount"]
    opted_out = transformation_agent.propose(df, fields, _fb(proposal, "no log and no class weight"), "classification", "churn")
    assert opted_out["feature_transforms"] == [] and opted_out["imbalance"]["strategy"] == "none"
    # clustering / no problem type: classic behaviour, nothing new proposed
    assert transformation_agent.propose(df, fields)["feature_transforms"] == []


def test_class_weight_context_reaches_supporting_models_only():
    from app.plugins.classification_base import class_weight_context
    from app.plugins.registry import CLASSIFICATION_PLUGIN_REGISTRY as REG

    lr, gb = REG["logistic_regression"], REG["gradient_boosting"]
    assert lr.build_configured(lr.param_grid({})[0]).get_params()["class_weight"] is None
    with class_weight_context("balanced"):
        assert lr.build_configured(lr.param_grid({})[0]).get_params()["class_weight"] == "balanced"
        gb.build_configured(gb.param_grid({})[0])  # unsupported estimator: silently unchanged, no error
    assert lr.build_configured(lr.param_grid({})[0]).get_params()["class_weight"] is None  # context restored


def test_log1p_transform_is_stateless_and_safe():
    from app.services import feature_engineering_service as fe

    df = _skewed_imbalanced_df()
    specs = fe.propose_log_transforms(df, ["amount", "age"], "churn")
    assert [s["column"] for s in specs] == ["amount"] and specs[0]["skew_after"] < specs[0]["skew_before"]
    out = fe.apply(df, specs)
    assert out["amount"].skew() < df["amount"].skew() and (out["age"] == df["age"]).all()
    assert df["amount"].max() > 100  # input frame not mutated
    negative = pd.DataFrame({"amount": [-5.0, None, 10.0]})
    res = fe.apply(negative, [{"column": "amount", "op": "log1p"}])["amount"]
    assert res.iloc[0] == 0.0 and pd.isna(res.iloc[1])  # negatives clipped, NaN preserved
    assert fe.apply(df, [{"column": "ghost", "op": "log1p"}]).equals(df)  # unknown column ignored
    assert abs(fe.inverse_value(fe.apply(pd.DataFrame({"amount": [99.0]}), specs)["amount"].iloc[0], "amount", specs) - 99.0) < 1e-6


def test_end_to_end_imbalanced_skewed_run_predicts_on_raw_values(dataset_factory):  # noqa: F811
    import joblib

    run = dataset_factory(_skewed_imbalanced_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(run.id)
        run = _drive_until(db, run.id, "awaiting_recommendation_approval")
        assert run.status == "awaiting_recommendation_approval", run.error_message

        tr = agent_decision_service.latest_decision(db, run.id, "transformation").decision_json
        assert tr["imbalance"]["strategy"] == "class_weight_balanced"
        assert [t["column"] for t in tr["feature_transforms"]] == ["amount"]

        from app.db.models import Job as JobORM
        job = db.get(JobORM, run.job_id)
        assert job.config_json.get("class_weight") == "balanced"
        plan = db.get(__import__("app.db.models", fromlist=["PreprocessingPlanORM"]).PreprocessingPlanORM, job.preprocessing_plan_id)
        assert [t["column"] for t in plan.feature_transforms] == ["amount"]

        rec = agent_decision_service.latest_decision(db, run.id, "recommendation")
        rec.status = "approved"
        db.commit()
        orchestrator.finalize_after_recommendation_approval(run.id)
        db.expire_all()
        run = db.get(PipelineRunORM, run.id)
        assert run.status == "completed", run.error_message

        # the persisted model takes RAW values, and the playground schema describes raw ranges
        pipeline = joblib.load(run.champion_model_path)
        assert pipeline.feature_transforms and pipeline.transform({"amount": 5000.0, "age": 40, "region": "n"}).shape[0] == 1
        result = pipeline.predict({"amount": 5000.0, "age": 40, "region": "n"})
        assert result["prediction"] in ("yes", "no")
        amount_schema = run.feature_schema_json["amount"]
        assert amount_schema["max"] > 100 and amount_schema["min"] <= amount_schema["default"] <= amount_schema["max"]
    finally:
        db.close()


# ---------------- Phase 4: adaptive HPO budget, chat actions ------------------------------

def test_hpo_budget_scales_with_data_size():
    from app.services import hpo_service

    assert hpo_service.plan_budget(500, 3)["tier"] == "standard"
    med = hpo_service.plan_budget(20_000, 3)
    assert (med["tier"], med["cv_folds"]) == ("medium", 3)
    big = hpo_service.plan_budget(80_000, 3)
    assert big["tier"] == "large" and big["force_random"] and big["n_iter"] < med["n_iter"] + 1


def test_hpo_stops_early_when_baseline_leaves_no_headroom():
    import numpy as np
    from app.services import hpo_service

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-5, 0.3, (60, 2)), rng.normal(5, 0.3, (60, 2))])
    y = np.array([0] * 60 + [1] * 60)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    result = hpo_service.optimize("random_forest", X[:90], y[:90], X[90:], y[90:], {}, problem_type="classification")
    assert result["early_stopped"] is True and result["n_trials"] == 1
    assert result["baseline_metrics"]["f1_macro"] >= hpo_service.EARLY_STOP_AT


def test_hpo_still_searches_when_there_is_headroom():
    import numpy as np
    from app.services import hpo_service

    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(0, 1.5, 200) > 0).astype(int)  # noisy: baseline cannot be near-perfect
    result = hpo_service.optimize("random_forest", X[:150], y[:150], X[150:], y[150:], {}, problem_type="classification")
    assert result["early_stopped"] is False and result["n_trials"] > 1


def test_chat_instruction_proposes_gated_action_and_applies_via_review(dataset_factory):  # noqa: F811
    from fastapi.testclient import TestClient

    from app.main import app

    run = dataset_factory(_classification_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(run.id)  # -> awaiting_problem_approval
        client = TestClient(app)

        question = client.post(f"/api/v1/pipeline-runs/{run.id}/chat", json={"question": "What is this dataset about?"}).json()
        assert question["suggested_action"] is None  # a question never proposes an action

        proposal = client.post(f"/api/v1/pipeline-runs/{run.id}/chat", json={"question": "Use a different target column"}).json()
        action = proposal["suggested_action"]
        assert action["agent_name"] == "problem_detection" and action["review_request"]["action"] == "reject"
        db.expire_all()
        assert agent_decision_service.latest_decision(db, run.id, "problem_detection").status == "proposed"  # nothing executed yet

        # user confirms -> the normal review endpoint (BackgroundTasks run synchronously in TestClient)
        resp = client.post(
            f"/api/v1/pipeline-runs/{run.id}/decisions/{action['decision_id']}/review",
            json={**action["review_request"], "reviewed_by": "tester"},
        )
        assert resp.status_code == 200
        db.expire_all()
        rows = agent_decision_service.all_decisions(db, run.id, "problem_detection")
        assert [r.status for r in rows] == ["rejected", "proposed"] and rows[1].decision_json["target_column"] != "churn"
        assert rows[0].override_reason == "Use a different target column"
    finally:
        db.close()


def test_chat_action_is_not_offered_for_non_revisable_or_no_pending_stage():
    from app.services import chat_action_service

    assert chat_action_service._IMPERATIVE.search("use random forest instead")
    assert not chat_action_service._IMPERATIVE.search("why was random forest chosen?")


# ---------------- Phase 5: task queue, auth, monitoring, memory, agent metrics -------------

def test_task_queue_serializes_same_key_and_captures_failures(monkeypatch):
    import time

    from app.services import task_queue_service as tq

    monkeypatch.setattr(tq, "BACKEND", "thread")
    events = []

    def work(tag):
        events.append(("start", tag))
        time.sleep(0.15)
        events.append(("end", tag))

    def boom():
        raise RuntimeError("expected failure")

    before = tq.stats()
    tq.submit(work, "a", key="same-run")
    tq.submit(work, "b", key="same-run")
    tq.submit(boom)
    deadline = time.time() + 10
    while time.time() < deadline:
        now = tq.stats()
        if now["completed"] - before["completed"] >= 2 and now["failed"] - before["failed"] >= 1:
            break
        time.sleep(0.05)
    now = tq.stats()
    assert now["completed"] - before["completed"] == 2 and now["failed"] - before["failed"] == 1
    # same key => strictly one after the other, never interleaved
    assert [e[0] for e in events] == ["start", "end", "start", "end"]
    assert now["backend"] == "thread" and now["workers"] >= 1


def test_task_queue_inline_backend_runs_synchronously(monkeypatch):
    from app.services import task_queue_service as tq

    monkeypatch.setattr(tq, "BACKEND", "inline")
    seen = []
    tq.submit(seen.append, 1, key="k")
    assert seen == [1]


def test_api_key_auth_protects_routes_but_not_health(monkeypatch):
    from fastapi.testclient import TestClient

    from app.core import auth
    from app.main import app

    client = TestClient(app)
    assert client.get("/api/v1/pipeline-runs/none").status_code == 404  # auth off by default: reaches the handler

    monkeypatch.setattr(auth, "API_KEYS", {"s3cret"})
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/pipeline-runs/none").status_code == 401
    assert client.get("/api/v1/pipeline-runs/none", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/pipeline-runs/none", headers={"X-API-Key": "s3cret"}).status_code == 404
    assert client.get("/api/v1/pipeline-runs/none?api_key=s3cret").status_code == 404  # download-link style
    assert client.get("/api/v1/agent-metrics").status_code == 401
    assert client.get("/api/v1/agent-metrics", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_database_url_is_environment_driven():
    from app.core import config
    from app.db import database

    assert config.DATABASE_URL.startswith(("sqlite", "postgres"))
    assert database._IS_SQLITE == config.DATABASE_URL.startswith("sqlite")


def _drift_schema():
    return {"age": {"type": "numeric", "min": 20.0, "max": 60.0}, "region": {"type": "categorical", "options": ["n", "s"]}}


def _drift_logs(n, bad_age_share=0.0, bad_region_share=0.0, conf_first=0.9, conf_last=0.9):
    logs = []
    for i in range(n):
        bad_age = i < int(n * bad_age_share)
        bad_region = i < int(n * bad_region_share)
        logs.append({
            "input": {"age": 200 if bad_age else 40, "region": "zzz" if bad_region else "n"},
            "output": {"confidence": conf_first if i < n // 2 else conf_last},
        })
    return logs


def test_monitoring_agent_detects_input_and_confidence_drift():
    from app.agents import monitoring_agent as ma

    assert ma.assess(_drift_schema(), _drift_logs(5))["status"] == "insufficient_data"
    ok = ma.assess(_drift_schema(), _drift_logs(40))
    assert ok["status"] == "ok" and not ok["recommend_retrain"]

    crit = ma.assess(_drift_schema(), _drift_logs(40, bad_age_share=0.5))
    assert crit["status"] == "critical" and crit["recommend_retrain"] and crit["features"][0]["feature"] == "age"

    warn = ma.assess(_drift_schema(), _drift_logs(40, bad_region_share=0.15))
    assert warn["status"] == "warning" and any(f["feature"] == "region" for f in warn["findings"])

    conf = ma.assess(_drift_schema(), _drift_logs(40, conf_first=0.95, conf_last=0.6))
    assert conf["status"] == "critical" and conf["confidence_drop"] > 0.2


def test_monitoring_endpoint_guards(dataset_factory):  # noqa: F811
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    assert client.get("/api/v1/pipeline-runs/nope/monitoring").status_code == 404
    run = dataset_factory(_classification_df(), "churn")
    assert client.get(f"/api/v1/pipeline-runs/{run.id}/monitoring").status_code == 409  # nothing deployed yet


def test_agent_metrics_summary_and_endpoint():
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services import agent_metrics_service as am

    def d(agent, status, by=None, conf=0.9, run="r1", **js):
        return {"pipeline_run_id": run, "agent_name": agent, "status": status, "approved_by": by, "confidence": conf, "decision_json": js}

    decisions = [
        d("cleaning_plan", "approved", "system"), d("cleaning_plan", "rejected", "ana"), d("cleaning_plan", "approved", "ana"),
        d("transformation", "approved", "system", source="llm"), d("transformation", "approved", "system", source="deterministic", fallback_reason="llm_error: 429 quota"),
        d("quality_check", "approved", "system", verdict="retry", run="r2"), d("quality_check", "approved", "system", verdict="proceed", run="r2"),
    ]
    out = am.summarize(decisions)
    cp = out["agents"]["cleaning_plan"]
    assert (cp["auto_approved"], cp["human_approved"], cp["rejected"]) == (1, 1, 1) and cp["human_rejection_rate"] == 0.5
    tr = out["agents"]["transformation"]
    assert tr["llm_share"] == 0.5 and tr["fallback_reasons"] == {"llm_error": 1}
    assert out["n_runs"] == 2 and out["avg_self_corrections_per_run"] == 0.5 and out["runs_needing_self_correction"] == 1
    assert out["avg_human_revisions_per_run"] == 0.5
    assert am.summarize([])["n_decisions"] == 0

    body = TestClient(app).get("/api/v1/agent-metrics").json()
    assert {"agents", "llm", "task_queue", "auto_approval_rate"} <= set(body)


def test_memory_shortlist_keeps_prior_winners_and_never_removes():
    from app.services import run_memory_service as mem

    shortlist = {
        "shortlist": [
            {"algorithm": "svm", "recommended": False, "rationale": "slow"},
            {"algorithm": "random_forest", "recommended": True, "rationale": "solid"},
        ],
        "selected_algorithms": ["random_forest"],
    }
    history = [{"champion_algorithm": "svm", "run_id": "a"}, {"champion_algorithm": "svm", "run_id": "b"},
               {"champion_algorithm": "random_forest", "run_id": "c"}]
    out = mem.apply_to_shortlist(shortlist, history, ["svm", "random_forest"])
    assert set(out["selected_algorithms"]) == {"svm", "random_forest"}  # prior winner re-included
    assert "Won 2 of 3" in out["shortlist"][0]["rationale"] and out["memory_note"].startswith("svm won 2 of 3")
    assert mem.apply_to_shortlist(shortlist, [], ["svm"]) is shortlist  # no history: untouched


def test_critic_uses_history():
    rec = {"confidence": "high", "top_choice": {"algorithm": "knn"}}
    kw = dict(problem_type="classification", recommendation=rec, quality={"flags": []}, validation=None, cleaning=None,
              n_rows=5000, retry_remedies=[])
    differs = critic_agent.review(**kw, history=[{"champion_algorithm": "svm"}, {"champion_algorithm": "svm"}])
    assert "differs_from_history" in {f["code"] for f in differs["findings"]}
    same = critic_agent.review(**kw, history=[{"champion_algorithm": "knn"}])
    assert "consistent_with_history" in {f["code"] for f in same["findings"]}


def test_second_run_on_same_problem_learns_from_the_first(dataset_factory):  # noqa: F811
    first = dataset_factory(_classification_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(first.id)
        done = _drive_until(db, first.id, "completed")
        assert done.status == "completed", done.error_message

        second = dataset_factory(_classification_df(), "churn")
        orchestrator.start(second.id)
        run = _drive_until(db, second.id, "awaiting_recommendation_approval")
        assert run.status == "awaiting_recommendation_approval", run.error_message

        shortlist = agent_decision_service.latest_decision(db, second.id, "algorithm_recommendation").decision_json
        assert shortlist.get("memory_note") and shortlist["prior_runs"]
        critic = agent_decision_service.latest_decision(db, second.id, "critic").decision_json
        assert {"differs_from_history", "consistent_with_history"} & {f["code"] for f in critic["findings"]}
    finally:
        db.close()


# ---------------- Consistent model-quality rating (executive summary / report) -------------

def test_model_assessment_levels_and_tie_are_independent():
    from app.services import model_assessment_service as ma

    assert ma.assess("classification", {"f1_macro": 0.90})["level"] == "Strong"
    assert ma.assess("classification", {"f1_macro": 0.75})["level"] == "Good"
    assert ma.assess("classification", {"f1_macro": 0.62})["level"] == "Fair"
    assert ma.assess("classification", {"f1_macro": 0.30})["level"] == "Weak"
    assert ma.assess("regression", {"r2": 0.85})["level"] == "Strong"
    assert ma.assess("clustering", {"silhouette": 0.1})["level"] == "Weak"
    assert ma.assess("classification", {})["level"] == "Unknown" and ma.assess(None, None)["level"] == "Unknown"
    perfect = ma.assess("classification", {"f1_macro": 1.0})
    assert perfect["level"] == "Verify" and "leakage" in perfect["explanation"]

    # a near-tie is a neutral note for good models, and irrelevant noise for weak/unknown ones
    assert ma.assess("classification", {"f1_macro": 0.9}, near_tie=True)["tie_note"]
    assert ma.assess("classification", {"f1_macro": 0.9}, near_tie=False)["tie_note"] is None
    assert ma.assess("classification", {"f1_macro": 0.2}, near_tie=True)["tie_note"] is None


class _FakeDecision:
    def __init__(self, decision_json):
        self.decision_json, self.status, self.human_edits_json = decision_json, "approved", None


def test_report_no_longer_calls_a_perfect_score_moderate():
    """Regression for: '100 out of 100 predictions correct' shown next to 'Moderate ... reasonable
    predictive capability' just because the top two models were nearly tied."""
    from app.agents import reporting_agent

    evaluation = _FakeDecision({"metrics": {"f1_macro": 1.0, "accuracy": 1.0}, "business_metrics": {"accuracy": "100 out of 100 predictions are expected to be correct."}})
    tied = _FakeDecision({"confidence": "low"})  # ranking closeness only
    out = reporting_agent._model_performance(evaluation, tied, "classification")
    assert out["confidence_level"] != "Moderate"
    assert out["confidence_level"] == "Verify"
    assert "reasonable predictive capability" not in out["confidence_explanation"]
    assert "almost equally well" in out["confidence_explanation"]  # tie is stated neutrally

    good = reporting_agent._model_performance(
        _FakeDecision({"metrics": {"f1_macro": 0.91}, "business_metrics": {}}), tied, "classification"
    )
    assert good["confidence_level"] == "Strong" and "low" not in good["confidence_level"].lower()
    weak = reporting_agent._model_performance(
        _FakeDecision({"metrics": {"f1_macro": 0.3}, "business_metrics": {}}), _FakeDecision({"confidence": "high"}), "classification"
    )
    assert weak["confidence_level"] == "Weak"


def test_executive_summary_uses_quality_level_not_tie_flag(dataset_factory):  # noqa: F811
    from fastapi.testclient import TestClient

    from app.main import app

    run = dataset_factory(_classification_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(run.id)
        _drive_until(db, run.id, "awaiting_recommendation_approval")
        body = TestClient(app).get(f"/api/v1/pipeline-runs/{run.id}/executive-summary").json()
        perf = body["performance_summary"]
        assert perf["level"] in {"Strong", "Good", "Fair", "Weak", "Verify"} and perf["explanation"]
        assert set(perf) >= {"headline_metric_sentence", "level", "explanation", "tie_note", "confidence"}
    finally:
        db.close()
