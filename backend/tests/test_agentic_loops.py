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
    monkeypatch.setattr(llm_service, "_active_config", lambda: {"provider": "fake", "api_key": "k", "model": "m"})
    monkeypatch.setitem(llm_service._TOOL_SESSIONS, "fake", _FakeSession)
    return _FakeSession


def test_tool_loop_executes_tools_and_returns_final_text(fake_llm):
    fake_llm.script = [("", [("c1", "column_stats", {"column": "age"}), ("c2", "nope", {})]), ('{"ok": 1}', [])]
    tools = dataset_tools.build_tools(_classification_df(), "churn")
    result = llm_service.run_tool_loop("sys", "usr", tools)
    assert result.text == '{"ok": 1}'
    assert [(c["tool"], c["ok"]) for c in result.calls] == [("column_stats", True), ("nope", False)]
    assert '"n_unique"' in fake_llm.last_results[0][1] and fake_llm.last_results[1][1].startswith("error")


def test_tool_loop_step_budget_and_unavailable(fake_llm, monkeypatch):
    fake_llm.script = [("", [("c", "outlier_report", {})])] * 10
    with pytest.raises(RuntimeError):
        llm_service.run_tool_loop("s", "u", dataset_tools.build_tools(_classification_df()), max_steps=3)
    monkeypatch.setattr(llm_service, "_active_config", lambda: None)
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
