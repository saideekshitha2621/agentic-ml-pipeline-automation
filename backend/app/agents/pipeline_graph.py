"""LangGraph state graph for the agent pipeline.

Pure sequencing/state-machine logic — contains no modeling logic itself, same as the
`orchestrator.py` this replaces. Every "propose" node opens its own DB session (mirrors
`job_runner_service.run_job`'s BackgroundTasks pattern) and does exactly what its matching
`advance_after_<x>_approval` function used to do: run one stage's agent, persist an
AgentDecision via `agent_decision_service`, and either continue immediately (auto-approved)
or pause for a human via the stage's dedicated no-op "gate" node.

Why a dedicated gate node per HITL checkpoint, instead of calling LangGraph's dynamic
`interrupt()` inline: `interrupt()` re-executes its node's body from the top on resume, so a
node that calls `_decide()` (an INSERT) before calling `interrupt()` would insert a duplicate
AgentDecision on resume. A gate node's only job is to exist as a static `interrupt_before`
target — its body never runs anything before the graph halts, so nothing can double-execute.

State graph (mirrors the docstring `orchestrator.py` used to carry):
  start -> gate_problem [HITL 1] ->
  after_problem -> (gate_validation [HITL 2] ->) after_validation ->
  (gate_cleaning [HITL 3] ->) after_cleaning ->
  (gate_transformation [HITL 4] ->) after_transformation ->
  (gate_split [HITL 5, classification/regression only] ->) algorithm_shortlist ->
  (gate_algorithm [HITL 6] ->) after_algorithm ->
  gate_recommendation [HITL 7] -> finalize
  (any propose node -> END, with run.status="failed" and error_message, on an unsupported
  problem type / blocking validation checks / a failed training job)

A propose node that doesn't need to pause just falls through to the next propose node in the
same `graph.invoke()` call — this is what makes a high-confidence, low-risk stage actually
skip the HITL wait, reproducing what `_advance_or_wait` did in the old orchestrator.
"""
from __future__ import annotations

import sqlite3
from typing import TypedDict

import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.core.config import STORAGE_DIR
from app.db.database import SessionLocal
from app.db.models import Dataset as DatasetORM
from app.db.models import Job as JobORM
from app.db.models import ModelRun as ModelRunORM
from app.db.models import PipelineRun as PipelineRunORM
from app.db.models import PreprocessingPlanORM
from app.services import (
    agent_decision_service,
    approval_policy_service,
    business_language_service,
    champion_service,
    error_translation_service,
    evaluation_service,
    hpo_service,
    job_runner_service,
    metric_glossary,
    preprocessing_service,
)
from app.agents import (
    algorithm_selection_llm,
    business_framing_agent,
    cleaning_plan_agent,
    data_profiling_agent,
    data_validation_agent,
    problem_detection_agent,
    recommendation_agent,
    reporting_agent,
    split_agent,
    transformation_agent,
)

SUPPORTED_PROBLEM_TYPES = {"clustering", "classification", "regression"}
# Both classification and regression are "supervised" for the purposes of every stage that
# branches on "does this need a train/test split / HPO / a champion model" — clustering is
# the only path with no target and no held-out evaluation set.
SUPERVISED_PROBLEM_TYPES = {"classification", "regression"}


class PipelineState(TypedDict, total=False):
    pipeline_run_id: str
    route: str  # internal signal a propose node sets for its own conditional edge


def _proposal(decision) -> dict:
    return decision.human_edits_json if decision.status == "edited" else decision.decision_json


def _decide(db, run, *, agent_name: str, stage: str, decision: dict, confidence, reasoning: str, auto_approve: bool | None = False):
    """Every AgentDecision in this pipeline passes through here so each one carries a
    `business_impact` sentence and an `approval_reason` — the audit trail for why a stage
    was auto-approved or sent for human review.

    `auto_approve`:
      - True  -> always informational, recorded automatically (e.g. profiling, evaluation).
      - False -> always mandatory human review (problem_detection, recommendation) —
                 these two carry outsized, hard-to-undo business consequences.
      - None  -> policy-driven (`approval_policy_service`): auto-approves when confidence
                 is high and no critical risk is detected in what was just proposed,
                 otherwise pauses for review with a recorded reason.
    """
    if auto_approve is None:
        auto_approve, approval_reason = approval_policy_service.decide(stage, decision, confidence)
    elif auto_approve is False:
        approval_reason = approval_policy_service.mandatory_reason()
    else:
        approval_reason = "Informational — recorded automatically, no approval needed."
    decision = {
        **decision,
        "business_impact": business_language_service.business_impact_for_stage(stage, decision),
        "approval_reason": approval_reason,
    }
    return agent_decision_service.record_decision(
        db, run, agent_name=agent_name, stage=stage, decision=decision,
        confidence=confidence, reasoning=reasoning, auto_approve=auto_approve,
    )


def _fail(db, pipeline_run_id: str, exc: Exception) -> None:
    db.rollback()
    run = db.get(PipelineRunORM, pipeline_run_id)
    if run:
        run.status = "failed"
        run.error_message = error_translation_service.to_business_message(exc)
        db.commit()


def _gate(state: PipelineState) -> dict:
    """No-op — exists only as a static `interrupt_before` target (see module docstring)."""
    return {}


def start_node(state: PipelineState) -> dict:
    """Stage 1-2: Data Profiling Agent -> Problem Detection Agent, then pause for HITL 1
    (always — `gate_problem` is unconditionally the next node)."""
    pipeline_run_id = state["pipeline_run_id"]
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return {}
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        profile = data_profiling_agent.analyze(df, dataset.profile_json)
        _decide(
            db, run, agent_name="data_profiling", stage="profiling",
            decision={"column_roles": profile["column_roles"], "dataset_understanding": profile["dataset_understanding"]},
            confidence=1.0, reasoning=data_profiling_agent.summarize(profile), auto_approve=True,
        )

        result = problem_detection_agent.detect(df, profile, declared_target=run.declared_target)
        _decide(
            db, run, agent_name="problem_detection", stage="problem_detection", decision=result,
            confidence=result["confidence"], reasoning=" ".join(result["reasoning"]), auto_approve=False,
        )
        run.status = "awaiting_problem_approval"
        db.commit()
        return {}
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def after_problem_node(state: PipelineState) -> dict:
    """Stage 3: Business Framing Agent -> Data Validation Agent, then pause for HITL 2
    (unless the policy auto-approves it, or a blocking check fails the run)."""
    pipeline_run_id = state["pipeline_run_id"]
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return {"route": "fail"}

        decision = agent_decision_service.latest_decision(db, pipeline_run_id, "problem_detection")
        proposal = _proposal(decision)
        problem_type = proposal["problem_type"]
        run.problem_type = problem_type
        run.declared_target = proposal.get("target_column") or run.declared_target

        if problem_type not in SUPPORTED_PROBLEM_TYPES:
            run.status = "failed"
            run.error_message = (
                f"Problem type '{problem_type}' was confirmed, but this build only executes the "
                f"{sorted(SUPPORTED_PROBLEM_TYPES)} paths end-to-end."
            )
            db.commit()
            return {"route": "fail"}

        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)
        profile = data_profiling_agent.analyze(df, dataset.profile_json)

        framing = business_framing_agent.frame(
            dataset.filename, dataset.n_rows, problem_type, run.declared_target,
            column_roles=profile.get("column_roles"),
        )
        _decide(
            db, run, agent_name="business_framing", stage="business_framing", decision=framing,
            confidence=1.0, reasoning=business_framing_agent.summarize(framing), auto_approve=True,
        )

        validation = data_validation_agent.validate(
            df, profile, target_column=run.declared_target, problem_type=run.problem_type
        )
        # Real confidence, not a flat constant — a clean validation is high-confidence,
        # critical findings are low-confidence, both feeding directly into whether this
        # stage can auto-approve (see approval_policy_service.decide).
        validation_confidence = {"ok": 0.95, "warning": 0.6, "critical": 0.25}.get(validation["overall_status"], 0.5)
        decision_row = _decide(
            db, run, agent_name="data_validation", stage="data_validation", decision=validation,
            confidence=validation_confidence, reasoning=data_validation_agent.summarize(validation), auto_approve=None,
        )

        # "Dataset size" and "Target column validity" are hard blockers — there is no
        # cleaning action that fixes an empty dataset or a missing/invalid target, so the
        # run stops here with a business-friendly message instead of waiting on a HITL
        # approval that could only ever end in a downstream crash.
        blocking = [
            c for c in validation["checks"]
            if c["name"] in ("Dataset size", "Target column validity") and c["status"] == "critical"
        ]
        if blocking:
            run.status = "failed"
            run.error_message = " ".join(c["detail"] for c in blocking)
            db.commit()
            return {"route": "fail"}

        if decision_row.status == "approved":
            db.commit()
            return {"route": "continue"}
        run.status = "awaiting_validation_approval"
        db.commit()
        return {"route": "gate"}
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def after_validation_node(state: PipelineState) -> dict:
    """Stage 4: Cleaning Plan Agent, then pause for HITL 3 unless auto-approved."""
    pipeline_run_id = state["pipeline_run_id"]
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return {"route": "fail"}
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        validation_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "data_validation")
        validation_result = _proposal(validation_decision)

        plan = cleaning_plan_agent.propose(df, validation_result, target_column=run.declared_target)
        # High confidence by default (the rules are deterministic and explainable); lower
        # when the plan itself signals real risk, so the policy can decide whether to pause.
        no_information_cols = [r for r in plan["recommendations"] if r.get("no_information")]
        drop_row_cols = [r for r in plan["recommendations"] if r["action"] == "drop_rows"]
        cleaning_confidence = 0.5 if no_information_cols else 0.7 if drop_row_cols else 0.9
        decision_row = _decide(
            db, run, agent_name="cleaning_plan", stage="cleaning_plan", decision=plan,
            confidence=cleaning_confidence, reasoning=cleaning_plan_agent.summarize(plan), auto_approve=None,
        )
        if decision_row.status == "approved":
            db.commit()
            return {"route": "continue"}
        run.status = "awaiting_cleaning_approval"
        db.commit()
        return {"route": "gate"}
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def after_cleaning_node(state: PipelineState) -> dict:
    """Stage 5: Transformation Agent, then pause for HITL 4 unless auto-approved."""
    pipeline_run_id = state["pipeline_run_id"]
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return {"route": "fail"}
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        cleaning_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "cleaning_plan")
        cleaning_plan = _proposal(cleaning_decision)
        plan_fields = cleaning_plan_agent.to_preprocessing_plan_fields(cleaning_plan)

        transformation = transformation_agent.propose(df, plan_fields)
        decision_row = _decide(
            db, run, agent_name="transformation", stage="transformation", decision=transformation,
            confidence=0.8, reasoning=transformation_agent.summarize(transformation), auto_approve=None,
        )
        if decision_row.status == "approved":
            db.commit()
            return {"route": "continue"}
        run.status = "awaiting_transformation_approval"
        db.commit()
        return {"route": "gate"}
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def after_transformation_node(state: PipelineState) -> dict:
    """Stage 6: Train/Test Split Agent. Classification/regression pauses for HITL 5 unless
    auto-approved; clustering has no split concept, so it's recorded informationally and the
    pipeline proceeds straight to the algorithm shortlist."""
    pipeline_run_id = state["pipeline_run_id"]
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return {"route": "fail"}
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        split = split_agent.recommend(df, run.problem_type, target_column=run.declared_target)
        applicable = run.problem_type in SUPERVISED_PROBLEM_TYPES
        decision_row = _decide(
            db, run, agent_name="train_test_split", stage="train_test_split", decision=split,
            confidence=0.8 if applicable else 1.0, reasoning=split_agent.summarize(split),
            auto_approve=None if applicable else True,
        )
        if not applicable or decision_row.status == "approved":
            db.commit()
            return {"route": "continue"}
        run.status = "awaiting_split_approval"
        db.commit()
        return {"route": "gate"}
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def algorithm_shortlist_node(state: PipelineState) -> dict:
    """Stage 7: Algorithm Shortlist Agent (LLM-driven, deterministic fallback), then pause
    for HITL 6 unless auto-approved."""
    pipeline_run_id = state["pipeline_run_id"]
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return {"route": "fail"}
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)
        profile = data_profiling_agent.analyze(df, dataset.profile_json)

        validation_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "data_validation")
        validation = _proposal(validation_decision) if validation_decision else None

        shortlist = algorithm_selection_llm.recommend(profile, validation, run.problem_type, df, run.declared_target)
        decision_row = _decide(
            db, run, agent_name="algorithm_recommendation", stage="algorithm_recommendation", decision=shortlist,
            confidence=0.8, reasoning=algorithm_selection_llm.summarize(shortlist), auto_approve=None,
        )
        if decision_row.status == "approved":
            db.commit()
            return {"route": "continue"}
        run.status = "awaiting_algorithm_approval"
        db.commit()
        return {"route": "gate"}
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


_HPO_KEY_METRIC = {"classification": "f1_macro", "regression": "r2"}


def _run_hpo(db, run: PipelineRunORM, job: JobORM, plan_fields: dict) -> None:
    dataset = db.get(DatasetORM, run.dataset_id)
    df = pd.read_csv(dataset.storage_path)
    X_train, X_test, y_train, y_test, _report = preprocessing_service.split_target(
        df, run.declared_target, plan_fields, test_size=job.test_size or 0.2,
        stratify=(run.problem_type == "classification"),
    )
    algorithms = sorted({r.algorithm for r in job.runs})
    key_metric = _HPO_KEY_METRIC[run.problem_type]
    results = []
    for algorithm in algorithms:
        result = hpo_service.optimize(
            algorithm, X_train.values, y_train.values, X_test.values, y_test.values, job.config_json,
            problem_type=run.problem_type,
        )
        results.append({k: v for k, v in result.items() if k != "run"})
        if result.get("skipped"):
            continue
        best_run = next((r for r in job.runs if r.algorithm == algorithm), None)
        if best_run is not None:
            best_run.baseline_metrics_json = result["baseline_metrics"]
            best_run.params_json = result["best_params"]
            best_run.metrics_json = result["optimized_metrics"]
    db.commit()

    reasoning = "; ".join(
        f"{r['algorithm']}: baseline {key_metric}={r.get('baseline_metrics', {}).get(key_metric)} -> "
        f"optimized {key_metric}={r.get('optimized_metrics', {}).get(key_metric)} ({r.get('n_trials', 0)} trial(s))"
        for r in results if not r.get("skipped")
    ) or "No algorithms had a registered search space."
    _decide(
        db, run, agent_name="hyperparameter_optimization", stage="hyperparameter_optimization",
        decision={"results": results}, confidence=1.0, reasoning=reasoning, auto_approve=True,
    )


_CLUSTERING_METRIC_ALIASES = {
    "silhouette": "silhouette_score",
    "davies_bouldin": "davies_bouldin_score",
    "calinski_harabasz": "calinski_harabasz_score",
}


def _record_evaluation_summary(db, run: PipelineRunORM, job: JobORM) -> None:
    best = min((r for r in job.runs if r.rank), key=lambda r: r.rank, default=None)
    if best is None:
        best = job.runs[0] if job.runs else None
    metrics = best.metrics_json if best else {}
    # ModelRun.metrics_json for clustering uses short keys (silhouette, davies_bouldin,
    # calinski_harabasz) while the glossary/business-language lookups key off the same
    # names ModelRun's typed columns use (*_score) — translate for those two lookups only,
    # `metrics` itself stays as-is since other readers (e.g. the frontend) expect the short keys.
    lookup_metrics = {_CLUSTERING_METRIC_ALIASES.get(k, k): v for k, v in metrics.items()}
    glossary = metric_glossary.explain(run.problem_type, lookup_metrics)
    business_metrics = {
        k: s for k, v in lookup_metrics.items()
        if v is not None and (s := business_language_service.metric_sentence(k, v, run.problem_type))
    }
    _decide(
        db, run, agent_name="evaluation", stage="evaluation",
        decision={
            "champion_algorithm": best.algorithm if best else None, "metrics": metrics, "glossary": glossary,
            "business_metrics": business_metrics,
        },
        confidence=1.0,
        reasoning=f"Top-ranked model so far: {best.algorithm if best else 'none'}. Metrics: {metrics}.",
        auto_approve=True,
    )


def after_algorithm_node(state: PipelineState) -> dict:
    """Stage 8-10: build the approved plan, run training, then (classification/regression)
    HPO, then an Evaluation summary, then the Recommendation Agent, then pause for HITL 7
    (always — recommendation is mandatory review) unless the training job itself failed."""
    pipeline_run_id = state["pipeline_run_id"]
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return {"route": "fail"}

        cleaning_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "cleaning_plan")
        transformation_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "transformation")
        split_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "train_test_split")
        algorithm_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "algorithm_recommendation")

        cleaning_plan = _proposal(cleaning_decision)
        plan_fields = cleaning_plan_agent.to_preprocessing_plan_fields(cleaning_plan)
        transformation = _proposal(transformation_decision)
        plan_fields["scaling_method"] = transformation.get("scaling_method", plan_fields["scaling_method"])
        split = _proposal(split_decision)
        algorithms = _proposal(algorithm_decision)["selected_algorithms"]

        plan_orm = PreprocessingPlanORM(dataset_id=run.dataset_id, **plan_fields)
        db.add(plan_orm)
        db.commit()
        db.refresh(plan_orm)

        job = JobORM(
            dataset_id=run.dataset_id,
            preprocessing_plan_id=plan_orm.id,
            config_json={},
            status="queued",
            log_lines=["Job queued by agent orchestrator."],
            problem_type=run.problem_type,
            target_column=run.declared_target,
            selected_algorithms=algorithms,
            test_size=split.get("test_size") or 0.2,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        run.job_id = job.id
        run.status = "training"
        db.commit()

        job_runner_service.run_job(job.id)  # dispatches on job.problem_type internally

        db.expire_all()
        job = db.get(JobORM, job.id)
        run = db.get(PipelineRunORM, pipeline_run_id)
        if job.status != "completed":
            run.status = "failed"
            run.error_message = job.error_message or f"{run.problem_type.capitalize()} job failed."
            db.commit()
            return {"route": "fail"}

        _decide(
            db, run, agent_name="model_selection", stage="training",
            decision={"algorithms_run": sorted({r.algorithm for r in job.runs})},
            confidence=1.0, reasoning=f"Ran all {len(job.runs)} shortlisted {run.problem_type} plugin candidate(s).",
            auto_approve=True,
        )

        if run.problem_type in SUPERVISED_PROBLEM_TYPES:
            run.status = "hyperparameter_optimization"
            db.commit()
            _run_hpo(db, run, job, plan_fields)

        db.expire_all()
        run = db.get(PipelineRunORM, pipeline_run_id)
        job = db.get(JobORM, job.id)
        _record_evaluation_summary(db, run, job)

        recommendation = recommendation_agent.build_recommendation(job, db)
        top = recommendation["top_choice"]
        _decide(
            db, run, agent_name="recommendation", stage="recommendation", decision=recommendation,
            confidence=0.85 if recommendation["confidence"] == "high" else 0.5,
            reasoning=f"Recommending {top['algorithm']} (rank #{top['rank']}): {top['rationale']}",
            auto_approve=False,
        )
        run.status = "awaiting_recommendation_approval"
        db.commit()
        return {"route": "gate"}
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def finalize_node(state: PipelineState) -> dict:
    """Stage 11-12: persist the approved champion as a reloadable prediction pipeline
    (classification/regression only — unlocks the Prediction Playground/Explainability;
    clustering has no "predict a new row" concept), then the Reporting Agent compiles the
    full decision trail into a business report."""
    pipeline_run_id = state["pipeline_run_id"]
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return {}

        recommendation_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "recommendation")
        recommendation = _proposal(recommendation_decision)
        top = recommendation["top_choice"]

        if run.problem_type in SUPERVISED_PROBLEM_TYPES:
            dataset = db.get(DatasetORM, run.dataset_id)
            df = pd.read_csv(dataset.storage_path)
            cleaning_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "cleaning_plan")
            transformation_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "transformation")
            plan_fields = cleaning_plan_agent.to_preprocessing_plan_fields(_proposal(cleaning_decision))
            plan_fields["scaling_method"] = _proposal(transformation_decision).get("scaling_method", plan_fields["scaling_method"])

            champion_run = db.get(ModelRunORM, top["cluster_run_id"])
            built = champion_service.build_and_persist(
                df, plan_fields, run.declared_target, champion_run.algorithm, champion_run.params_json, run.id,
                problem_type=run.problem_type,
            )
            run.champion_model_path = built["model_path"]
            run.champion_run_id = champion_run.id
            run.feature_schema_json = built["feature_schema"]
            run.feature_importance_json = {"features": built["feature_importance"], "target_classes": built["target_classes"]}
            db.commit()

        run.status = "reporting"
        db.commit()

        dataset = db.get(DatasetORM, run.dataset_id)
        job = db.get(JobORM, run.job_id) if run.job_id else None
        decisions = list(run.decisions)

        run.status = "completed"  # set before building the report so final_outcome reflects it
        report = reporting_agent.build_report(run, decisions, dataset, job)
        _decide(
            db, run, agent_name="reporting", stage="reporting", decision={"report": report},
            confidence=1.0, reasoning="Compiled the full agent decision trail into a business-readable report.",
            auto_approve=True,
        )
        db.commit()
        return {}
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def _route(state: PipelineState) -> str:
    return state["route"]


def _build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("start", start_node)
    graph.add_node("gate_problem", _gate)
    graph.add_node("after_problem", after_problem_node)
    graph.add_node("gate_validation", _gate)
    graph.add_node("after_validation", after_validation_node)
    graph.add_node("gate_cleaning", _gate)
    graph.add_node("after_cleaning", after_cleaning_node)
    graph.add_node("gate_transformation", _gate)
    graph.add_node("after_transformation", after_transformation_node)
    graph.add_node("gate_split", _gate)
    graph.add_node("algorithm_shortlist", algorithm_shortlist_node)
    graph.add_node("gate_algorithm", _gate)
    graph.add_node("after_algorithm", after_algorithm_node)
    graph.add_node("gate_recommendation", _gate)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "start")
    graph.add_edge("start", "gate_problem")  # HITL 1 is always mandatory (auto_approve=False)
    graph.add_edge("gate_problem", "after_problem")
    graph.add_conditional_edges(
        "after_problem", _route, {"fail": END, "gate": "gate_validation", "continue": "after_validation"}
    )
    graph.add_edge("gate_validation", "after_validation")
    graph.add_conditional_edges(
        "after_validation", _route, {"gate": "gate_cleaning", "continue": "after_cleaning"}
    )
    graph.add_edge("gate_cleaning", "after_cleaning")
    graph.add_conditional_edges(
        "after_cleaning", _route, {"gate": "gate_transformation", "continue": "after_transformation"}
    )
    graph.add_edge("gate_transformation", "after_transformation")
    graph.add_conditional_edges(
        "after_transformation", _route, {"gate": "gate_split", "continue": "algorithm_shortlist"}
    )
    graph.add_edge("gate_split", "algorithm_shortlist")
    graph.add_conditional_edges(
        "algorithm_shortlist", _route, {"gate": "gate_algorithm", "continue": "after_algorithm"}
    )
    graph.add_edge("gate_algorithm", "after_algorithm")
    graph.add_conditional_edges(
        "after_algorithm", _route, {"fail": END, "gate": "gate_recommendation"}  # HITL 7 always mandatory
    )
    graph.add_edge("gate_recommendation", "finalize")
    graph.add_edge("finalize", END)

    conn = sqlite3.connect(str(STORAGE_DIR / "langgraph_checkpoints.db"), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "gate_problem", "gate_validation", "gate_cleaning", "gate_transformation",
            "gate_split", "gate_algorithm", "gate_recommendation",
        ],
    )


compiled_graph = _build_graph()
