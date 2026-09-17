"""Pipeline Orchestrator.

Pure sequencing/state-machine logic — contains no modeling logic itself. Each function
opens its own DB session (it runs inside FastAPI BackgroundTasks, same pattern as
`job_runner_service.run_job`) and advances `pipeline_runs.status` by exactly one stage,
persisting an AgentDecision at each step via `agent_decision_service`.

State graph:
  profiling -> awaiting_problem_approval -[HITL 1]->
  data_validation -> awaiting_validation_approval -[HITL 2]->
  cleaning_plan -> awaiting_cleaning_approval -[HITL 3]->
  transformation -> awaiting_transformation_approval -[HITL 4]->
  train_test_split -> awaiting_split_approval -[HITL 5, classification only]->
  algorithm_recommendation -> awaiting_algorithm_approval -[HITL 6]->
  training -> hyperparameter_optimization -> evaluation ->
  awaiting_recommendation_approval -[HITL 7]->
  reporting -> completed
  (any stage -> failed, with error_message)

Every `advance_after_<x>_approval` reads the just-reviewed decision via
`agent_decision_service.latest_decision`, and — matching the existing problem_detection
pattern — uses `human_edits_json` when `status == "edited"`, else `decision_json`.
"""
from __future__ import annotations

import pandas as pd

from app.db.database import SessionLocal
from app.db.models import Dataset as DatasetORM
from app.db.models import Job as JobORM
from app.db.models import ModelRun as ModelRunORM
from app.db.models import PipelineRun as PipelineRunORM
from app.db.models import PreprocessingPlanORM
from app.services import (
    agent_decision_service,
    champion_service,
    evaluation_service,
    hpo_service,
    job_runner_service,
    metric_glossary,
    preprocessing_service,
)
from app.agents import (
    algorithm_shortlist_agent,
    cleaning_plan_agent,
    data_profiling_agent,
    data_validation_agent,
    problem_detection_agent,
    recommendation_agent,
    reporting_agent,
    split_agent,
    transformation_agent,
)

SUPPORTED_PROBLEM_TYPES = {"clustering", "classification"}  # regression is a later phase


def _proposal(decision) -> dict:
    return decision.human_edits_json if decision.status == "edited" else decision.decision_json


def _fail(db, pipeline_run_id: str, exc: Exception) -> None:
    db.rollback()
    run = db.get(PipelineRunORM, pipeline_run_id)
    if run:
        run.status = "failed"
        run.error_message = str(exc)
        db.commit()


def start(pipeline_run_id: str) -> None:
    """Stage 1-2: Data Profiling Agent -> Problem Detection Agent, then pause for HITL 1."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        profile = data_profiling_agent.analyze(df, dataset.profile_json)
        agent_decision_service.record_decision(
            db, run, agent_name="data_profiling", stage="profiling",
            decision={"column_roles": profile["column_roles"], "dataset_understanding": profile["dataset_understanding"]},
            confidence=1.0, reasoning=data_profiling_agent.summarize(profile), auto_approve=True,
        )

        result = problem_detection_agent.detect(df, profile, declared_target=run.declared_target)
        agent_decision_service.record_decision(
            db, run, agent_name="problem_detection", stage="problem_detection", decision=result,
            confidence=result["confidence"], reasoning=" ".join(result["reasoning"]), auto_approve=False,
        )
        run.status = "awaiting_problem_approval"
        db.commit()
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def advance_after_problem_approval(pipeline_run_id: str) -> None:
    """Stage 3: Data Validation Agent, then pause for HITL 2."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return

        decision = agent_decision_service.latest_decision(db, pipeline_run_id, "problem_detection")
        proposal = _proposal(decision)
        problem_type = proposal["problem_type"]
        run.problem_type = problem_type
        run.declared_target = proposal.get("target_column") or run.declared_target

        if problem_type not in SUPPORTED_PROBLEM_TYPES:
            run.status = "failed"
            run.error_message = (
                f"Problem type '{problem_type}' was confirmed, but this build only executes the "
                f"{sorted(SUPPORTED_PROBLEM_TYPES)} paths end-to-end (regression is a later phase)."
            )
            db.commit()
            return

        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)
        profile = data_profiling_agent.analyze(df, dataset.profile_json)

        validation = data_validation_agent.validate(df, profile, target_column=run.declared_target)
        agent_decision_service.record_decision(
            db, run, agent_name="data_validation", stage="data_validation", decision=validation,
            confidence=1.0, reasoning=data_validation_agent.summarize(validation), auto_approve=False,
        )
        run.status = "awaiting_validation_approval"
        db.commit()
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def advance_after_validation_approval(pipeline_run_id: str) -> None:
    """Stage 4: Cleaning Plan Agent, then pause for HITL 3."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        validation_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "data_validation")
        validation_result = _proposal(validation_decision)

        plan = cleaning_plan_agent.propose(df, validation_result, target_column=run.declared_target)
        agent_decision_service.record_decision(
            db, run, agent_name="cleaning_plan", stage="cleaning_plan", decision=plan,
            confidence=0.75, reasoning=cleaning_plan_agent.summarize(plan), auto_approve=False,
        )
        run.status = "awaiting_cleaning_approval"
        db.commit()
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def advance_after_cleaning_approval(pipeline_run_id: str) -> None:
    """Stage 5: Transformation Agent, then pause for HITL 4."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        cleaning_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "cleaning_plan")
        cleaning_plan = _proposal(cleaning_decision)
        plan_fields = cleaning_plan_agent.to_preprocessing_plan_fields(cleaning_plan)

        transformation = transformation_agent.propose(df, plan_fields)
        agent_decision_service.record_decision(
            db, run, agent_name="transformation", stage="transformation", decision=transformation,
            confidence=0.8, reasoning=transformation_agent.summarize(transformation), auto_approve=False,
        )
        run.status = "awaiting_transformation_approval"
        db.commit()
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def advance_after_transformation_approval(pipeline_run_id: str) -> None:
    """Stage 6: Train/Test Split Agent. Classification pauses for HITL 5; clustering has no
    split concept, so it's recorded informationally and the pipeline proceeds straight to
    the algorithm shortlist."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        split = split_agent.recommend(df, run.problem_type, target_column=run.declared_target)
        applicable = run.problem_type == "classification"
        agent_decision_service.record_decision(
            db, run, agent_name="train_test_split", stage="train_test_split", decision=split,
            confidence=0.8 if applicable else 1.0, reasoning=split_agent.summarize(split),
            auto_approve=not applicable,
        )
        if applicable:
            run.status = "awaiting_split_approval"
            db.commit()
        else:
            db.commit()
            _advance_to_algorithm_shortlist(db, run)
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def advance_after_split_approval(pipeline_run_id: str) -> None:
    """Stage 7: Algorithm Shortlist Agent, then pause for HITL 6."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return
        _advance_to_algorithm_shortlist(db, run)
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def _advance_to_algorithm_shortlist(db, run: PipelineRunORM) -> None:
    dataset = db.get(DatasetORM, run.dataset_id)
    df = pd.read_csv(dataset.storage_path)
    profile = data_profiling_agent.analyze(df, dataset.profile_json)

    shortlist = algorithm_shortlist_agent.recommend(profile, run.problem_type)
    agent_decision_service.record_decision(
        db, run, agent_name="algorithm_recommendation", stage="algorithm_recommendation", decision=shortlist,
        confidence=0.7, reasoning=algorithm_shortlist_agent.summarize(shortlist), auto_approve=False,
    )
    run.status = "awaiting_algorithm_approval"
    db.commit()


def advance_after_algorithm_approval(pipeline_run_id: str) -> None:
    """Stage 8-10: build the approved plan, run training, then (classification) HPO, then
    an Evaluation summary, then the Recommendation Agent, then pause for HITL 7."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return

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
            return

        agent_decision_service.record_decision(
            db, run, agent_name="model_selection", stage="training",
            decision={"algorithms_run": sorted({r.algorithm for r in job.runs})},
            confidence=1.0, reasoning=f"Ran all {len(job.runs)} shortlisted {run.problem_type} plugin candidate(s).",
            auto_approve=True,
        )

        if run.problem_type == "classification":
            run.status = "hyperparameter_optimization"
            db.commit()
            _run_hpo(db, run, job, plan_fields)

        db.expire_all()
        run = db.get(PipelineRunORM, pipeline_run_id)
        job = db.get(JobORM, job.id)
        _record_evaluation_summary(db, run, job)

        recommendation = recommendation_agent.build_recommendation(job, db)
        top = recommendation["top_choice"]
        agent_decision_service.record_decision(
            db, run, agent_name="recommendation", stage="recommendation", decision=recommendation,
            confidence=0.85 if recommendation["confidence"] == "high" else 0.5,
            reasoning=f"Recommending {top['algorithm']} (rank #{top['rank']}): {top['rationale']}",
            auto_approve=False,
        )
        run.status = "awaiting_recommendation_approval"
        db.commit()
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()


def _run_hpo(db, run: PipelineRunORM, job: JobORM, plan_fields: dict) -> None:
    dataset = db.get(DatasetORM, run.dataset_id)
    df = pd.read_csv(dataset.storage_path)
    X_train, X_test, y_train, y_test, _report = preprocessing_service.split_target(
        df, run.declared_target, plan_fields, test_size=job.test_size or 0.2, stratify=True,
    )
    algorithms = sorted({r.algorithm for r in job.runs})
    results = []
    for algorithm in algorithms:
        result = hpo_service.optimize(algorithm, X_train.values, y_train.values, X_test.values, y_test.values, job.config_json)
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
        f"{r['algorithm']}: baseline f1={r.get('baseline_metrics', {}).get('f1_macro')} -> "
        f"optimized f1={r.get('optimized_metrics', {}).get('f1_macro')} ({r.get('n_trials', 0)} trial(s))"
        for r in results if not r.get("skipped")
    ) or "No algorithms had a registered search space."
    agent_decision_service.record_decision(
        db, run, agent_name="hyperparameter_optimization", stage="hyperparameter_optimization",
        decision={"results": results}, confidence=1.0, reasoning=reasoning, auto_approve=True,
    )


def _record_evaluation_summary(db, run: PipelineRunORM, job: JobORM) -> None:
    best = min((r for r in job.runs if r.rank), key=lambda r: r.rank, default=None)
    if best is None:
        best = job.runs[0] if job.runs else None
    metrics = best.metrics_json if best else {}
    glossary = metric_glossary.explain(run.problem_type, metrics)
    agent_decision_service.record_decision(
        db, run, agent_name="evaluation", stage="evaluation",
        decision={"champion_algorithm": best.algorithm if best else None, "metrics": metrics, "glossary": glossary},
        confidence=1.0,
        reasoning=f"Top-ranked model so far: {best.algorithm if best else 'none'}. Metrics: {metrics}.",
        auto_approve=True,
    )


def finalize_after_recommendation_approval(pipeline_run_id: str) -> None:
    """Stage 11-12: persist the approved champion as a reloadable prediction pipeline
    (classification only — unlocks the Prediction Playground/Explainability), then the
    Reporting Agent compiles the full decision trail into a business report."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return

        recommendation_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "recommendation")
        recommendation = _proposal(recommendation_decision)
        top = recommendation["top_choice"]

        if run.problem_type == "classification":
            dataset = db.get(DatasetORM, run.dataset_id)
            df = pd.read_csv(dataset.storage_path)
            cleaning_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "cleaning_plan")
            transformation_decision = agent_decision_service.latest_decision(db, pipeline_run_id, "transformation")
            plan_fields = cleaning_plan_agent.to_preprocessing_plan_fields(_proposal(cleaning_decision))
            plan_fields["scaling_method"] = _proposal(transformation_decision).get("scaling_method", plan_fields["scaling_method"])

            champion_run = db.get(ModelRunORM, top["cluster_run_id"])
            built = champion_service.build_and_persist(
                df, plan_fields, run.declared_target, champion_run.algorithm, champion_run.params_json, run.id,
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
        agent_decision_service.record_decision(
            db, run, agent_name="reporting", stage="reporting", decision={"report": report},
            confidence=1.0, reasoning="Compiled the full agent decision trail into a business-readable report.",
            auto_approve=True,
        )
        db.commit()
    except Exception as exc:  # noqa: BLE001
        _fail(db, pipeline_run_id, exc)
        raise
    finally:
        db.close()
