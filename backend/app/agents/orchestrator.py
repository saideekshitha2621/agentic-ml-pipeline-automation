"""Pipeline Orchestrator.

Pure sequencing/state-machine logic — contains no modeling logic itself. Each function
opens its own DB session (it runs inside FastAPI BackgroundTasks, same pattern as
`job_runner_service.run_job`) and advances `pipeline_runs.status` by exactly one stage,
persisting an AgentDecision at each step via `agent_decision_service`.

State graph:
  profiling -> awaiting_problem_approval -[HITL gate 1]->
  model_execution -> awaiting_recommendation_approval -[HITL gate 2]->
  reporting -> completed
  (any stage -> failed, with error_message)

When the confirmed problem type is 'clustering', this delegates to the existing,
unmodified `clustering_service` / `job_runner_service` / `jobs` / `cluster_runs` tables —
the orchestrator just creates a Job row and calls `job_runner_service.run_job()` exactly as
`POST /jobs` does today, then reads back the result the same way `GET /jobs/{id}` does.
"""
from __future__ import annotations

import pandas as pd

from app.db.database import SessionLocal
from app.db.models import Dataset as DatasetORM
from app.db.models import Job as JobORM
from app.db.models import PipelineRun as PipelineRunORM
from app.db.models import PreprocessingPlanORM
from app.services import agent_decision_service, job_runner_service, preprocessing_service
from app.agents import data_profiling_agent, problem_detection_agent, recommendation_agent, reporting_agent

SUPPORTED_PROBLEM_TYPES = {"clustering"}  # classification/regression are a later phase


def start(pipeline_run_id: str) -> None:
    """Stage 1-2: Data Profiling Agent -> Problem Detection Agent, then pause for HITL gate 1."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return
        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        profile = data_profiling_agent.analyze(df, dataset.profile_json)
        agent_decision_service.record_decision(
            db,
            run,
            agent_name="data_profiling",
            stage="profiling",
            decision={"column_roles": profile["column_roles"]},
            confidence=1.0,
            reasoning=data_profiling_agent.summarize(profile),
            auto_approve=True,
        )

        result = problem_detection_agent.detect(df, profile, declared_target=run.declared_target)
        agent_decision_service.record_decision(
            db,
            run,
            agent_name="problem_detection",
            stage="problem_detection",
            decision=result,
            confidence=result["confidence"],
            reasoning=" ".join(result["reasoning"]),
            auto_approve=False,
        )
        run.status = "awaiting_problem_approval"
        db.commit()
    except Exception as exc:  # noqa: BLE001 - surface any failure on the run
        db.rollback()
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run:
            run.status = "failed"
            run.error_message = str(exc)
            db.commit()
        raise
    finally:
        db.close()


def advance_after_problem_approval(pipeline_run_id: str) -> None:
    """Stage 3-4: apply preprocessing + run the model-execution step for the confirmed
    problem type, then Evaluation (via the existing ranking) + Recommendation Agent,
    then pause for HITL gate 2."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return

        decision = agent_decision_service.latest_decision(db, pipeline_run_id, "problem_detection")
        proposal = decision.human_edits_json if decision.status == "edited" else decision.decision_json
        problem_type = proposal["problem_type"]
        run.problem_type = problem_type

        if problem_type not in SUPPORTED_PROBLEM_TYPES:
            run.status = "failed"
            run.error_message = (
                f"Problem type '{problem_type}' was confirmed, but this build only executes the "
                "'clustering' path end-to-end (classification/regression are a later phase)."
            )
            db.commit()
            return

        dataset = db.get(DatasetORM, run.dataset_id)
        df = pd.read_csv(dataset.storage_path)

        default_plan = preprocessing_service.detect_default_plan(df)
        plan_orm = PreprocessingPlanORM(dataset_id=run.dataset_id, **default_plan)
        db.add(plan_orm)
        db.commit()
        db.refresh(plan_orm)

        agent_decision_service.record_decision(
            db,
            run,
            agent_name="preprocessing",
            stage="preprocessing",
            decision=default_plan,
            confidence=0.7,
            reasoning=(
                "Auto-detected default preprocessing plan: median/mode imputation, standard "
                "scaling, id-like columns excluded. (Preprocessing review gate is not enforced "
                "in this build — the plan is applied directly.)"
            ),
            auto_approve=True,
        )

        job = JobORM(
            dataset_id=run.dataset_id,
            preprocessing_plan_id=plan_orm.id,
            config_json={},
            status="queued",
            log_lines=["Job queued by agent orchestrator."],
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        run.job_id = job.id
        run.status = "model_execution"
        db.commit()

        job_runner_service.run_job(job.id)  # reuses the existing, unmodified clustering pipeline

        db.expire_all()
        job = db.get(JobORM, job.id)
        if job.status != "completed":
            run.status = "failed"
            run.error_message = job.error_message or "Clustering job failed."
            db.commit()
            return

        agent_decision_service.record_decision(
            db,
            run,
            agent_name="model_selection",
            stage="model_execution",
            decision={"algorithms_run": sorted({r.algorithm for r in job.runs})},
            confidence=1.0,
            reasoning=f"Ran all {len(job.runs)} registered clustering plugin candidate(s) via the existing clustering_service.",
            auto_approve=True,
        )

        recommendation = recommendation_agent.build_recommendation(job, db)
        top = recommendation["top_choice"]
        agent_decision_service.record_decision(
            db,
            run,
            agent_name="recommendation",
            stage="recommendation",
            decision=recommendation,
            confidence=0.85 if recommendation["confidence"] == "high" else 0.5,
            reasoning=f"Recommending {top['algorithm']} (rank #{top['rank']}): {top['rationale']}",
            auto_approve=False,
        )
        run.status = "awaiting_recommendation_approval"
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run:
            run.status = "failed"
            run.error_message = str(exc)
            db.commit()
        raise
    finally:
        db.close()


def finalize_after_recommendation_approval(pipeline_run_id: str) -> None:
    """Stage 5: Reporting Agent compiles the full decision trail into a business report."""
    db = SessionLocal()
    try:
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run is None:
            return
        run.status = "reporting"
        db.commit()

        dataset = db.get(DatasetORM, run.dataset_id)
        job = db.get(JobORM, run.job_id) if run.job_id else None
        decisions = list(run.decisions)

        run.status = "completed"  # set before building the report so final_outcome reflects it
        report = reporting_agent.build_report(run, decisions, dataset, job)
        agent_decision_service.record_decision(
            db,
            run,
            agent_name="reporting",
            stage="reporting",
            decision={"report": report},
            confidence=1.0,
            reasoning="Compiled the full agent decision trail into a business-readable report.",
            auto_approve=True,
        )
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        run = db.get(PipelineRunORM, pipeline_run_id)
        if run:
            run.status = "failed"
            run.error_message = str(exc)
            db.commit()
        raise
    finally:
        db.close()
