"""End-to-end coverage for the LangGraph conversion of the pipeline orchestrator.

Drives `orchestrator.py`'s public API directly (same style as `test_new_agents.py`), against
the app's real SQLite DB (there is no test-DB isolation fixture in this project), and cleans
up every row/file it creates in a `finally` block. The specific regression this guards
against is graph-node replay creating duplicate `AgentDecision` rows on resume — see
`pipeline_graph.py`'s module docstring for why gate nodes exist.
"""
from __future__ import annotations

import uuid

import pandas as pd
import pytest

from app.agents import orchestrator
from app.db.database import SessionLocal
from app.db.models import AgentDecision as AgentDecisionORM
from app.db.models import Dataset as DatasetORM
from app.db.models import Job as JobORM
from app.db.models import ModelRun as ModelRunORM
from app.db.models import PipelineRun as PipelineRunORM
from app.db.models import PreprocessingPlanORM


def _classification_df(n=60):
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "age": [20 + (i % 40) for i in range(n)],
            "income": [30000.0 + i * 137.5 for i in range(n)],
            "region": (["north", "south", "east", "west"] * (n // 4 + 1))[:n],
            "churn": (["yes", "no"] * (n // 2))[:n],
        }
    )


def _clustering_df(n=60):
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "age": [20 + (i % 40) for i in range(n)],
            "income": [30000.0 + i * 137.5 for i in range(n)],
            "region": (["north", "south", "east", "west"] * (n // 4 + 1))[:n],
        }
    )


@pytest.fixture
def dataset_factory(tmp_path):
    db = SessionLocal()
    created_ids = []

    def _make(df: pd.DataFrame, target_column: str | None) -> PipelineRunORM:
        csv_path = tmp_path / f"{uuid.uuid4()}.csv"
        df.to_csv(csv_path, index=False)

        dataset = DatasetORM(
            filename=csv_path.name, storage_path=str(csv_path),
            n_rows=len(df), n_columns=len(df.columns), data_quality_score=90.0,
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        run = PipelineRunORM(dataset_id=dataset.id, declared_target=target_column, status="profiling")
        db.add(run)
        db.commit()
        db.refresh(run)

        created_ids.append((dataset.id, run.id))
        return run

    yield _make

    for dataset_id, run_id in created_ids:
        run = db.get(PipelineRunORM, run_id)
        if run:
            db.query(AgentDecisionORM).filter_by(pipeline_run_id=run_id).delete()
            if run.job_id:
                job = db.get(JobORM, run.job_id)
                if job:
                    db.query(ModelRunORM).filter_by(job_id=job.id).delete()
                    if job.preprocessing_plan_id:
                        plan = db.get(PreprocessingPlanORM, job.preprocessing_plan_id)
                        if plan:
                            db.delete(plan)
                    db.delete(job)
            db.delete(run)
        dataset = db.get(DatasetORM, dataset_id)
        if dataset:
            db.delete(dataset)
        db.commit()
    db.close()


def _approve_latest_pending(db, pipeline_run_id: str) -> AgentDecisionORM:
    decision = (
        db.query(AgentDecisionORM)
        .filter_by(pipeline_run_id=pipeline_run_id, status="proposed")
        .order_by(AgentDecisionORM.created_at.desc())
        .first()
    )
    assert decision is not None, "expected a pending decision to approve"
    decision.status = "approved"
    db.commit()
    return decision


def _decision_counts(db, pipeline_run_id: str) -> dict[str, int]:
    rows = db.query(AgentDecisionORM).filter_by(pipeline_run_id=pipeline_run_id).all()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.agent_name] = counts.get(r.agent_name, 0) + 1
    return counts


def test_classification_run_walks_every_gate_with_no_duplicate_decisions(dataset_factory):
    run = dataset_factory(_classification_df(), "churn")
    db = SessionLocal()
    try:
        orchestrator.start(run.id)
        db.expire_all()
        run = db.get(PipelineRunORM, run.id)
        assert run.status == "awaiting_problem_approval"

        _approve_latest_pending(db, run.id)
        orchestrator.advance_after_problem_approval(run.id)
        db.expire_all()
        run = db.get(PipelineRunORM, run.id)
        assert run.status in ("awaiting_validation_approval", "awaiting_cleaning_approval")

        # Walk forward, approving whatever is currently pending, until the run leaves every
        # awaiting_*_approval status — mirrors what the router's review endpoint does one
        # decision at a time, just driven directly instead of through HTTP.
        dispatch = {
            "data_validation": orchestrator.advance_after_validation_approval,
            "cleaning_plan": orchestrator.advance_after_cleaning_approval,
            "transformation": orchestrator.advance_after_transformation_approval,
            "train_test_split": orchestrator.advance_after_split_approval,
            "algorithm_recommendation": orchestrator.advance_after_algorithm_approval,
            "recommendation": orchestrator.finalize_after_recommendation_approval,
        }
        for _ in range(10):
            db.expire_all()
            run = db.get(PipelineRunORM, run.id)
            if run.status in ("completed", "failed"):
                break
            pending = (
                db.query(AgentDecisionORM)
                .filter_by(pipeline_run_id=run.id, status="proposed")
                .order_by(AgentDecisionORM.created_at.desc())
                .first()
            )
            assert pending is not None, f"no pending decision but status is {run.status}"
            agent_name = pending.agent_name
            pending.status = "approved"
            db.commit()
            dispatch[agent_name](run.id)

        db.expire_all()
        run = db.get(PipelineRunORM, run.id)
        assert run.status == "completed", run.error_message

        counts = _decision_counts(db, run.id)
        assert all(c == 1 for c in counts.values()), counts
        for expected_agent in (
            "data_profiling", "problem_detection", "business_framing", "data_validation",
            "cleaning_plan", "transformation", "train_test_split", "algorithm_recommendation",
            "model_selection", "hyperparameter_optimization", "evaluation", "recommendation", "reporting",
        ):
            assert expected_agent in counts, f"missing decision for {expected_agent}"
    finally:
        db.close()


def test_clustering_run_skips_split_hitl(dataset_factory):
    run = dataset_factory(_clustering_df(), None)
    db = SessionLocal()
    try:
        orchestrator.start(run.id)
        _approve_latest_pending(db, run.id)
        orchestrator.advance_after_problem_approval(run.id)

        db.expire_all()
        run = db.get(PipelineRunORM, run.id)
        # Clustering has no train_test_split HITL — after problem approval it should never
        # land on awaiting_split_approval no matter how far the auto-approve chain runs.
        assert run.status != "awaiting_split_approval"
    finally:
        db.close()
