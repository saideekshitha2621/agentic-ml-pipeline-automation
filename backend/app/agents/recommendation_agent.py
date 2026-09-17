"""Recommendation Agent.

Extends the existing `recommendation_service` (strengths/weaknesses per run) and
`ranking_service` (composite score + rationale) — both unchanged — with comparative
"why not chosen" text for runner-ups, and an overall confidence signal for the HITL
reviewer (a near-tie between #1 and #2 deserves more scrutiny than a clear winner).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Job as JobORM
from app.db.models import ModelRun as ModelRunORM
from app.services import ranking_service, recommendation_service

NEAR_TIE_COMPOSITE_DELTA = 0.05


def _run_dict(run: ModelRunORM) -> dict:
    base = {
        "algorithm": run.algorithm,
        "params": run.params_json,
        "composite_score": run.composite_score,
        "rank": run.rank,
    }
    if run.problem_type == "clustering":
        base.update(
            {
                "n_clusters": run.n_clusters,
                "n_noise": run.n_noise,
                "noise_pct": run.noise_pct,
                "silhouette_score": run.silhouette_score,
                "davies_bouldin_score": run.davies_bouldin_score,
                "calinski_harabasz_score": run.calinski_harabasz_score,
            }
        )
    else:
        base.update(run.metrics_json or {})
    return base


def build_recommendation(job: JobORM, db: Session) -> dict:
    top_runs = (
        db.query(ModelRunORM)
        .filter(ModelRunORM.job_id == job.id, ModelRunORM.rank.isnot(None))
        .order_by(ModelRunORM.rank)
        .limit(3)
        .all()
    )
    if not top_runs:
        raise ValueError("No ranked cluster runs available for this job yet.")

    entries = []
    for run in top_runs:
        run_dict = _run_dict(run)
        strengths, weaknesses = recommendation_service.strengths_and_weaknesses(
            run_dict, problem_type=run.problem_type
        )
        rationale = ranking_service.rationale_for(
            pd.Series({**run_dict, "run_id": run.id}), problem_type=run.problem_type
        )
        # labels_path holds cluster labels for clustering runs, predicted classes for
        # classification runs — either way, "the per-row output array this run produced".
        outputs = np.load(run.labels_path)
        breakdown = (
            recommendation_service.cluster_size_breakdown(outputs)
            if run.problem_type == "clustering"
            else recommendation_service.prediction_class_breakdown(outputs)
        )
        entries.append(
            {
                "cluster_run_id": run.id,
                "algorithm": run.algorithm,
                "params": run.params_json,
                "rank": run.rank,
                "composite_score": run.composite_score,
                "rationale": rationale,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "cluster_size_breakdown": breakdown,
            }
        )

    top = entries[0]
    alternatives = []
    for alt in entries[1:]:
        deltas = []
        if top["composite_score"] is not None and alt["composite_score"] is not None:
            diff = top["composite_score"] - alt["composite_score"]
            deltas.append(f"composite score {diff:+.3f} lower than the recommended model")
        why_not = (
            f"{alt['algorithm']} (#{alt['rank']}): " + "; ".join(deltas)
            if deltas
            else f"{alt['algorithm']} ranked #{alt['rank']}."
        )
        alternatives.append({**alt, "why_not_chosen": why_not})

    confidence = "high"
    if len(entries) > 1:
        top_score = top["composite_score"] or 0.0
        second_score = entries[1]["composite_score"] or 0.0
        if (top_score - second_score) < NEAR_TIE_COMPOSITE_DELTA:
            confidence = "low"

    return {"top_choice": top, "alternatives": alternatives, "confidence": confidence}
