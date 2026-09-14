from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Approval as ApprovalORM
from app.db.models import ClusterRun as ClusterRunORM
from app.db.models import Job as JobORM
from app.schemas.approval import Approval, ApprovalRequest
from app.schemas.leaderboard import Recommendation
from app.services import ranking_service, recommendation_service

router = APIRouter(prefix="/api/v1/jobs", tags=["approval"])


def _get_job_or_404(job_id: str, db: Session) -> JobORM:
    job = db.get(JobORM, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@router.get("/{job_id}/recommendations", response_model=list[Recommendation])
def get_recommendations(job_id: str, db: Session = Depends(get_db)):
    _get_job_or_404(job_id, db)
    top_runs = (
        db.query(ClusterRunORM)
        .filter(ClusterRunORM.job_id == job_id, ClusterRunORM.rank.isnot(None))
        .order_by(ClusterRunORM.rank)
        .limit(3)
        .all()
    )
    if not top_runs:
        raise HTTPException(409, "No ranked runs yet — the job may still be running.")

    recommendations = []
    for run in top_runs:
        run_dict = {
            "algorithm": run.algorithm,
            "params": run.params_json,
            "n_clusters": run.n_clusters,
            "n_noise": run.n_noise,
            "noise_pct": run.noise_pct,
            "silhouette_score": run.silhouette_score,
            "davies_bouldin_score": run.davies_bouldin_score,
            "calinski_harabasz_score": run.calinski_harabasz_score,
            "rank": run.rank,
        }
        strengths, weaknesses = recommendation_service.strengths_and_weaknesses(run_dict)
        labels = np.load(run.labels_path)
        rationale = ranking_service.rationale_for(pd.Series({**run_dict, "run_id": run.id}))
        recommendations.append(
            Recommendation(
                cluster_run=run,
                rationale=rationale,
                strengths=strengths,
                weaknesses=weaknesses,
                cluster_size_breakdown=recommendation_service.cluster_size_breakdown(labels),
            )
        )
    return recommendations


@router.post("/{job_id}/approve", response_model=Approval)
def approve_model(job_id: str, body: ApprovalRequest, db: Session = Depends(get_db)):
    _get_job_or_404(job_id, db)

    run = db.get(ClusterRunORM, body.cluster_run_id)
    if not run or run.job_id != job_id:
        raise HTTPException(404, "Cluster run not found for this job.")

    existing = db.query(ApprovalORM).filter_by(job_id=job_id).first()
    if existing:
        raise HTTPException(409, "This job already has an approved model.")

    approval = ApprovalORM(
        job_id=job_id,
        cluster_run_id=body.cluster_run_id,
        approved_by=body.approved_by,
        notes=body.notes,
    )
    db.add(approval)
    db.commit()
    db.refresh(approval)
    return approval
