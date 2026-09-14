from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ClusterRun as ClusterRunORM
from app.db.models import Job as JobORM
from app.schemas.leaderboard import ClusterRun

router = APIRouter(prefix="/api/v1/jobs", tags=["leaderboard"])


def _get_job_or_404(job_id: str, db: Session) -> JobORM:
    job = db.get(JobORM, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@router.get("/{job_id}/leaderboard", response_model=list[ClusterRun])
def get_leaderboard(job_id: str, db: Session = Depends(get_db)):
    _get_job_or_404(job_id, db)
    return (
        db.query(ClusterRunORM)
        .filter_by(job_id=job_id)
        .order_by(ClusterRunORM.rank.is_(None), ClusterRunORM.rank)
        .all()
    )


@router.get("/{job_id}/compare", response_model=list[ClusterRun])
def compare_runs(job_id: str, run_ids: str = Query(..., description="comma-separated run ids"), db: Session = Depends(get_db)):
    _get_job_or_404(job_id, db)
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    runs = db.query(ClusterRunORM).filter(ClusterRunORM.job_id == job_id, ClusterRunORM.id.in_(ids)).all()
    if len(runs) != len(ids):
        raise HTTPException(404, "One or more run_ids not found for this job.")
    return runs
