from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Dataset as DatasetORM
from app.db.models import Job as JobORM
from app.db.models import PreprocessingPlanORM
from app.schemas.job import Job, JobCreateRequest
from app.services.job_runner_service import run_job

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.post("", response_model=Job)
def create_job(body: JobCreateRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    dataset = db.get(DatasetORM, body.dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    plan = db.get(PreprocessingPlanORM, body.preprocessing_plan_id)
    if not plan or plan.dataset_id != body.dataset_id:
        raise HTTPException(404, "Preprocessing plan not found for this dataset.")

    job = JobORM(
        dataset_id=body.dataset_id,
        preprocessing_plan_id=body.preprocessing_plan_id,
        config_json=body.config,
        status="queued",
        log_lines=["Job queued."],
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(run_job, job.id)
    return job


@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(JobORM, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job
