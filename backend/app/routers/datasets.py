from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import DATASETS_DIR
from app.db.database import get_db
from app.db.models import AgentDecision, Approval, ClusterInterpretation, ModelRun
from app.db.models import Dataset as DatasetORM
from app.db.models import ExportArtifact, Job, PipelineRun, PreprocessingPlanORM
from app.schemas.dataset import DataProfile, Dataset
from app.services import profiling_service

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.post("", response_model=Dataset)
async def upload_dataset(file: UploadFile, db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    dataset_id = str(uuid.uuid4())
    dest = DATASETS_DIR / f"{dataset_id}.csv"
    contents = await file.read()
    dest.write_bytes(contents)

    try:
        df = pd.read_csv(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    profile = profiling_service.build_profile(df)

    dataset = DatasetORM(
        id=dataset_id,
        filename=file.filename,
        storage_path=str(dest),
        n_rows=profile["n_rows"],
        n_columns=profile["n_columns"],
        data_quality_score=profile["data_quality_score"],
        profile_json=profile,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("", response_model=list[Dataset])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(DatasetORM).order_by(DatasetORM.uploaded_at.desc()).all()


@router.get("/{dataset_id}", response_model=Dataset)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.get(DatasetORM, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    return dataset


@router.get("/{dataset_id}/profile", response_model=DataProfile)
def get_profile(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.get(DatasetORM, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    return dataset.profile_json


@router.delete("/{dataset_id}", status_code=204, response_model=None)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.get(DatasetORM, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")

    for job in db.query(Job).filter(Job.dataset_id == dataset_id).all():
        db.query(ExportArtifact).filter(ExportArtifact.job_id == job.id).delete()
        db.query(Approval).filter(Approval.job_id == job.id).delete()
        for run in db.query(ModelRun).filter(ModelRun.job_id == job.id).all():
            db.query(ClusterInterpretation).filter(
                ClusterInterpretation.cluster_run_id == run.id
            ).delete()
            db.delete(run)
        db.delete(job)

    for pipeline_run in db.query(PipelineRun).filter(PipelineRun.dataset_id == dataset_id).all():
        db.query(AgentDecision).filter(AgentDecision.pipeline_run_id == pipeline_run.id).delete()
        db.delete(pipeline_run)

    db.query(PreprocessingPlanORM).filter(PreprocessingPlanORM.dataset_id == dataset_id).delete()

    db.delete(dataset)
    db.commit()

    Path(dataset.storage_path).unlink(missing_ok=True)
