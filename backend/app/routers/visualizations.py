from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sklearn.decomposition import PCA
from sqlalchemy.orm import Session

from app.core.config import JOBS_DIR
from app.db.database import get_db
from app.db.models import Approval as ApprovalORM
from app.db.models import ClusterInterpretation as ClusterInterpretationORM
from app.db.models import Job as JobORM
from app.db.models import ModelRun as ModelRunORM
from app.schemas.visualization import ClusterInterpretation, VisualizationBundle
from app.services import explanation_service

router = APIRouter(prefix="/api/v1/jobs", tags=["visualizations"])


def _resolve_run(job_id: str, run_id: str | None, db: Session) -> ModelRunORM:
    if run_id:
        run = db.get(ModelRunORM, run_id)
        if not run or run.job_id != job_id:
            raise HTTPException(404, "Cluster run not found for this job.")
        return run

    approval = db.query(ApprovalORM).filter_by(job_id=job_id).first()
    if approval:
        return db.get(ModelRunORM, approval.cluster_run_id)

    top = (
        db.query(ModelRunORM)
        .filter(ModelRunORM.job_id == job_id, ModelRunORM.rank.isnot(None))
        .order_by(ModelRunORM.rank)
        .first()
    )
    if not top:
        raise HTTPException(409, "No ranked runs available yet.")
    return top


@router.get("/{job_id}/visualizations", response_model=VisualizationBundle)
def get_visualizations(job_id: str, run_id: str | None = Query(default=None), db: Session = Depends(get_db)):
    job = db.get(JobORM, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    run = _resolve_run(job_id, run_id, db)
    labels = np.load(run.labels_path)

    job_dir = JOBS_DIR / job_id
    model_path = job_dir / "model.csv" if (job_dir / "model.csv").exists() else job_dir / "encoded.csv"
    model_df = pd.read_csv(model_path)

    coords = PCA(n_components=2, random_state=42).fit_transform(model_df.values) if model_df.shape[1] > 2 else model_df.values
    scatter = [
        {"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4),
         "cluster": "noise" if labels[i] == -1 else str(int(labels[i]))}
        for i in range(len(labels))
    ]

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    cluster_sizes = [
        {"cluster": "noise" if u == -1 else str(int(u)), "count": int(c), "pct_of_total": round(c / total * 100, 2)}
        for u, c in zip(unique, counts)
    ]

    all_runs = db.query(ModelRunORM).filter_by(job_id=job_id).all()
    metric_comparison = [
        {
            "run_id": r.id,
            "algorithm": r.algorithm,
            "silhouette_score": r.silhouette_score,
            "davies_bouldin_score": r.davies_bouldin_score,
            "calinski_harabasz_score": r.calinski_harabasz_score,
        }
        for r in all_runs
    ]

    return {"scatter": scatter, "cluster_sizes": cluster_sizes, "metric_comparison": metric_comparison}


@router.get("/{job_id}/interpretation", response_model=ClusterInterpretation)
def get_interpretation(job_id: str, run_id: str | None = Query(default=None), db: Session = Depends(get_db)):
    job = db.get(JobORM, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    run = _resolve_run(job_id, run_id, db)

    cached = db.query(ClusterInterpretationORM).filter_by(cluster_run_id=run.id).first()
    if cached:
        return {
            "profiles": cached.profiles_json,
            "feature_importance": cached.feature_importance_json,
            "summaries": cached.summaries_json,
            "suggested_names": cached.suggested_names_json,
        }

    job_dir = JOBS_DIR / job_id
    cleaned_df = pd.read_csv(job_dir / "cleaned.csv")
    encoded_df = pd.read_csv(job_dir / "encoded.csv")
    labels = np.load(run.labels_path)

    result = explanation_service.explain(cleaned_df, encoded_df, labels)

    db.add(
        ClusterInterpretationORM(
            cluster_run_id=run.id,
            profiles_json=result["profiles"],
            feature_importance_json=result["feature_importance"],
            summaries_json=result["summaries"],
            suggested_names_json=result["suggested_names"],
        )
    )
    db.commit()
    return result
