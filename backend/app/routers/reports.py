from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import EXPORTS_DIR, JOBS_DIR
from app.db.database import get_db
from app.db.models import ClusterInterpretation as ClusterInterpretationORM
from app.db.models import ExportArtifact, Job as JobORM
from app.db.models import ModelRun as ModelRunORM
from app.services import export_service
from app.routers.visualizations import _resolve_run

router = APIRouter(prefix="/api/v1/jobs", tags=["reports"])

MEDIA_TYPES = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pdf": "application/pdf",
}


@router.get("/{job_id}/export")
def export_job(
    job_id: str,
    format: str = Query(..., pattern="^(csv|xlsx|pdf)$"),
    run_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    job = db.get(JobORM, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    run = _resolve_run(job_id, run_id, db)

    job_dir = JOBS_DIR / job_id
    cleaned_df = pd.read_csv(job_dir / "cleaned.csv")
    labels = np.load(run.labels_path)

    interpretation_orm = db.query(ClusterInterpretationORM).filter_by(cluster_run_id=run.id).first()
    interpretation = (
        {
            "profiles": interpretation_orm.profiles_json,
            "feature_importance": interpretation_orm.feature_importance_json,
            "summaries": interpretation_orm.summaries_json,
            "suggested_names": interpretation_orm.suggested_names_json,
        }
        if interpretation_orm
        else {"profiles": [], "feature_importance": [], "summaries": {}, "suggested_names": {}}
    )

    clustered_df = cleaned_df.copy()
    clustered_df["cluster"] = ["noise" if l == -1 else str(int(l)) for l in labels]
    clustered_df["cluster_name"] = clustered_df["cluster"].map(interpretation["suggested_names"]).fillna(
        clustered_df["cluster"]
    )

    all_runs = db.query(ModelRunORM).filter_by(job_id=job_id).order_by(ModelRunORM.rank.is_(None), ModelRunORM.rank).all()
    leaderboard_df = pd.DataFrame(
        [
            {
                "algorithm": r.algorithm,
                "params": r.params_json,
                "n_clusters": r.n_clusters,
                "n_noise": r.n_noise,
                "silhouette_score": r.silhouette_score,
                "davies_bouldin_score": r.davies_bouldin_score,
                "calinski_harabasz_score": r.calinski_harabasz_score,
                "composite_score": r.composite_score,
                "rank": r.rank,
            }
            for r in all_runs
        ]
    )

    export_id = str(uuid.uuid4())
    out_dir = EXPORTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        out_path = out_dir / f"{export_id}.csv"
        export_service.export_csv(clustered_df, out_path)
    elif format == "xlsx":
        out_path = out_dir / f"{export_id}.xlsx"
        export_service.export_excel(leaderboard_df, clustered_df, interpretation, out_path)
    else:
        out_path = out_dir / f"{export_id}.pdf"
        chosen_model_info = {
            "algorithm": run.algorithm,
            "params": run.params_json,
            "n_clusters": run.n_clusters,
            "n_noise": run.n_noise,
            "silhouette_score": run.silhouette_score,
            "davies_bouldin_score": run.davies_bouldin_score,
            "calinski_harabasz_score": run.calinski_harabasz_score,
        }
        export_service.export_pdf(chosen_model_info, leaderboard_df, interpretation, out_path)

    db.add(ExportArtifact(job_id=job_id, format=format, storage_path=str(out_path)))
    db.commit()

    return FileResponse(
        path=out_path,
        media_type=MEDIA_TYPES[format],
        filename=f"clustering_results_{job_id[:8]}.{format}",
    )
