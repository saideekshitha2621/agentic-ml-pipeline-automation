"""Orchestrates preprocessing -> PCA -> clustering -> evaluation -> ranking for one job.

Runs inside a FastAPI BackgroundTasks thread, so it opens its own DB session rather than
reusing the request-scoped one. Progress/log lines are written to the Job row so the
Model Execution screen can poll GET /jobs/{id}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import JOBS_DIR, LABELS_DIR
from app.db.database import SessionLocal
from app.db.models import ClusterRun, Job, PreprocessingPlanORM
from app.services import clustering_service, evaluation_service, pca_service, preprocessing_service, ranking_service


def _plan_to_dict(plan: PreprocessingPlanORM) -> dict:
    return {
        "numerical_columns": plan.numerical_columns,
        "categorical_columns": plan.categorical_columns,
        "dropped_columns": plan.dropped_columns,
        "numerical_impute_strategy": plan.numerical_impute_strategy,
        "categorical_impute_strategy": plan.categorical_impute_strategy,
        "scaling_method": plan.scaling_method,
        "drop_duplicates": plan.drop_duplicates,
        "pca_enabled": plan.pca_enabled,
        "pca_variance_target": plan.pca_variance_target,
    }


def run_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = "running"
        job.log_lines = ["Job started."]
        db.commit()

        def log(pct: float, message: str):
            job.progress_pct = round(pct, 1)
            job.log_lines = job.log_lines + [message]
            db.commit()

        dataset = job.dataset
        plan_orm = db.get(PreprocessingPlanORM, job.preprocessing_plan_id)
        plan = _plan_to_dict(plan_orm)

        log(2, "Loading dataset...")
        df = pd.read_csv(dataset.storage_path)

        log(5, "Applying preprocessing plan...")
        cleaned_df, encoded_df, _report = preprocessing_service.apply_plan(df, plan)

        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(job_dir / "cleaned.csv", index=False)
        encoded_df.to_csv(job_dir / "encoded.csv", index=False)

        model_df = encoded_df
        pca_report: dict = {}
        if plan["pca_enabled"]:
            log(10, "Applying PCA...")
            model_df, pca_report = pca_service.apply(encoded_df, plan["pca_variance_target"])
            job.pca_report_json = pca_report
            model_df.to_csv(job_dir / "model.csv", index=False)
            db.commit()

        X = model_df.values

        log(15, "Running clustering algorithms...")

        def progress_cb(pct: float, message: str):
            log(15 + pct * 0.7, message)

        plugin_runs = clustering_service.run_all(X, job.config_json, progress_cb=progress_cb)
        if not plugin_runs:
            job.status = "failed"
            job.error_message = "No algorithm produced a usable clustering (>=2 clusters) for this data."
            db.commit()
            return

        log(88, f"Evaluating {len(plugin_runs)} clustering run(s)...")
        run_label_dir = LABELS_DIR / job_id
        run_label_dir.mkdir(parents=True, exist_ok=True)

        evaluated_rows = []
        orm_runs = []
        for i, run in enumerate(plugin_runs):
            metrics = evaluation_service.evaluate(X, run)
            labels_path = run_label_dir / f"{i}.npy"
            np.save(labels_path, run.labels)

            orm_run = ClusterRun(
                job_id=job_id,
                algorithm=run.algorithm,
                params_json=run.params,
                extra_json=run.extra,
                n_clusters=run.n_clusters,
                n_noise=run.n_noise,
                silhouette_score=metrics["silhouette_score"],
                davies_bouldin_score=metrics["davies_bouldin_score"],
                calinski_harabasz_score=metrics["calinski_harabasz_score"],
                labels_path=str(labels_path),
            )
            db.add(orm_run)
            orm_runs.append(orm_run)
            evaluated_rows.append(
                {
                    "run_id": i,
                    "algorithm": run.algorithm,
                    "params": run.params,
                    "n_clusters": run.n_clusters,
                    "n_noise": run.n_noise,
                    **metrics,
                }
            )
        db.flush()

        log(94, "Ranking models...")
        leaderboard = ranking_service.rank(evaluated_rows)
        rank_by_index = {int(r["run_id"]): r for r in leaderboard.to_dict("records")}
        for i, orm_run in enumerate(orm_runs):
            lb_row = rank_by_index.get(i)
            if lb_row:
                orm_run.composite_score = lb_row.get("composite_score")
                orm_run.rank = lb_row.get("rank")
                orm_run.noise_pct = lb_row.get("noise_pct") or 0.0

        job.status = "completed"
        job.progress_pct = 100.0
        job.log_lines = job.log_lines + ["Job completed."]
        db.commit()
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        db.rollback()
        job = db.get(Job, job_id)
        if job:
            job.status = "failed"
            job.error_message = str(exc)
            db.commit()
        raise
    finally:
        db.close()
