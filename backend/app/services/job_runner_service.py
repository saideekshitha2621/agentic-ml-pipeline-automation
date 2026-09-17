"""Orchestrates preprocessing -> model execution -> evaluation -> ranking for one job.

Runs inside a FastAPI BackgroundTasks thread, so it opens its own DB session rather than
reusing the request-scoped one. Progress/log lines are written to the Job row so the
Model Execution screen can poll GET /jobs/{id}.

run_job() dispatches on job.problem_type: clustering keeps its original preprocess -> PCA
-> clustering_service -> evaluate -> rank shape verbatim (_run_clustering); classification
takes a leakage-safe train/test split instead of PCA and has no noise/cluster-count concept
(_run_classification). Both converge on the same ModelRun row shape and the same
ranking_service.rank(problem_type=...) call.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import JOBS_DIR, LABELS_DIR, PREDICTIONS_DIR
from app.db.database import SessionLocal
from app.db.models import Job, ModelRun, PreprocessingPlanORM
from app.services import (
    classification_service,
    clustering_service,
    evaluation_service,
    pca_service,
    preprocessing_service,
    ranking_service,
)


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
        "column_actions": plan.column_actions,
    }


def _run_clustering(db, job: Job, df: pd.DataFrame, plan: dict, job_dir: Path, log) -> None:
    log(5, "Applying preprocessing plan...")
    cleaned_df, encoded_df, _report = preprocessing_service.apply_plan(df, plan)
    cleaned_df.to_csv(job_dir / "cleaned.csv", index=False)
    encoded_df.to_csv(job_dir / "encoded.csv", index=False)

    model_df = encoded_df
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

    def on_algorithm_status(name: str, status: str):
        job.training_status_json = {**job.training_status_json, name: status}
        db.commit()

    plugin_runs = clustering_service.run_all(
        X, job.config_json, progress_cb=progress_cb,
        algorithms=job.selected_algorithms or None, on_algorithm_status=on_algorithm_status,
    )
    if not plugin_runs:
        job.status = "failed"
        job.error_message = "No algorithm produced a usable clustering (>=2 clusters) for this data."
        db.commit()
        return

    log(88, f"Evaluating {len(plugin_runs)} clustering run(s)...")
    run_label_dir = LABELS_DIR / job.id
    run_label_dir.mkdir(parents=True, exist_ok=True)

    evaluated_rows = []
    orm_runs = []
    for i, run in enumerate(plugin_runs):
        metrics = evaluation_service.evaluate(X, run)
        labels_path = run_label_dir / f"{i}.npy"
        np.save(labels_path, run.labels)

        orm_run = ModelRun(
            job_id=job.id,
            problem_type="clustering",
            algorithm=run.algorithm,
            params_json=run.params,
            extra_json=run.extra,
            n_clusters=run.n_clusters,
            n_noise=run.n_noise,
            silhouette_score=metrics["silhouette_score"],
            davies_bouldin_score=metrics["davies_bouldin_score"],
            calinski_harabasz_score=metrics["calinski_harabasz_score"],
            labels_path=str(labels_path),
            metrics_json={
                "silhouette": metrics["silhouette_score"],
                "davies_bouldin": metrics["davies_bouldin_score"],
                "calinski_harabasz": metrics["calinski_harabasz_score"],
            },
            artifacts_json={"labels_path": str(labels_path)},
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
    leaderboard = ranking_service.rank(evaluated_rows, problem_type="clustering")
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


def _run_classification(db, job: Job, df: pd.DataFrame, plan: dict, job_dir: Path, log) -> None:
    if not job.target_column:
        job.status = "failed"
        job.error_message = "No target_column set on this job — classification needs a target to predict."
        db.commit()
        return

    log(5, "Splitting into train/test (leakage-safe)...")
    X_train, X_test, y_train, y_test, split_report = preprocessing_service.split_target(
        df, job.target_column, plan, test_size=job.test_size or 0.2, stratify=True
    )
    X_train.to_csv(job_dir / "train_features.csv", index=False)
    X_test.to_csv(job_dir / "test_features.csv", index=False)

    log(15, "Running classification algorithms...")

    def progress_cb(pct: float, message: str):
        log(15 + pct * 0.7, message)

    def on_algorithm_status(name: str, status: str):
        job.training_status_json = {**job.training_status_json, name: status}
        db.commit()

    plugin_runs = classification_service.run_all(
        X_train.values, y_train.values, X_test.values, job.config_json, progress_cb=progress_cb,
        algorithms=job.selected_algorithms or None, on_algorithm_status=on_algorithm_status,
    )
    if not plugin_runs:
        job.status = "failed"
        job.error_message = "No algorithm produced a usable model for this data."
        db.commit()
        return

    log(88, f"Evaluating {len(plugin_runs)} classification run(s)...")
    run_pred_dir = PREDICTIONS_DIR / job.id
    run_pred_dir.mkdir(parents=True, exist_ok=True)

    evaluated_rows = []
    orm_runs = []
    y_test_values = y_test.values
    for i, run in enumerate(plugin_runs):
        metrics = evaluation_service.evaluate_classification(y_test_values, run)
        preds_path = run_pred_dir / f"{i}.npy"
        # y_pred mirrors y_train's dtype, which for a string-labeled target (e.g. "yes"/"no")
        # is a numpy object array — np.load refuses those under the default
        # allow_pickle=False. Casting to a fixed-width string dtype keeps every .npy file in
        # this app (cluster labels are always int) loadable the same way, with no special
        # case needed at any read site.
        np.save(preds_path, np.asarray(run.y_pred, dtype=str))

        orm_run = ModelRun(
            job_id=job.id,
            problem_type="classification",
            algorithm=run.algorithm,
            params_json=run.params,
            extra_json=run.extra,
            labels_path=str(preds_path),  # reused as "the per-row output array this run produced"
            metrics_json=metrics,
            artifacts_json={"predictions_path": str(preds_path)},
        )
        db.add(orm_run)
        orm_runs.append(orm_run)
        evaluated_rows.append({"run_id": i, "algorithm": run.algorithm, "params": run.params, **metrics})
    db.flush()

    log(94, "Ranking models...")
    leaderboard = ranking_service.rank(evaluated_rows, problem_type="classification")
    rank_by_index = {int(r["run_id"]): r for r in leaderboard.to_dict("records")}
    for i, orm_run in enumerate(orm_runs):
        lb_row = rank_by_index.get(i)
        if lb_row:
            orm_run.composite_score = lb_row.get("composite_score")
            orm_run.rank = lb_row.get("rank")

    job.status = "completed"
    job.progress_pct = 100.0
    job.log_lines = job.log_lines + ["Job completed."]
    db.commit()


_RUNNERS = {
    "clustering": _run_clustering,
    "classification": _run_classification,
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

        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        runner = _RUNNERS.get(job.problem_type)
        if runner is None:
            job.status = "failed"
            job.error_message = f"No execution path registered for problem_type '{job.problem_type}'."
            db.commit()
            return

        runner(db, job, df, plan, job_dir, log)
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
