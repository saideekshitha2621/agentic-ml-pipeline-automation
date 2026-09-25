"""Chart-ready data for the pipeline run page: model comparison, feature importance, and
either a confusion matrix (classification) or a 2-D PCA cluster plot (clustering).

Read-only — everything is derived from rows/artifacts the pipeline already persisted
(ModelRun.metrics_json, PipelineRun.feature_importance_json, the job's model.csv and the
run's labels .npy), so no new storage or migration is needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sqlalchemy.orm import Session

from app.core.config import JOBS_DIR
from app.db.models import Job as JobORM
from app.db.models import ModelRun as ModelRunORM
from app.db.models import PipelineRun as PipelineRunORM
from app.services import agent_decision_service

TOP_FEATURES = 12
MAX_SCATTER_POINTS = 1500

# key, label, higher_is_better — first entry per problem type is the headline metric.
_METRICS = {
    "classification": [
        ("f1_macro", "F1 (macro)", True), ("accuracy", "Accuracy", True),
        ("precision_macro", "Precision (macro)", True), ("recall_macro", "Recall (macro)", True),
        ("roc_auc", "ROC-AUC", True),
    ],
    "regression": [("r2", "R²", True), ("rmse", "RMSE", False), ("mae", "MAE", False)],
    "clustering": [
        ("silhouette_score", "Silhouette", True), ("calinski_harabasz_score", "Calinski-Harabasz", True),
        ("davies_bouldin_score", "Davies-Bouldin", False),
    ],
}


def _metric_value(run: ModelRunORM, key: str) -> float | None:
    if run.problem_type == "clustering":
        value = getattr(run, key, None)
    else:
        value = (run.metrics_json or {}).get(key)
    return round(float(value), 4) if isinstance(value, (int, float)) else None


def _recommended_run(db: Session, run: PipelineRunORM, best_by_algorithm: list[ModelRunORM]) -> ModelRunORM | None:
    """The model the pipeline recommends: the persisted champion, else the recommendation
    agent's top choice, else simply the top-ranked run."""
    wanted = run.champion_run_id
    if not wanted:
        decision = agent_decision_service.latest_decision(db, run.id, "recommendation")
        wanted = ((decision.decision_json or {}).get("top_choice") or {}).get("cluster_run_id") if decision else None
    for r in best_by_algorithm:
        if r.id == wanted:
            return r
    return best_by_algorithm[0] if best_by_algorithm else None


def _model_comparison(problem_type: str, models: list[ModelRunORM], recommended: ModelRunORM | None) -> dict:
    metrics = _METRICS.get(problem_type, [])
    return {
        "metrics": [{"key": k, "label": label, "higher_is_better": hib} for k, label, hib in metrics],
        "models": [
            {
                "run_id": m.id,
                "algorithm": m.algorithm,
                "rank": m.rank,
                "composite_score": round(m.composite_score, 4) if m.composite_score is not None else None,
                "is_recommended": recommended is not None and m.id == recommended.id,
                "values": {k: _metric_value(m, k) for k, _, _ in metrics},
                **({"n_clusters": m.n_clusters} if problem_type == "clustering" else {}),
            }
            for m in models
        ],
    }


def _cluster_plot(job: JobORM, model: ModelRunORM) -> dict | None:
    job_dir = JOBS_DIR / job.id
    source = next((p for p in (job_dir / "model.csv", job_dir / "encoded.csv") if p.exists()), None)
    try:
        labels = np.load(model.labels_path)
    except (OSError, ValueError):
        return None
    if source is None:
        return None
    X = pd.read_csv(source).select_dtypes("number").to_numpy(dtype=float)
    if len(X) != len(labels) or X.shape[1] == 0:
        return None

    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    if n_components == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    ratios = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    ratios += [0.0] * (2 - len(ratios))

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(labels), MAX_SCATTER_POINTS, replace=False)) if len(labels) > MAX_SCATTER_POINTS else np.arange(len(labels))

    unique, counts = np.unique(labels, return_counts=True)
    return {
        "algorithm": model.algorithm,
        "explained_variance": ratios,
        "n_points_total": int(len(labels)),
        "points": [[round(float(coords[i, 0]), 3), round(float(coords[i, 1]), 3), int(labels[i])] for i in idx],
        "clusters": [
            {"label": int(u), "size": int(c), "pct": round(float(c) / len(labels) * 100, 1), "is_noise": int(u) == -1}
            for u, c in zip(unique, counts)
        ],
    }


def build(db: Session, run: PipelineRunORM) -> dict:
    problem_type = run.problem_type
    job = db.get(JobORM, run.job_id) if run.job_id else None
    result: dict = {
        "problem_type": problem_type, "status": run.status,
        "model_comparison": None, "feature_importance": None, "confusion_matrix": None, "cluster_plot": None,
    }
    if job is None or not problem_type:
        return result

    # HPO only rewrites the best row per algorithm, so compare each algorithm's best-ranked run.
    best: dict[str, ModelRunORM] = {}
    for r in job.runs:
        if r.rank is None:
            continue
        if r.algorithm not in best or r.rank < best[r.algorithm].rank:
            best[r.algorithm] = r
    models = sorted(best.values(), key=lambda r: r.rank)
    if not models:
        return result

    recommended = _recommended_run(db, run, models)
    result["model_comparison"] = _model_comparison(problem_type, models, recommended)
    result["recommended_algorithm"] = recommended.algorithm if recommended else None

    features = (run.feature_importance_json or {}).get("features")
    if features:
        result["feature_importance"] = {
            "algorithm": recommended.algorithm if recommended else None,
            "features": features[:TOP_FEATURES],
            "total_features": len(features),
        }

    if problem_type == "classification" and recommended:
        cm = (recommended.metrics_json or {}).get("confusion_matrix")
        if cm and cm.get("matrix"):
            result["confusion_matrix"] = {"algorithm": recommended.algorithm, **cm}

    if problem_type == "clustering" and recommended:
        result["cluster_plot"] = _cluster_plot(job, recommended)

    return result
