from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String)
    storage_path: Mapped[str] = mapped_column(String)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    n_rows: Mapped[int] = mapped_column(Integer)
    n_columns: Mapped[int] = mapped_column(Integer)
    data_quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    profile_json: Mapped[dict] = mapped_column(JSON, default=dict)

    plans: Mapped[list["PreprocessingPlanORM"]] = relationship(back_populates="dataset")
    jobs: Mapped[list["Job"]] = relationship(back_populates="dataset")


class PreprocessingPlanORM(Base):
    __tablename__ = "preprocessing_plans"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    numerical_columns: Mapped[list] = mapped_column(JSON, default=list)
    categorical_columns: Mapped[list] = mapped_column(JSON, default=list)
    dropped_columns: Mapped[list] = mapped_column(JSON, default=list)
    numerical_impute_strategy: Mapped[str] = mapped_column(String, default="median")
    categorical_impute_strategy: Mapped[str] = mapped_column(String, default="mode")
    scaling_method: Mapped[str] = mapped_column(String, default="standard")
    drop_duplicates: Mapped[bool] = mapped_column(Boolean, default=True)
    pca_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    pca_variance_target: Mapped[float] = mapped_column(Float, default=0.95)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="plans")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    preprocessing_plan_id: Mapped[str] = mapped_column(ForeignKey("preprocessing_plans.id"))
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|running|completed|failed
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0)
    log_lines: Mapped[list] = mapped_column(JSON, default=list)
    pca_report_json: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="jobs")
    runs: Mapped[list["ClusterRun"]] = relationship(back_populates="job")
    approval: Mapped["Approval"] = relationship(back_populates="job", uselist=False)


class ClusterRun(Base):
    __tablename__ = "cluster_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    algorithm: Mapped[str] = mapped_column(String)
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    extra_json: Mapped[dict] = mapped_column(JSON, default=dict)  # e.g. {"inertia": ...} for KMeans
    n_clusters: Mapped[int] = mapped_column(Integer, default=0)
    n_noise: Mapped[int] = mapped_column(Integer, default=0)
    noise_pct: Mapped[float] = mapped_column(Float, default=0.0)
    silhouette_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    davies_bouldin_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    calinski_harabasz_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    composite_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    labels_path: Mapped[str] = mapped_column(String)

    job: Mapped["Job"] = relationship(back_populates="runs")
    interpretation: Mapped["ClusterInterpretation"] = relationship(
        back_populates="cluster_run", uselist=False
    )


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True)
    cluster_run_id: Mapped[str] = mapped_column(ForeignKey("cluster_runs.id"))
    approved_by: Mapped[str] = mapped_column(String)
    approved_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped["Job"] = relationship(back_populates="approval")


class ClusterInterpretation(Base):
    __tablename__ = "cluster_interpretations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    cluster_run_id: Mapped[str] = mapped_column(ForeignKey("cluster_runs.id"), unique=True)
    profiles_json: Mapped[list] = mapped_column(JSON, default=list)
    feature_importance_json: Mapped[list] = mapped_column(JSON, default=list)
    summaries_json: Mapped[dict] = mapped_column(JSON, default=dict)
    suggested_names_json: Mapped[dict] = mapped_column(JSON, default=dict)

    cluster_run: Mapped["ClusterRun"] = relationship(back_populates="interpretation")


class ExportArtifact(Base):
    __tablename__ = "export_artifacts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    format: Mapped[str] = mapped_column(String)  # csv|xlsx|pdf
    storage_path: Mapped[str] = mapped_column(String)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
