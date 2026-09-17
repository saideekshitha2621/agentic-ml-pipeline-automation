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
    # {column: {"action": "mean"|"median"|"mode"|"drop_rows"|"fill_zero"|"fill_custom"|
    # "business_rule"|"keep", "custom_value": Any}} — per-column imputation strategy from the
    # Cleaning Plan HITL stage. Empty for plans built by the old detect_default_plan() flow,
    # which keeps using numerical_impute_strategy/categorical_impute_strategy globally.
    column_actions: Mapped[dict] = mapped_column(JSON, default=dict)

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
    # "clustering" | "classification" | "regression" — only "clustering" has an execution
    # path today (see agents/orchestrator.py SUPPORTED_PROBLEM_TYPES); the column exists now
    # so ModelRun rows and every job-scoped service can key off job.problem_type directly
    # instead of having it threaded through as a parameter.
    problem_type: Mapped[str] = mapped_column(String, default="clustering")
    target_column: Mapped[str | None] = mapped_column(String, nullable=True)
    # Restricts execution to this subset of registered plugin names when set (populated by
    # the algorithm-shortlist HITL gate); empty list means "run everything registered", same
    # as today's behavior.
    selected_algorithms: Mapped[list] = mapped_column(JSON, default=list)
    test_size: Mapped[float] = mapped_column(Float, default=0.2)
    # {algorithm_name: "running"|"completed"|"failed"} — updated live as each plugin trains,
    # polled by the Training Progress stage instead of only the aggregate progress_pct.
    training_status_json: Mapped[dict] = mapped_column(JSON, default=dict)

    dataset: Mapped["Dataset"] = relationship(back_populates="jobs")
    runs: Mapped[list["ModelRun"]] = relationship(back_populates="job")
    approval: Mapped["Approval"] = relationship(back_populates="job", uselist=False)


class ModelRun(Base):
    """One algorithm x hyperparameter combination evaluated for a Job, for any problem_type.

    Columns that mean the same thing for every task (algorithm, params, composite_score,
    rank) stay typed. Columns whose shape depends on the task (silhouette/davies_bouldin/
    calinski_harabasz for clustering; accuracy/f1/roc_auc for classification; rmse/r2 for
    regression) live in `metrics_json` instead of one column per metric per task.

    The clustering-specific typed columns below (n_clusters, n_noise, noise_pct,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score, labels_path) are kept
    alongside metrics_json/artifacts_json rather than removed: every existing router
    (leaderboard, approval, visualizations, reports) still reads them directly, and
    rewriting those call sites is out of scope for this pass. job_runner_service now
    populates both the typed columns and their metrics_json/artifacts_json mirror for
    every run, so either can be read without a behavior change.
    """

    __tablename__ = "model_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    problem_type: Mapped[str] = mapped_column(String, default="clustering")
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
    # Generic mirrors — task-agnostic shape for future classification/regression readers
    # (e.g. {"accuracy": .., "f1_macro": ..} or {"rmse": .., "r2": ..}) without disturbing
    # the typed columns clustering's existing routers already depend on.
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    artifacts_json: Mapped[dict] = mapped_column(JSON, default=dict)  # e.g. {"labels_path": ...}
    baseline_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)  # pre-HPO metrics, for comparison

    job: Mapped["Job"] = relationship(back_populates="runs")
    interpretation: Mapped["ClusterInterpretation"] = relationship(
        back_populates="cluster_run", uselist=False
    )


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True)
    cluster_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.id"))
    approved_by: Mapped[str] = mapped_column(String)
    approved_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped["Job"] = relationship(back_populates="approval")


class ClusterInterpretation(Base):
    __tablename__ = "cluster_interpretations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    cluster_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.id"), unique=True)
    profiles_json: Mapped[list] = mapped_column(JSON, default=list)
    feature_importance_json: Mapped[list] = mapped_column(JSON, default=list)
    summaries_json: Mapped[dict] = mapped_column(JSON, default=dict)
    suggested_names_json: Mapped[dict] = mapped_column(JSON, default=dict)

    cluster_run: Mapped["ModelRun"] = relationship(back_populates="interpretation")


class ExportArtifact(Base):
    __tablename__ = "export_artifacts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    format: Mapped[str] = mapped_column(String)  # csv|xlsx|pdf
    storage_path: Mapped[str] = mapped_column(String)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class PipelineRun(Base):
    """One agent-orchestrated run: profiling -> problem detection -> preprocessing ->
    model execution -> recommendation -> reporting. Additive to the existing Job-centric
    clustering flow — when problem_type == 'clustering', a PipelineRun creates and delegates
    to a regular Job row (see agents/orchestrator.py), so the existing jobs/cluster_runs/
    approvals tables and their routers keep working completely unchanged."""

    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    declared_target: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="profiling")
    # profiling | awaiting_problem_approval | model_execution |
    # awaiting_recommendation_approval | reporting | completed | failed
    problem_type: Mapped[str | None] = mapped_column(String, nullable=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Populated once the recommended model is approved and refit on the full dataset —
    # unlocks the Prediction Playground and Explainability stages.
    champion_model_path: Mapped[str | None] = mapped_column(String, nullable=True)
    champion_run_id: Mapped[str | None] = mapped_column(ForeignKey("model_runs.id"), nullable=True)
    feature_schema_json: Mapped[dict] = mapped_column(JSON, default=dict)
    feature_importance_json: Mapped[dict] = mapped_column(JSON, default=dict)

    decisions: Mapped[list["AgentDecision"]] = relationship(
        back_populates="pipeline_run", order_by="AgentDecision.created_at"
    )


class AgentDecision(Base):
    """Durable record of every agent judgment call: what was proposed, how confident the
    agent was, why, and what a human did about it. Powers the HITL review cards, the agent
    activity timeline, and the Reporting Agent's decision-trail narrative."""

    __tablename__ = "agent_decisions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    pipeline_run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id"))
    agent_name: Mapped[str] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String)
    decision_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning_text: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String, default="proposed")  # proposed|approved|edited|rejected
    human_edits_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    pipeline_run: Mapped["PipelineRun"] = relationship(back_populates="decisions")


class ChatMessage(Base):
    """Conversational Q&A history for a pipeline run — grounding context for each answer
    is assembled at answer-time from the run's own AgentDecision trail, not stored here."""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    pipeline_run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id"))
    role: Mapped[str] = mapped_column(String)  # user|assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class PredictionLog(Base):
    """Audit trail of Prediction Playground requests against a run's champion model."""

    __tablename__ = "prediction_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    pipeline_run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id"))
    input_json: Mapped[dict] = mapped_column(JSON, default=dict)
    output_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class AppSettings(Base):
    """Singleton row (id is always 'default') holding the runtime-editable LLM provider
    configuration set via the Settings page. Takes precedence over the ANTHROPIC_API_KEY/
    GEMINI_API_KEY/OPENAI_API_KEY env vars when llm_provider+llm_api_key are both set — see
    services/llm_config_service.py::get_active_config(). Stored in plaintext in this local,
    single-user SQLite database, the same trust level as every other row in it; the API
    only ever returns it back masked (see llm_config_service.mask())."""

    __tablename__ = "app_settings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: "default")
    llm_provider: Mapped[str | None] = mapped_column(String, nullable=True)  # anthropic|gemini|openai
    llm_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
