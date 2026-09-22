# Unsupervised AutoML Platform — Architecture

Monorepo containing the existing `ml_automation` core pipeline (reused, not duplicated),
a FastAPI backend service layer around it, and a React/TypeScript/MUI frontend.

## 1. Folder Structure

```
ML Automation/
├── ml_automation/                 # existing core algorithms (preprocessing, clustering,
│   │                               # evaluation, ranking, interpretation) — reused as a library
│   ├── preprocessing.py
│   ├── dimensionality_reduction.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── ranking.py
│   ├── interpretation.py
│   ├── reporting.py
│   └── pipeline.py
│
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, router registration, CORS
│   │   ├── core/
│   │   │   ├── config.py          # settings (storage paths, DB url, default grids)
│   │   │   └── logging.py
│   │   ├── db/
│   │   │   ├── database.py        # SQLAlchemy engine/session
│   │   │   └── models.py          # ORM: Dataset, PreprocessingPlan, Job, ClusterRun, Approval
│   │   ├── schemas/                # Pydantic request/response models (1 file per resource)
│   │   │   ├── dataset.py
│   │   │   └── pipeline.py
│   │   ├── services/                # business logic, orchestrates ml_automation + plugins
│   │   │   ├── profiling_service.py     # data quality score, outliers, category suggestions
│   │   │   ├── preprocessing_service.py
│   │   │   ├── pca_service.py
│   │   │   ├── clustering_service.py    # runs the plugin registry over a config-driven grid
│   │   │   ├── evaluation_service.py
│   │   │   ├── ranking_service.py
│   │   │   ├── recommendation_service.py # top-3 + strengths/weaknesses copy
│   │   │   ├── explanation_service.py    # cluster profiles / feature importance / naming
│   │   │   └── export_service.py         # CSV / Excel (openpyxl) / PDF (reportlab)
│   │   ├── plugins/                 # plugin-based clustering framework
│   │   │   ├── base.py              # ClusteringPlugin ABC: name, param_grid(), fit_predict()
│   │   │   ├── registry.py          # @register_plugin decorator + PLUGIN_REGISTRY dict
│   │   │   ├── kmeans_plugin.py
│   │   │   ├── dbscan_plugin.py
│   │   │   ├── hierarchical_plugin.py
│   │   │   ├── gmm_plugin.py
│   │   │   ├── spectral_plugin.py
│   │   │   ├── birch_plugin.py
│   │   │   └── optics_plugin.py
│   │   ├── routers/                 # thin HTTP layer, 1 file per screen/resource
│   │   │   ├── datasets.py
│   │   │   ├── pipeline.py          # agentic pipeline runs, HITL review, reports
│   │   │   ├── prediction.py
│   │   │   ├── chat.py
│   │   │   └── monitoring.py
│   │   └── storage/                 # runtime: uploaded CSVs, exports (gitignored)
│   ├── tests/
│   │   ├── test_preprocessing_service.py
│   │   ├── test_clustering_registry.py
│   │   └── test_ranking_service.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── api/                     # typed API client + React Query hooks, 1 file/resource
│   │   │   ├── client.ts
│   │   │   ├── datasets.ts
│   │   │   ├── pipeline.ts
│   │   │   ├── prediction.ts
│   │   │   └── chat.ts
│   │   ├── pages/                   # one component per screen
│   │   │   ├── DashboardPage.tsx
│   │   │   ├── UploadPage.tsx
│   │   │   ├── DataQualityPage.tsx     # standalone profiling view, opened from the Dashboard
│   │   │   ├── PipelineRunPage.tsx     # the agentic run: HITL review + built-in report
│   │   │   └── PredictionPlaygroundPage.tsx
│   │   ├── components/              # shared: NavShell, MetricCard, pipeline/ (ChatPanel,
│   │   │   │                         # StageContent, ExecutiveSummaryCard)
│   │   ├── types/                   # TS types mirroring backend Pydantic schemas
│   │   ├── App.tsx                  # router + MUI theme + layout shell
│   │   ├── theme.ts
│   │   └── main.tsx
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml
└── ARCHITECTURE.md
```

**Why this split:** `ml_automation` stays the single source of truth for the actual ML logic
(already built, tested via CLI/Streamlit). The backend does not reimplement clustering math —
`services/` calls into `ml_automation` and the new `plugins/` (which extend it with Spectral,
Birch, OPTICS). This avoids the classic mistake of forking algorithm code between a "script
version" and a "web version."

## 2. Backend Architecture

**Layering:** `routers` (HTTP/validation only) → `services` (business logic, orchestration,
persistence calls) → `plugins`/`ml_automation` (pure algorithm code, no I/O). Routers never
touch the DB or sklearn directly — this is what makes the service layer independently testable
and lets the plugin framework be swapped or extended without touching HTTP code.

**Plugin-based clustering framework** (`plugins/`): each algorithm implements
```python
class ClusteringPlugin(ABC):
    name: str
    def param_grid(self, config: dict) -> list[dict]: ...
    def fit_predict(self, X: np.ndarray, params: dict) -> np.ndarray: ...
```
and self-registers via `@register_plugin`. `clustering_service.run_job()` iterates
`PLUGIN_REGISTRY`, expands each plugin's grid from the job's **configuration** (a JSON blob
saved per job — "configuration-driven execution"), and executes every combination. Adding
Spectral/Birch/OPTICS (or a future algorithm) means adding one file — no router or service
change required.

**Background execution:** starting a pipeline run (`POST /pipeline-runs`) returns immediately;
the orchestrator (`app/agents/pipeline_graph.py`) advances the run through an in-process task
queue (`task_queue_service`, bounded and serialized per run — the roadmap below calls out
swapping this for Celery/RQ + Redis once concurrent multi-user load requires it). Training still
creates a `Job` row internally (`progress_pct`, `training_status_json`) so the frontend can poll
`GET /pipeline-runs/{id}/training-progress` for the stepper's progress bar.

**Storage:** uploaded CSVs and export artifacts live on local disk under `backend/app/storage/`
(swap for S3/Blob storage in production — see roadmap). Structured metadata (datasets, plans,
jobs, runs, approvals) lives in the relational DB described below.

## 3. Frontend Architecture

Upload now goes straight into the **agentic pipeline** (`PipelineRunPage`): the agent profiles,
cleans, transforms, trains, tunes and evaluates the dataset itself, pausing at HITL gates for
review (`DecisionReviewCard`) and finishing with a built-in report (PDF export). A separate
manual, screen-by-screen pipeline (preprocessing review → PCA → model execution → leaderboard →
comparison → approval → visualizations → reports) existed earlier but had no way to reach it from
the UI and was removed; `Dataset`/`Job`/`ClusterRun`/etc. tables and their services still exist
because the agentic pipeline uses them internally.

- **Routing/layout:** `App.tsx` renders an MUI `Drawer` + `AppBar` shell with react-router routes:
  Dashboard, Upload, a standalone Data Quality view (opened from the Dashboard's dataset table),
  the agentic Pipeline Run screen, and the Prediction Playground.
- **Data fetching:** every screen uses **React Query** (`useQuery`/`useMutation`) against the
  typed client in `src/api/`, never raw `fetch` in components. Reviewing a HITL decision
  (`useReviewDecision`) invalidates the run's decisions/executive-summary so the stepper and
  report reflect the latest server state immediately.
- **Polling:** `PipelineRunPage` polls training progress and pipeline-run status while a run is
  active, stopping automatically once the run reaches `completed`/`failed`.
- **State:** server state lives entirely in React Query's cache; the only client-only state is the
  current `datasetId`/`pipelineRunId` in `WorkspaceContext` (persisted to `sessionStorage`), so a
  screen is always resumable from a shared link/refresh.

## 4. Database Schema

Relational (SQLite for dev via SQLAlchemy, swap to Postgres for prod — same models).

```
datasets
  id                UUID PK
  filename           TEXT
  storage_path       TEXT
  uploaded_at        DATETIME
  n_rows             INT
  n_columns          INT
  data_quality_score FLOAT              -- 0-100, from profiling_service
  profile_json       JSON               -- missing values, duplicates, outliers, dtype summary

preprocessing_plans
  id                 UUID PK
  dataset_id         UUID FK -> datasets.id
  numerical_columns   JSON
  categorical_columns JSON
  dropped_columns     JSON
  numerical_impute_strategy   TEXT       -- median | mean
  categorical_impute_strategy TEXT       -- mode
  scaling_method      TEXT               -- standard | minmax | robust | none
  drop_duplicates     BOOLEAN
  pca_enabled         BOOLEAN
  pca_variance_target FLOAT
  created_at          DATETIME
  is_active           BOOLEAN            -- latest plan the user saved for this dataset

jobs
  id                 UUID PK
  dataset_id         UUID FK -> datasets.id
  preprocessing_plan_id UUID FK -> preprocessing_plans.id
  config_json        JSON               -- hyperparameter grid overrides per algorithm
  status             TEXT               -- queued | running | completed | failed
  progress_pct       FLOAT
  log_lines          JSON               -- list[str], appended during execution
  pca_report_json    JSON
  started_at         DATETIME
  completed_at       DATETIME
  error_message      TEXT NULL

cluster_runs
  id                 UUID PK
  job_id             UUID FK -> jobs.id
  algorithm          TEXT
  params_json        JSON
  n_clusters         INT
  n_noise            INT
  silhouette_score    FLOAT NULL
  davies_bouldin_score FLOAT NULL
  calinski_harabasz_score FLOAT NULL
  composite_score     FLOAT NULL
  rank                INT NULL
  labels_path         TEXT              -- npy file on disk (avoid bloating the DB row)

approvals
  id                 UUID PK
  job_id             UUID FK -> jobs.id (unique — one approved model per job)
  cluster_run_id     UUID FK -> cluster_runs.id
  approved_by        TEXT               -- user identifier/email
  approved_at        DATETIME
  notes              TEXT NULL

cluster_interpretations
  id                 UUID PK
  cluster_run_id     UUID FK -> cluster_runs.id (unique)
  profiles_json       JSON              -- per-cluster mean/mode vs overall baseline
  feature_importance_json JSON
  summaries_json       JSON             -- business-language text per cluster
  suggested_names_json JSON

export_artifacts
  id                 UUID PK
  job_id             UUID FK -> jobs.id
  format             TEXT               -- csv | xlsx | pdf
  storage_path        TEXT
  generated_at         DATETIME
```

## 5. API Contracts

All under `/api/v1`. Bodies/responses are Pydantic-validated; errors follow
`{"detail": "..."}` (FastAPI default) with proper 4xx codes (422 validation, 404 not found,
409 e.g. approving a job twice).

| Method & Path | Purpose | Request | Response |
|---|---|---|---|
| `POST /datasets` | Upload CSV (multipart) | file | `Dataset` (id, filename, n_rows, n_columns, data_quality_score) |
| `GET /datasets` | Upload history / dashboard list | — | `Dataset[]` |
| `GET /datasets/{id}` | Dataset summary for Dashboard | — | `Dataset` |
| `GET /datasets/{id}/profile` | Data Quality screen | — | `DataProfile` (missing table, duplicates, outliers, category-standardization suggestions) |
| `DELETE /datasets/{id}` | Delete a dataset (and its plans/jobs/pipeline runs) | — | 204 |
| `POST /pipeline-runs` | Start the agentic pipeline (Upload/Dashboard) | `{dataset_id, learning_type?}` | `PipelineRun` (status=profiling) |
| `GET /pipeline-runs/{id}` | Poll run status | — | `PipelineRun` |
| `GET /pipeline-runs/{id}/decisions` | Agent Activity Timeline | — | `AgentDecision[]` |
| `POST /pipeline-runs/{id}/decisions/{decision_id}/review` | HITL approve/edit/reject a proposed decision | `{action, edits?, reason?, reviewed_by}` | `AgentDecision` |
| `GET /pipeline-runs/{id}/training-progress` | Per-algorithm training status while `status=training` | — | `{algorithms: [...], job_status, progress_pct}` |
| `GET /pipeline-runs/{id}/executive-summary` | Executive Summary card, updates as stages complete | — | `ExecutiveSummary` |
| `GET /pipeline-runs/{id}/recommendation` | Top choice + alternatives once evaluated | — | `AgentRecommendation` |
| `GET /pipeline-runs/{id}/report` | Full business report (once `status=completed`) | — | `PipelineReport` |
| `GET /pipeline-runs/{id}/report/export` | Download the report as PDF | — | file stream |
| `GET /pipeline-runs/{id}/prediction-schema` | Prediction Playground input form | — | feature schema |
| `POST /pipeline-runs/{id}/predict` | Score a single row against the champion model | row payload | prediction |

## 6. Implementation Roadmap / Development Phases

| Phase | Scope | Status |
|---|---|---|
| **0. Core pipeline** | `ml_automation` package (preprocessing, PCA, 4 algorithms, metrics, ranking, interpretation, reporting) + CLI/Streamlit | ✅ done (prior session) |
| **1. Backend vertical slice** | FastAPI app, DB models, plugin framework (+Spectral/Birch/OPTICS), services wrapping `ml_automation`, routers for upload → preprocess → PCA → run → leaderboard → approve → visualize → export (CSV/Excel/PDF) | **this session** |
| **2. Frontend vertical slice** | React/TS/MUI app, React Query hooks, all 11 pages wired to the real API, Recharts + Plotly visualizations | **this session** |
| **3. Hardening** | Auth (even basic), input validation edge cases, outlier detection depth, category-standardization heuristics, pagination on datasets/jobs lists, error boundaries | follow-up |
| **4. Scale-out** | Swap `BackgroundTasks` for Celery/RQ + Redis, swap local disk for S3/Blob, Postgres in prod, multi-tenant dataset isolation | follow-up |
| **5. Ops** | Docker Compose → CI pipeline, structured logging/metrics, docker healthchecks, e2e tests (Playwright) on top of the unit tests added in phase 1-2 | follow-up |

## 7. Development Phases — this session's build order

1. Backend: DB models + migrations-free `create_all` bootstrap
2. Backend: plugin framework + 7 clustering plugins
3. Backend: services (profiling, preprocessing, pca, clustering, evaluation, ranking,
   recommendation, explanation, export)
4. Backend: routers + `main.py` wiring, CORS for the Vite dev server
5. Backend: unit tests for services/plugins
6. Frontend: Vite scaffold, theme, layout shell, routing
7. Frontend: API client + React Query hooks per resource
8. Frontend: 11 pages, wired to live endpoints
9. Docker: backend + frontend Dockerfiles + `docker-compose.yml`
10. Smoke test: run both services, upload the existing `sample_customers.csv`, walk the full flow
