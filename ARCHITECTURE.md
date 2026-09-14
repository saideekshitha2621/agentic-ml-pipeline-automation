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
│   │   │   ├── preprocessing.py
│   │   │   ├── pca.py
│   │   │   ├── job.py
│   │   │   ├── leaderboard.py
│   │   │   ├── approval.py
│   │   │   └── visualization.py
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
│   │   │   ├── preprocessing.py
│   │   │   ├── pca.py
│   │   │   ├── jobs.py
│   │   │   ├── leaderboard.py
│   │   │   ├── approval.py
│   │   │   ├── visualizations.py
│   │   │   └── reports.py
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
│   │   │   ├── preprocessing.ts
│   │   │   ├── pca.ts
│   │   │   ├── jobs.ts
│   │   │   ├── leaderboard.ts
│   │   │   ├── approval.ts
│   │   │   └── reports.ts
│   │   ├── pages/                   # one component per screen (11 screens)
│   │   │   ├── DashboardPage.tsx
│   │   │   ├── UploadPage.tsx
│   │   │   ├── DataQualityPage.tsx
│   │   │   ├── PreprocessingReviewPage.tsx
│   │   │   ├── PCAPage.tsx
│   │   │   ├── ModelExecutionPage.tsx
│   │   │   ├── LeaderboardPage.tsx
│   │   │   ├── ComparisonPage.tsx
│   │   │   ├── ApprovalPage.tsx
│   │   │   ├── VisualizationPage.tsx
│   │   │   └── ReportsPage.tsx
│   │   ├── components/              # shared: NavShell, DataTable, MetricCard, ScatterChart...
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

**Background execution:** `POST /jobs` returns immediately with `job_id` and status `queued`;
actual training runs in a FastAPI `BackgroundTasks` thread (MVP-appropriate — the roadmap below
calls out swapping this for Celery/RQ + Redis once concurrent multi-user load requires it). Progress
and log lines are written to the `Job` row (`progress_pct`, `log_lines` JSON array) so the frontend
can poll `GET /jobs/{id}` for the Execution Screen's progress bar and log tail.

**Storage:** uploaded CSVs and export artifacts live on local disk under `backend/app/storage/`
(swap for S3/Blob storage in production — see roadmap). Structured metadata (datasets, plans,
jobs, runs, approvals) lives in the relational DB described below.

## 3. Frontend Architecture

- **Routing/layout:** `App.tsx` renders an MUI `Drawer` + `AppBar` shell with react-router routes
  for the 11 screens, mirroring the pipeline's stage order so the nav doubles as a progress map
  (same idea as the existing Streamlit sidebar stepper, ported to a proper SPA).
- **Data fetching:** every screen uses **React Query** (`useQuery`/`useMutation`) against the
  typed client in `src/api/`, never raw `fetch` in components. Mutations (upload, save plan,
  apply PCA, start job, approve model) invalidate the relevant query keys so downstream screens
  always reflect the latest server state — this is what lets a user go back and change the
  preprocessing plan without stale data leaking into the leaderboard.
- **Polling:** `ModelExecutionPage` uses `useQuery` with `refetchInterval` while `job.status ===
  'running'` to drive the progress bar and log tail, stopping automatically on `completed`/`failed`.
- **Charts:** Recharts for standard bar/line (leaderboard metric bars, explained-variance,
  cluster-size bars); Plotly (`react-plotly.js`) for the PCA/cluster scatter plots, since Plotly's
  built-in zoom/pan/hover is a better fit for exploring point clouds than Recharts.
- **State:** server state lives entirely in React Query's cache; the only client-only state is
  form inputs mid-edit (e.g. the preprocessing plan editor before "Save") and the current
  `datasetId`/`jobId` in the URL (`/jobs/:jobId/leaderboard`), so a screen is always resumable
  from a shared link/refresh.

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
| `GET /datasets/{id}/preprocessing-plan` | Preprocessing Review screen (auto-detected defaults) | — | `PreprocessingPlan` |
| `PUT /datasets/{id}/preprocessing-plan` | Save human edits | `PreprocessingPlanUpdate` | `PreprocessingPlan` |
| `POST /datasets/{id}/preprocess` | Apply the active plan | `{plan_id}` | `PreprocessingReport` |
| `GET /datasets/{id}/pca-preview` | PCA screen: variance curve before committing | — | `{explained_variance_ratio, cumulative_variance}` |
| `POST /datasets/{id}/pca` | Apply PCA at chosen variance/component target | `{n_components}` | `PCAReport` |
| `POST /jobs` | Start clustering run (Model Execution screen) | `{dataset_id, preprocessing_plan_id, config}` | `Job` (status=queued) |
| `GET /jobs/{id}` | Poll status/progress/logs | — | `Job` |
| `GET /jobs/{id}/leaderboard` | Leaderboard screen | — | `ClusterRun[]` (sorted by rank) |
| `GET /jobs/{id}/compare?run_ids=a,b,c` | Model Comparison screen | — | `ClusterRun[]` with aligned metrics |
| `GET /jobs/{id}/recommendations` | HITL Approval screen: top 3 + rationale | — | `Recommendation[]` (strengths/weaknesses copy) |
| `POST /jobs/{id}/approve` | Human approves final model | `{cluster_run_id, approved_by, notes?}` | `Approval` |
| `GET /jobs/{id}/visualizations` | Visualization screen data | — | `{scatter: [...], cluster_sizes: [...], metric_comparison: [...]}` |
| `GET /jobs/{id}/interpretation` | Cluster explanation engine output | — | `ClusterInterpretation` |
| `GET /jobs/{id}/export?format=csv\|xlsx\|pdf` | Reports screen download | — | file stream |

## 6. Sample UI Wireframes (text layout)

```
┌─ Dashboard ─────────────────────────────────────────────┐
│ AppBar: Unsupervised AutoML          [Upload Dataset]    │
│ ┌────────────┬────────────┬────────────┬──────────────┐ │
│ │ Rows: 12.4k│ Columns: 18│ DQ Score:87 │ Missing: 4.2%│ │
│ └────────────┴────────────┴────────────┴──────────────┘ │
│ Recent Datasets table (name, uploaded, DQ score, →)      │
└────────────────────────────────────────────────────────┘

┌─ Preprocessing Review ──────────────────────────────────┐
│ Left: column list w/ type chips (editable dropdown)      │
│ Right panel: Impute strategy [median▾] Scaling [Std▾]    │
│              [x] Drop duplicates   [ ] Enable PCA         │
│              [Save Plan] [Continue →]                     │
└────────────────────────────────────────────────────────┘

┌─ Model Execution ───────────────────────────────────────┐
│ Algorithm chips: KMeans DBSCAN Hierarchical GMM           │
│                  Spectral Birch OPTICS   [Run Pipeline]   │
│ Progress: ████████████░░░░░░  62%                         │
│ Log tail (monospace, auto-scroll)                          │
└────────────────────────────────────────────────────────┘

┌─ Leaderboard ────────────────────────────────────────────┐
│ Table: Algorithm | Params | k | Noise | Sil | DB | CH |Rank│
│ Row click → select for Comparison  [Compare Selected]      │
└────────────────────────────────────────────────────────┘

┌─ HITL Approval ──────────────────────────────────────────┐
│ 3 cards: #1 DBSCAN  #2 KMeans  #3 GMM                     │
│ each: metrics + "Why it ranked here" + [Approve this model]│
└────────────────────────────────────────────────────────┘
```

## 7. Implementation Roadmap / Development Phases

| Phase | Scope | Status |
|---|---|---|
| **0. Core pipeline** | `ml_automation` package (preprocessing, PCA, 4 algorithms, metrics, ranking, interpretation, reporting) + CLI/Streamlit | ✅ done (prior session) |
| **1. Backend vertical slice** | FastAPI app, DB models, plugin framework (+Spectral/Birch/OPTICS), services wrapping `ml_automation`, routers for upload → preprocess → PCA → run → leaderboard → approve → visualize → export (CSV/Excel/PDF) | **this session** |
| **2. Frontend vertical slice** | React/TS/MUI app, React Query hooks, all 11 pages wired to the real API, Recharts + Plotly visualizations | **this session** |
| **3. Hardening** | Auth (even basic), input validation edge cases, outlier detection depth, category-standardization heuristics, pagination on datasets/jobs lists, error boundaries | follow-up |
| **4. Scale-out** | Swap `BackgroundTasks` for Celery/RQ + Redis, swap local disk for S3/Blob, Postgres in prod, multi-tenant dataset isolation | follow-up |
| **5. Ops** | Docker Compose → CI pipeline, structured logging/metrics, docker healthchecks, e2e tests (Playwright) on top of the unit tests added in phase 1-2 | follow-up |

## 8. Development Phases — this session's build order

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
