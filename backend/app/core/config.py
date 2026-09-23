"""App settings + wiring so `backend/` can import the sibling `ml_automation` package
without duplicating any algorithm code."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # makes `import ml_automation` work

# Without this, setting a value in backend/.env does nothing — os.environ.get() below would
# just see whatever was already in the process environment when the server started. This
# loads backend/.env into os.environ (if present) before anything reads from it; an
# already-set real environment variable still takes precedence (override=False, the default).
load_dotenv(BACKEND_DIR / ".env")

STORAGE_DIR = BACKEND_DIR / "app" / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
LABELS_DIR = STORAGE_DIR / "labels"
PREDICTIONS_DIR = STORAGE_DIR / "predictions"  # classification/regression per-run predictions (.npy)
EXPORTS_DIR = STORAGE_DIR / "exports"
JOBS_DIR = STORAGE_DIR / "jobs"
MODELS_DIR = STORAGE_DIR / "models"  # persisted champion prediction pipelines (.joblib)
for d in (DATASETS_DIR, LABELS_DIR, PREDICTIONS_DIR, EXPORTS_DIR, JOBS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Unset by default — llm_service falls back to deterministic templates for every narrative
# and the chat endpoint returns a "no LLM configured" message rather than failing. Set at
# least one of these three (see backend/.env) to enable that provider.
#
# Multiple keys per provider (recommended for production — a quota/rate-limit hit on one
# key fails over to the next instead of stalling): in addition to the single var below,
# also set any of
#   <PROVIDER>_API_KEY_1=..., <PROVIDER>_API_KEY_2=..., ...   (numbered, any count)
#   <PROVIDER>_API_KEYS=key-one,key-two,key-three             (comma-separated)
# for PROVIDER in ANTHROPIC / OPENAI / GEMINI. All variants found are pooled together and
# deduplicated — see app/services/llm_key_manager.py for the health-tracking/failover
# logic, and LLM_KEY_WARNING_THRESHOLD / LLM_KEY_EXHAUSTED_THRESHOLD /
# LLM_KEY_COOLDOWN_SECONDS / LLM_KEY_AUTH_COOLDOWN_SECONDS / LLM_MAX_RETRIES_PER_REQUEST /
# LLM_PROVIDER_ORDER for its configurable thresholds.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# SQLite by default (zero setup). For PostgreSQL set DATABASE_URL, e.g.
# postgresql+psycopg://user:password@host:5432/automl  (and `pip install -r requirements-postgres.txt`).
DATABASE_URL = os.environ.get("DATABASE_URL") or f"sqlite:///{(STORAGE_DIR / 'app.db').as_posix()}"

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

DEFAULT_HYPERPARAMETER_CONFIG = {
    "kmeans": {"k_range": [2, 8]},
    "hierarchical": {"k_range": [2, 8], "linkages": ["ward", "complete", "average", "single"]},
    "dbscan": {"min_samples": [3, 5, 10]},
    "gmm": {"component_range": [2, 8], "covariance_types": ["full"]},
    "spectral": {"k_range": [2, 8]},
    "birch": {"k_range": [2, 8], "threshold": 0.5},
    "optics": {"min_samples": [3, 5, 10]},
}

# Kept as a separate constant rather than nesting under DEFAULT_HYPERPARAMETER_CONFIG so
# clustering_service.run_all()'s `DEFAULT_HYPERPARAMETER_CONFIG.get(name, {})` lookup (keyed
# directly by clustering algorithm name) doesn't have to change.
DEFAULT_CLASSIFICATION_HYPERPARAMETER_CONFIG = {
    "logistic_regression": {"C": [0.1, 1.0, 10.0]},
    "random_forest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "gradient_boosting": {"n_estimators": [100], "learning_rate": [0.05, 0.1]},
    "svm": {"C": [1.0, 10.0]},
    "knn": {"n_neighbors": [3, 5, 7]},
}

# Same shape/purpose as its classification counterpart above, keyed by the regression
# plugin registry's algorithm names.
DEFAULT_REGRESSION_HYPERPARAMETER_CONFIG = {
    "linear_regression": {"alpha": [0.1, 1.0, 10.0]},
    "random_forest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "gradient_boosting": {"n_estimators": [100], "learning_rate": [0.05, 0.1]},
    "svr": {"C": [1.0, 10.0]},
    "knn": {"n_neighbors": [3, 5, 7]},
}

RANDOM_STATE = 42
