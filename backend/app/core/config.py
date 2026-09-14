"""App settings + wiring so `backend/` can import the sibling `ml_automation` package
without duplicating any algorithm code."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # makes `import ml_automation` work

STORAGE_DIR = BACKEND_DIR / "app" / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
LABELS_DIR = STORAGE_DIR / "labels"
EXPORTS_DIR = STORAGE_DIR / "exports"
JOBS_DIR = STORAGE_DIR / "jobs"
for d in (DATASETS_DIR, LABELS_DIR, EXPORTS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{(STORAGE_DIR / 'app.db').as_posix()}"

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

RANDOM_STATE = 42
