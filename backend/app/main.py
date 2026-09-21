from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.auth import require_api_key
from app.core.config import CORS_ORIGINS
from app.db.database import init_db
from app.routers import (
    approval,
    chat,
    datasets,
    jobs,
    leaderboard,
    monitoring,
    pca,
    pipeline,
    prediction,
    preprocessing,
    reports,
    visualizations,
)

app = FastAPI(title="Unsupervised AutoML Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    import app.plugins  # noqa: F401  (registers all clustering plugins)

    init_db()


_protected = [Depends(require_api_key)]  # no-op unless API_KEYS is set (see app/core/auth.py)

app.include_router(datasets.router, dependencies=_protected)
app.include_router(preprocessing.router, dependencies=_protected)
app.include_router(pca.router, dependencies=_protected)
app.include_router(jobs.router, dependencies=_protected)
app.include_router(leaderboard.router, dependencies=_protected)
app.include_router(approval.router, dependencies=_protected)
app.include_router(visualizations.router, dependencies=_protected)
app.include_router(reports.router, dependencies=_protected)
app.include_router(pipeline.router, dependencies=_protected)
app.include_router(prediction.router, dependencies=_protected)
app.include_router(chat.router, dependencies=_protected)
app.include_router(monitoring.router, dependencies=_protected)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}
