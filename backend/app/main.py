from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import CORS_ORIGINS
from app.db.database import init_db
from app.routers import (
    approval,
    chat,
    datasets,
    jobs,
    leaderboard,
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


app.include_router(datasets.router)
app.include_router(preprocessing.router)
app.include_router(pca.router)
app.include_router(jobs.router)
app.include_router(leaderboard.router)
app.include_router(approval.router)
app.include_router(visualizations.router)
app.include_router(reports.router)
app.include_router(pipeline.router)
app.include_router(prediction.router)
app.include_router(chat.router)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}
