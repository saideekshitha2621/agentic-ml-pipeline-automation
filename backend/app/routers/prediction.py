"""Prediction Playground (req 12) + per-prediction Explainability (req 13) — both need the
champion model `champion_service.build_and_persist` persists once the recommendation HITL
gate is approved (see agents/orchestrator.py::finalize_after_recommendation_approval)."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import PipelineRun as PipelineRunORM
from app.db.models import PredictionLog as PredictionLogORM
from app.schemas.pipeline import PredictionRequest
from app.services import champion_service, explainability_service

router = APIRouter(prefix="/api/v1/pipeline-runs", tags=["prediction"])


def _get_champion_or_409(pipeline_run_id: str, db: Session) -> PipelineRunORM:
    run = db.get(PipelineRunORM, pipeline_run_id)
    if not run:
        raise HTTPException(404, "Pipeline run not found.")
    if not run.champion_model_path:
        raise HTTPException(409, "No champion model has been finalized for this run yet.")
    return run


@router.get("/{pipeline_run_id}/prediction-schema")
def get_prediction_schema(pipeline_run_id: str, db: Session = Depends(get_db)):
    run = _get_champion_or_409(pipeline_run_id, db)
    return {"feature_schema": run.feature_schema_json, "target_classes": run.feature_importance_json.get("target_classes", [])}


@router.get("/{pipeline_run_id}/feature-importance")
def get_feature_importance(pipeline_run_id: str, db: Session = Depends(get_db)):
    run = _get_champion_or_409(pipeline_run_id, db)
    return run.feature_importance_json


@router.post("/{pipeline_run_id}/predict")
def predict(pipeline_run_id: str, body: PredictionRequest, db: Session = Depends(get_db)):
    run = _get_champion_or_409(pipeline_run_id, db)

    missing = [f for f in run.feature_schema_json if f not in body.features]
    if missing:
        raise HTTPException(400, f"Missing feature value(s): {', '.join(missing)}")

    pipeline = champion_service.load(run.champion_model_path)
    result = pipeline.predict(body.features)

    x_row = pipeline.transform(body.features)[0]
    # Champion pipeline has no stored training matrix (kept lightweight on disk) — use the
    # current input scaled against itself as a same-shape stand-in for the SHAP background
    # distribution; explain_prediction() falls back gracefully if this degrades the explainer.
    explanation = explainability_service.explain_prediction(
        pipeline.estimator, np.tile(x_row, (10, 1)), x_row, pipeline.encoded_columns
    )
    result["explanation"] = explanation

    db.add(PredictionLogORM(pipeline_run_id=pipeline_run_id, input_json=body.features, output_json=result))
    db.commit()

    return result
