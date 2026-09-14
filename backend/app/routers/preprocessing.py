from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Dataset as DatasetORM
from app.db.models import PreprocessingPlanORM
from app.schemas.preprocessing import PreprocessingPlan, PreprocessingPlanUpdate, PreprocessingReport
from app.services import preprocessing_service

router = APIRouter(prefix="/api/v1/datasets", tags=["preprocessing"])


def _get_dataset_or_404(dataset_id: str, db: Session) -> DatasetORM:
    dataset = db.get(DatasetORM, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    return dataset


@router.get("/{dataset_id}/preprocessing-plan", response_model=PreprocessingPlan)
def get_plan(dataset_id: str, db: Session = Depends(get_db)):
    dataset = _get_dataset_or_404(dataset_id, db)

    existing = (
        db.query(PreprocessingPlanORM)
        .filter_by(dataset_id=dataset_id, is_active=True)
        .order_by(PreprocessingPlanORM.created_at.desc())
        .first()
    )
    if existing:
        return existing

    df = pd.read_csv(dataset.storage_path)
    default = preprocessing_service.detect_default_plan(df)
    plan = PreprocessingPlanORM(dataset_id=dataset_id, **default)
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


@router.put("/{dataset_id}/preprocessing-plan", response_model=PreprocessingPlan)
def update_plan(dataset_id: str, body: PreprocessingPlanUpdate, db: Session = Depends(get_db)):
    _get_dataset_or_404(dataset_id, db)

    db.query(PreprocessingPlanORM).filter_by(dataset_id=dataset_id, is_active=True).update(
        {"is_active": False}
    )
    plan = PreprocessingPlanORM(dataset_id=dataset_id, is_active=True, **body.model_dump())
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


@router.post("/{dataset_id}/preprocess", response_model=PreprocessingReport)
def apply_preprocessing(dataset_id: str, plan_id: str, db: Session = Depends(get_db)):
    dataset = _get_dataset_or_404(dataset_id, db)
    plan_orm = db.get(PreprocessingPlanORM, plan_id)
    if not plan_orm or plan_orm.dataset_id != dataset_id:
        raise HTTPException(404, "Preprocessing plan not found for this dataset.")

    df = pd.read_csv(dataset.storage_path)
    plan = {
        "numerical_columns": plan_orm.numerical_columns,
        "categorical_columns": plan_orm.categorical_columns,
        "dropped_columns": plan_orm.dropped_columns,
        "numerical_impute_strategy": plan_orm.numerical_impute_strategy,
        "categorical_impute_strategy": plan_orm.categorical_impute_strategy,
        "scaling_method": plan_orm.scaling_method,
        "drop_duplicates": plan_orm.drop_duplicates,
    }
    _cleaned, _encoded, report = preprocessing_service.apply_plan(df, plan)
    return report
