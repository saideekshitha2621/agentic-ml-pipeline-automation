from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Dataset as DatasetORM
from app.db.models import PreprocessingPlanORM
from app.schemas.pca import PCAApplyRequest, PCAPreview, PCAReport
from app.services import pca_service, preprocessing_service

router = APIRouter(prefix="/api/v1/datasets", tags=["pca"])


def _encoded_df_for_active_plan(dataset_id: str, db: Session) -> pd.DataFrame:
    dataset = db.get(DatasetORM, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    plan_orm = (
        db.query(PreprocessingPlanORM)
        .filter_by(dataset_id=dataset_id, is_active=True)
        .order_by(PreprocessingPlanORM.created_at.desc())
        .first()
    )
    if not plan_orm:
        raise HTTPException(409, "No preprocessing plan saved yet — visit Preprocessing Review first.")

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
    _cleaned, encoded, _report = preprocessing_service.apply_plan(df, plan)
    return encoded


@router.get("/{dataset_id}/pca-preview", response_model=PCAPreview)
def pca_preview(dataset_id: str, db: Session = Depends(get_db)):
    encoded_df = _encoded_df_for_active_plan(dataset_id, db)
    return pca_service.preview(encoded_df)


@router.post("/{dataset_id}/pca", response_model=PCAReport)
def pca_apply(dataset_id: str, body: PCAApplyRequest, db: Session = Depends(get_db)):
    encoded_df = _encoded_df_for_active_plan(dataset_id, db)
    _pca_df, report = pca_service.apply(encoded_df, body.n_components)
    return report
