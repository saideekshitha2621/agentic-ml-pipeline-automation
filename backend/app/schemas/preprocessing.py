from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PreprocessingPlan(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    numerical_columns: list[str]
    categorical_columns: list[str]
    dropped_columns: list[str]
    numerical_impute_strategy: str
    categorical_impute_strategy: str
    scaling_method: str
    drop_duplicates: bool
    pca_enabled: bool
    pca_variance_target: float


class PreprocessingPlanUpdate(BaseModel):
    numerical_columns: list[str]
    categorical_columns: list[str]
    dropped_columns: list[str] = []
    numerical_impute_strategy: str = "median"
    categorical_impute_strategy: str = "mode"
    scaling_method: str = "standard"  # standard | minmax | robust | none
    drop_duplicates: bool = True
    pca_enabled: bool = False
    pca_variance_target: float = 0.95


class PreprocessingReport(BaseModel):
    initial_shape: list[int]
    final_shape: list[int]
    duplicates_found: int
    duplicates_removed: int
    imputation: dict
    scaling: dict
