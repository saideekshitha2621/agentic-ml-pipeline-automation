from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Dataset(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    uploaded_at: datetime
    n_rows: int
    n_columns: int
    data_quality_score: float


class MissingValueRow(BaseModel):
    column: str
    dtype: str
    missing_count: int
    missing_pct: float
    n_unique: int


class OutlierRow(BaseModel):
    column: str
    method: str
    n_outliers: int
    pct_outliers: float
    lower_bound: float | None = None
    upper_bound: float | None = None


class CategorySuggestion(BaseModel):
    column: str
    issue: str
    examples: list[str]
    suggestion: str


class DataProfile(BaseModel):
    n_rows: int
    n_columns: int
    n_duplicates: int
    data_quality_score: float
    missing_values: list[MissingValueRow]
    outliers: list[OutlierRow]
    category_suggestions: list[CategorySuggestion]
