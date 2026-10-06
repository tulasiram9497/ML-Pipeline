from __future__ import annotations

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    data_path: str
    target: str
    task: str | None = None
    model_name: str = "logistic_regression"


class PredictRequest(BaseModel):
    model_path: str
    payload: dict


class ProfileRequest(BaseModel):
    data_path: str


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
