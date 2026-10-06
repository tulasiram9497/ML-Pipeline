from __future__ import annotations

from fastapi import FastAPI

from ml_pipeline.api.schemas import HealthResponse, PredictRequest, ProfileRequest, TrainRequest
from ml_pipeline.data.loader import DataLoader
from ml_pipeline.data.validator import DataValidator
from ml_pipeline.tasks.detector import TaskDetector

app = FastAPI(title="AutoML Production Pipeline API")


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return {"models": ["logistic_regression", "random_forest", "decision_tree"]}


@app.post("/train")
def train(request: TrainRequest) -> dict:
    df = DataLoader.load_dataset(request.data_path)
    task = TaskDetector.detect(df, request.target, override=request.task)
    validation = DataValidator.validate(df, request.target)
    if not validation.valid:
        raise ValueError("Dataset validation failed: " + "; ".join(validation.issues))
    return {"status": "trained", "task": task, "target": request.target}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    return {"status": "ok", "prediction": "model output placeholder"}


@app.post("/evaluate")
def evaluate() -> dict:
    return {"status": "ok", "evaluation": {"score": 0.0}}


@app.get("/metrics")
def metrics() -> dict:
    return {"metrics": {"accuracy": 0.0}}


@app.get("/feature-importance")
def feature_importance() -> dict:
    return {"feature_importance": {}}


@app.post("/profile-dataset")
def profile_dataset(request: ProfileRequest) -> dict:
    df = DataLoader.load_dataset(request.data_path)
    return DataLoader.describe_dataset(df)
