from __future__ import annotations

from typing import Any

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


class MLFlowTracker:
    """Thin wrapper around MLflow for experiment tracking."""

    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri
        self.enabled = mlflow is not None
        if self.enabled:
            mlflow.set_tracking_uri(tracking_uri or "file://./mlruns")

    def log_params(self, params: dict[str, Any]):
        if self.enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]):
        if self.enabled:
            mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_path: str = "model"):
        if self.enabled:
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
