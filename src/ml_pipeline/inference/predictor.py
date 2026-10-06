from __future__ import annotations

from typing import Any

import pandas as pd


class Predictor:
    """Reusable prediction layer for trained models and preprocessors."""

    def __init__(self, model: Any, preprocessor: Any, feature_columns: list[str], task: str):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns
        self.task = task

    def predict(self, data: pd.DataFrame | dict):
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame([data])
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        transformed = self.preprocessor.transform(frame[self.feature_columns])
        return self.model.predict(transformed)

    def predict_proba(self, data: pd.DataFrame | dict):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("This model does not support prediction probabilities.")
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame([data])
        transformed = self.preprocessor.transform(frame[self.feature_columns])
        return self.model.predict_proba(transformed)
