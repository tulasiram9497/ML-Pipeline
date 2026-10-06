from __future__ import annotations

import pandas as pd


class TaskDetector:
    """Detect supervised or unsupervised task from the target column."""

    @staticmethod
    def detect(df: pd.DataFrame, target: str | None = None, override: str | None = None) -> str:
        if override:
            return override.lower()
        if target is None:
            return "clustering"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")

        series = df[target]
        unique_count = series.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(series):
            if unique_count <= 20:
                return "classification"
            return "regression"

        return "classification"

    @staticmethod
    def validate_supported(task: str) -> str:
        normalized = task.lower()
        if normalized not in {"classification", "regression", "clustering"}:
            raise ValueError(f"Unsupported task: {task!r}")
        return normalized
