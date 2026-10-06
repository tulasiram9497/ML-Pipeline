from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score


class BaselineModel:
    """Create baseline models and report performance."""

    @staticmethod
    def build(task: str):
        if task == "classification":
            return DummyClassifier(strategy="most_frequent")
        if task == "regression":
            return DummyRegressor(strategy="median")
        raise ValueError(f"Baseline not supported for task '{task}'.")

    @staticmethod
    def evaluate(task: str, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        model = BaselineModel.build(task)
        if task == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for train_idx, valid_idx in cv.split(X, y):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = model.predict(X.iloc[valid_idx])
                scores.append(f1_score(y.iloc[valid_idx], pred, average="weighted"))
            return {"metric": "f1_weighted", "score": float(sum(scores) / len(scores)), "scores": scores}

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores: list[float] = []
        for train_idx, valid_idx in cv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[valid_idx])
            scores.append(r2_score(y.iloc[valid_idx], pred))
        return {"metric": "r2", "score": float(sum(scores) / len(scores)), "scores": scores}
