from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


class CrossValidator:
    """Cross-validation helper with sensible defaults."""

    @staticmethod
    def stratified_kfold(n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @staticmethod
    def kfold(n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @staticmethod
    def evaluate(model: Any, X: Any, y: Any, task: str, n_splits: int = 5):
        if task == "classification":
            cv = CrossValidator.stratified_kfold(n_splits=n_splits)
            scores = []
            for train_idx, valid_idx in cv.split(X, y):
                X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
                X_valid = X.iloc[valid_idx] if hasattr(X, "iloc") else X[valid_idx]
                y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
                y_valid = y.iloc[valid_idx] if hasattr(y, "iloc") else y[valid_idx]
                model.fit(X_train, y_train)
                prediction = model.predict(X_valid)
                scores.append(float(np.mean(prediction == y_valid)))
            return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores}

        cv = CrossValidator.kfold(n_splits=n_splits)
        scores = []
        for train_idx, valid_idx in cv.split(X):
            X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            X_valid = X.iloc[valid_idx] if hasattr(X, "iloc") else X[valid_idx]
            y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            y_valid = y.iloc[valid_idx] if hasattr(y, "iloc") else y[valid_idx]
            model.fit(X_train, y_train)
            prediction = model.predict(X_valid)
            scores.append(float(np.mean((prediction - y_valid) ** 2)))
        return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores}
