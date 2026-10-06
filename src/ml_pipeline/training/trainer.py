from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_pipeline.data.validator import DataValidator
from ml_pipeline.evaluation.classification_metrics import compute_classification_metrics
from ml_pipeline.evaluation.regression_metrics import compute_regression_metrics
from ml_pipeline.models.registry import get_model
from ml_pipeline.preprocessing.transformer import FeatureTransformer
from ml_pipeline.tasks.detector import TaskDetector
from ml_pipeline.training.cross_validation import CrossValidator


@dataclass
class TrainResult:
    task: str
    model_name: str
    metrics: dict[str, Any]
    cv_summary: dict[str, Any]
    fitted_model: Any
    preprocessor: Any


class ModelTrainer:
    """Train a supervised model with preprocessing and evaluation."""

    @staticmethod
    def train(
        df: pd.DataFrame,
        target: str,
        task: str | None = None,
        model_name: str = "logistic_regression",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainResult:
        validation = DataValidator.validate(df, target=target)
        if not validation.valid:
            raise ValueError("Dataset validation failed: " + "; ".join(validation.issues))

        resolved_task = TaskDetector.detect(df, target, override=task)
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if resolved_task == "classification" else None
        )

        preprocessor = FeatureTransformer.build_preprocessor(X_train, resolved_task)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        model = get_model(resolved_task, model_name)
        model.fit(X_train_processed, y_train)

        predictions = model.predict(X_test_processed)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test_processed)
        else:
            probas = None

        if resolved_task == "classification":
            metrics = compute_classification_metrics(y_test, predictions, probas)
        else:
            metrics = compute_regression_metrics(y_test, predictions)

        cv_summary = CrossValidator.evaluate(
            model,
            X_train_processed,
            y_train,
            resolved_task,
            n_splits=5,
        )

        return TrainResult(
            task=resolved_task,
            model_name=model_name,
            metrics=metrics,
            cv_summary=cv_summary,
            fitted_model=model,
            preprocessor=preprocessor,
        )
