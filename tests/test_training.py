import pandas as pd

from ml_pipeline.training.trainer import ModelTrainer


def test_model_trainer_classification(classification_df):
    result = ModelTrainer.train(classification_df, target="target", model_name="logistic_regression")
    assert result.task == "classification"
    assert "f1" in result.metrics
    assert result.metrics["f1"] >= 0.0
    assert result.cv_summary["mean"] >= 0.0


def test_model_trainer_regression(regression_df):
    result = ModelTrainer.train(regression_df, target="target", task="regression", model_name="linear_regression")
    assert result.task == "regression"
    assert "r2" in result.metrics
    assert result.metrics["r2"] <= 1.0
