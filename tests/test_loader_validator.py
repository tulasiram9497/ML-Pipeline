from pathlib import Path

import pandas as pd

from ml_pipeline.data.loader import DataLoader
from ml_pipeline.data.validator import DataValidator
from ml_pipeline.tasks.detector import TaskDetector


def test_loader_profiles_dataset(tmp_path):
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": ["x", "y", "z"],
        "target": [0, 1, 0],
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)

    loaded = DataLoader.load_dataset(path)
    profile = DataLoader.profile_dataset(loaded)

    assert loaded.shape == (3, 3)
    assert profile.rows == 3
    assert profile.columns == 3
    assert len(profile.numerical_features) == 2


def test_validator_reports_issues_on_empty_or_bad_target():
    df = pd.DataFrame({"feature": [1, 2], "target": [None, None]})
    result = DataValidator.validate(df, target="target")
    assert result.valid is False


def test_task_detector_detects_classification_and_clustering():
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    assert TaskDetector.detect(df, target="target") == "classification"
    assert TaskDetector.detect(pd.DataFrame({"feature": [1, 2, 3]}), target=None) == "clustering"
