import pandas as pd
import pytest


@pytest.fixture
def classification_df():
    data = {
        "age": [22, 25, 30, 45, 41, 32, 38, 29, 50, 52, 28, 35, 44, 60, 23, 27, 48, 39, 54, 33],
        "income": [30000, 29000, 42000, 60000, 62000, 45000, 50000, 33000, 68000, 71000, 31000, 47000, 65000, 90000, 28000, 36000, 72000, 52000, 82000, 49000],
        "gender": ["F", "M", "F", "M", "F", "M", "M", "F", "F", "M", "F", "F", "M", "M", "F", "M", "F", "M", "F", "M"],
        "target": [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def regression_df():
    data = {
        "feature_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "feature_b": [10, 11, 9, 14, 8, 17, 15, 19, 20, 22, 18, 23],
        "target": [20, 24, 28, 30, 32, 39, 42, 48, 50, 56, 61, 63],
    }
    return pd.DataFrame(data)
