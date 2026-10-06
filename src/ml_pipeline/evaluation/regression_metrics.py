from __future__ import annotations

import numpy as np
from sklearn import metrics


def compute_regression_metrics(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    return {
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(metrics.r2_score(y_true, y_pred)),
    }
