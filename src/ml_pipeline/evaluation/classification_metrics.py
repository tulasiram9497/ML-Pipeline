from __future__ import annotations

import numpy as np
from sklearn import metrics


def compute_classification_metrics(y_true, y_pred, y_proba=None):
    metrics_dict = {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(metrics.log_loss(y_true, y_proba, labels=np.unique(y_true))) if y_proba is not None else None,
        "confusion_matrix": metrics.confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        try:
            metrics_dict["roc_auc"] = float(metrics.roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba, multi_class="ovr", average="macro"))
        except Exception:
            metrics_dict["roc_auc"] = None

    return metrics_dict
