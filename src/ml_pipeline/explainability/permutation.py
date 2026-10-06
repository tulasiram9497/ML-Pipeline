from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance


class PermutationImportance:
    """Model-agnostic feature importance using permutation."""

    @staticmethod
    def evaluate(model: Any, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, random_state: int = 42):
        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
        return {name: float(score) for name, score in zip(X.columns, result.importances_mean)}
