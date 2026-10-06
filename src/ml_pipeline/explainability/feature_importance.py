from __future__ import annotations

from typing import Any


class FeatureImportance:
    """Extract feature importance from tree-based models."""

    @staticmethod
    def from_model(model: Any, feature_names: list[str]) -> dict[str, float]:
        if not hasattr(model, "feature_importances_"):
            return {name: 0.0 for name in feature_names}
        importance = model.feature_importances_
        return {name: float(score) for name, score in zip(feature_names, importance)}
