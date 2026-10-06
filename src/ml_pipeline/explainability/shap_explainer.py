from __future__ import annotations

from typing import Any

import pandas as pd


class ShapExplainer:
    """Attempt to compute SHAP values when the optional dependency is available."""

    @staticmethod
    def explain(model: Any, X: pd.DataFrame):
        try:
            import shap
        except ImportError:
            return {"status": "skipped", "reason": "SHAP is not installed."}

        explainer = shap.Explainer(model, X)
        values = explainer(X)
        return {"status": "ok", "values": values}
