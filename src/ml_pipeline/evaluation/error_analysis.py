from __future__ import annotations

import pandas as pd


class ErrorAnalyzer:
    """Generate a lightweight error analysis summary."""

    @staticmethod
    def summarize(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        comparison = pd.DataFrame({"actual": y_true, "predicted": y_pred})
        comparison["error"] = comparison["actual"] - comparison["predicted"]
        return {
            "mean_error": float(comparison["error"].mean()),
            "max_abs_error": float(comparison["error"].abs().max()),
            "false_positive_rate": float((comparison["actual"] == 0) & (comparison["predicted"] == 1)).mean(),
        }
