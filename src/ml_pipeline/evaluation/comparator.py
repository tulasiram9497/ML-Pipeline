from __future__ import annotations

from typing import Any


class ModelComparator:
    """Rank model results by a primary metric."""

    @staticmethod
    def rank(results: list[dict[str, Any]], metric_key: str = "f1") -> list[dict[str, Any]]:
        ranked = sorted(results, key=lambda item: item["metrics"].get(metric_key, float("-inf")), reverse=True)
        for index, result in enumerate(ranked, start=1):
            result["rank"] = index
        return ranked
