from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataValidator:
    """Validate ML datasets before training."""

    @staticmethod
    def validate(df: pd.DataFrame, target: str | None = None) -> ValidationResult:
        issues: list[str] = []
        warnings: list[str] = []

        if df.empty:
            issues.append("Dataset is empty.")

        if df.duplicated().any():
            warnings.append(f"Dataset contains {int(df.duplicated().sum())} duplicate rows.")

        if target is not None and target not in df.columns:
            issues.append(f"Target column '{target}' is missing from the dataset.")

        if target is not None and df[target].isna().any():
            issues.append(f"Target column '{target}' contains missing values.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col].replace([np.inf, -np.inf], np.nan)).any():
                issues.append(f"Column '{col}' contains infinite values.")

        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                warnings.append(f"Column '{col}' has near-zero variance or is constant.")

        if target is not None and df[target].dtype.kind in "biufc":
            unique_count = df[target].nunique(dropna=True)
            if unique_count == 1:
                issues.append("Target has only one unique value; supervised learning is not meaningful.")
        else:
            if target is not None:
                unique_count = df[target].nunique(dropna=True)
                if unique_count <= 1:
                    issues.append("Target has only one unique value; supervised learning is not meaningful.")

        for col in df.columns:
            if df[col].dtype == object and df[col].astype(str).str.contains("\n").any():
                warnings.append(f"Column '{col}' may contain unexpected category formatting.")

        return ValidationResult(valid=not issues, issues=issues, warnings=warnings)

    @staticmethod
    def check_feature_leakage(df: pd.DataFrame, target: str) -> list[str]:
        risks: list[str] = []
        for col in df.columns:
            if col == target:
                continue
            if df[col].nunique(dropna=True) == len(df):
                risks.append(f"Column '{col}' looks like an identifier and may leak information.")
        return risks
