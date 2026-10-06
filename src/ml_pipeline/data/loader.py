from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ml_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    numerical_features: list[str]
    categorical_features: list[str]
    datetime_features: list[str]
    potential_id_columns: list[str]
    missing_values: dict[str, float]
    duplicate_ratio: float
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    summary: dict[str, Any] = field(default_factory=dict)


class DataLoader:
    """Reusable data loader for tabular datasets."""

    @staticmethod
    def load_dataset(path: str | Path) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported dataset format: {suffix or 'unknown'}")

        logger.info("Loaded dataset with shape %s", df.shape)
        return df

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
        categorical_cols = [
            col
            for col in df.columns
            if col not in numeric_cols and col not in datetime_cols and df[col].dtype == "object"
        ]

        missing_values = {
            col: round(float(df[col].isna().mean() * 100), 3)
            for col in df.columns
            if df[col].isna().sum() > 0
        }
        duplicate_ratio = round(float(df.duplicated().mean() * 100), 3)
        constant_columns = [
            col for col in df.columns if df[col].nunique(dropna=True) <= 1
        ]
        high_cardinality_columns = [
            col for col in categorical_cols if df[col].nunique(dropna=True) > max(10, len(df) * 0.1)
        ]
        candidates = []
        for col in df.columns:
            if df[col].nunique(dropna=True) == len(df) and col not in datetime_cols:
                candidates.append(col)
        potential_id_columns = candidates

        return DatasetProfile(
            rows=int(len(df)),
            columns=int(df.shape[1]),
            numerical_features=numeric_cols,
            categorical_features=categorical_cols,
            datetime_features=datetime_cols,
            potential_id_columns=potential_id_columns,
            missing_values=missing_values,
            duplicate_ratio=duplicate_ratio,
            constant_columns=constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            summary={
                "dataset_shape": df.shape,
                "column_types": {
                    "numeric": numeric_cols,
                    "categorical": categorical_cols,
                    "datetime": datetime_cols,
                },
                "missing_values_total": int(df.isna().sum().sum()),
            },
        )

    @staticmethod
    def describe_dataset(df: pd.DataFrame) -> dict[str, Any]:
        profile = DataLoader.profile_dataset(df)
        return {
            "dataset_shape": {"rows": profile.rows, "columns": profile.columns},
            "numerical_features": len(profile.numerical_features),
            "categorical_features": len(profile.categorical_features),
            "datetime_features": len(profile.datetime_features),
            "potential_id_columns": len(profile.potential_id_columns),
            "missing_values": profile.missing_values,
            "duplicate_ratio": profile.duplicate_ratio,
            "constant_columns": profile.constant_columns,
            "high_cardinality_columns": profile.high_cardinality_columns,
        }
