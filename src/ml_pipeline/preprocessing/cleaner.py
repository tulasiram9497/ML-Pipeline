from __future__ import annotations

import pandas as pd


class DataCleaner:
    """Basic cleaning utilities for tabular ML data."""

    @staticmethod
    def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, df.nunique(dropna=True) > 1]

    @staticmethod
    def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(method="ffill").fillna(method="bfill")

    @staticmethod
    def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().copy()
