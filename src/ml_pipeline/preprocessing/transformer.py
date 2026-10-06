from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureTransformer:
    """Create a reusable preprocessing pipeline."""

    @staticmethod
    def build_preprocessor(X: pd.DataFrame, task: str | None = None) -> ColumnTransformer:
        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = [
            col for col in X.columns if col not in numeric_features
        ]

        transformers: list[tuple[str, Pipeline, list[str]]] = []

        if numeric_features:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_pipeline, numeric_features))

        if categorical_features:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, categorical_features))

        return ColumnTransformer(transformers=transformers, remainder="drop")
