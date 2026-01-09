from __future__ import annotations

from typing import Dict, List

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(categorical_features: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(max_iter=1000, multi_class="auto")
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def save_model(path: str, pipeline: Pipeline, feature_columns: List[str]) -> None:
    bundle: Dict[str, object] = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
    }
    joblib.dump(bundle, path)


def load_model(path: str) -> Dict[str, object]:
    return joblib.load(path)
