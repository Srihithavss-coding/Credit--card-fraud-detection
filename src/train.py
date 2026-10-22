"""Chronological splitting and leakage-safe model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_SEED = 42
FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Amount_log1p",
]


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add a deterministic, nonnegative-amount log feature without fitting."""
    features = data.drop(columns="Class").copy()
    if (features["Amount"] < 0).any():
        raise ValueError("Amount_log1p requires nonnegative amounts.")
    features["Amount_log1p"] = np.log1p(features["Amount"])
    return features.loc[:, FEATURE_COLUMNS]


def chronological_split(data: pd.DataFrame) -> tuple[DataSplits, pd.DataFrame]:
    """Split ordered unique observations 60/20/20 without stratifying time."""
    if not data["Time"].is_monotonic_increasing:
        raise ValueError("Chronological split requires rows ordered by Time.")
    features = make_features(data)
    labels = data["Class"].reset_index(drop=True)
    n = len(data)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    boundaries = [0, train_end, validation_end, n]
    slices = [slice(boundaries[i], boundaries[i + 1]) for i in range(3)]
    splits = DataSplits(
        features.iloc[slices[0]].reset_index(drop=True),
        labels.iloc[slices[0]].reset_index(drop=True),
        features.iloc[slices[1]].reset_index(drop=True),
        labels.iloc[slices[1]].reset_index(drop=True),
        features.iloc[slices[2]].reset_index(drop=True),
        labels.iloc[slices[2]].reset_index(drop=True),
    )
    return splits, features


def build_models() -> dict[str, object]:
    """Return compact, pre-specified model configurations."""
    return {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=RANDOM_SEED,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        ),
    }


def fit_models(splits: DataSplits) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit on training only and compare validation ranking and 0.5 metrics."""
    models = {
        name: estimator
        for name, estimator in build_models().items()
        if name != "Dummy (prior)"
    }
    rows = []
    for name, estimator in models.items():
        if name == "HistGradientBoosting":
            weights = compute_sample_weight("balanced", splits.y_train)
            estimator.fit(splits.X_train, splits.y_train, sample_weight=weights)
        else:
            estimator.fit(splits.X_train, splits.y_train)
        probabilities = estimator.predict_proba(splits.X_validation)[:, 1]
        rows.append(
            {
                "model": name,
                "validation_roc_auc": roc_auc_score(
                    splits.y_validation, probabilities
                ),
                "validation_average_precision": average_precision_score(
                    splits.y_validation, probabilities
                ),
                "precision_at_0_5": precision_score(
                    splits.y_validation, probabilities >= 0.5, zero_division=0
                ),
                "recall_at_0_5": recall_score(
                    splits.y_validation, probabilities >= 0.5, zero_division=0
                ),
                "f1_at_0_5": f1_score(
                    splits.y_validation, probabilities >= 0.5, zero_division=0
                ),
                "configuration": str(estimator),
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        "validation_average_precision", ascending=False
    ).reset_index(drop=True)
    return models, comparison
