from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluate import select_f2_threshold, threshold_metrics


def test_threshold_metrics_returns_precision_recall_f2_and_confusion_counts() -> None:
    labels = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.3, 0.9])

    metrics = threshold_metrics(labels, probabilities, threshold=0.5)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f2"] == pytest.approx(0.5)
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1


def test_threshold_metrics_handles_no_predicted_positives() -> None:
    metrics = threshold_metrics(
        pd.Series([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]), threshold=0.5
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f2"] == 0.0
    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 2
    assert metrics["true_positives"] == 0


def test_select_f2_threshold_handles_zero_precision_and_recall() -> None:
    threshold, table = select_f2_threshold(
        pd.Series([0, 1]), np.array([0.9, 0.1])
    )

    zero_f2_row = table.loc[table["threshold"] == 0.9].iloc[0]
    assert threshold == pytest.approx(0.1)
    assert np.isfinite(table["f2"]).all()
    assert zero_f2_row["precision"] == 0.0
    assert zero_f2_row["recall"] == 0.0
    assert zero_f2_row["f2"] == 0.0


def test_select_f2_threshold_requires_at_least_one_positive_label() -> None:
    with pytest.raises(ValueError, match="without positive labels"):
        select_f2_threshold(pd.Series([0, 0]), np.array([0.1, 0.9]))
