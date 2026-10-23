"""Validation threshold selection and one-shot test evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_metrics(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alert_volume": int(predictions.sum()),
    }


def select_f2_threshold(labels: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose the F2-maximizing threshold using validation labels only."""
    labels_array = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    if not labels_array.any():
        raise ValueError("Cannot select an F2 threshold without positive labels.")
    precision, recall, thresholds = precision_recall_curve(
        labels_array, probabilities
    )
    beta_squared = 4.0
    f2_denominator = beta_squared * precision[:-1] + recall[:-1]
    f2 = np.divide(
        (1 + beta_squared) * precision[:-1] * recall[:-1],
        f2_denominator,
        out=np.zeros_like(f2_denominator),
        where=f2_denominator > 0,
    )
    predicted_positive = len(probabilities) - np.searchsorted(
        np.sort(probabilities), thresholds, side="left"
    )
    true_positives = np.rint(recall[:-1] * labels_array.sum()).astype(int)
    false_positives = predicted_positive - true_positives
    false_negatives = int(labels_array.sum()) - true_positives
    true_negatives = len(labels_array) - predicted_positive - false_negatives
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
            "f1": 2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-15),
            "f2": f2,
            "specificity": true_negatives
            / np.maximum(true_negatives + false_positives, 1),
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
            "alert_volume": predicted_positive,
        }
    )
    best_f2 = table["f2"].max()
    threshold = float(table.loc[table["f2"] == best_f2, "threshold"].max())
    return threshold, table


def test_metrics(
    labels: pd.Series, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, float | int], np.ndarray, dict[str, dict[str, float]]]:
    """Compute final metrics once after model and threshold selection."""
    selected = threshold_metrics(labels, probabilities, threshold)
    selected.update(
        {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "fraud_prevalence": float(labels.mean()),
            "test_fraud_count": int(labels.sum()),
        }
    )
    matrix = confusion_matrix(labels, probabilities >= threshold, labels=[0, 1])
    report = classification_report(
        labels,
        probabilities >= threshold,
        labels=[0, 1],
        target_names=["legitimate", "fraud"],
        output_dict=True,
        zero_division=0,
    )
    return selected, matrix, report
