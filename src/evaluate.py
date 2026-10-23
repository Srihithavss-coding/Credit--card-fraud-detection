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


def _validated_labels_and_probabilities(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize inputs shared by the evaluation functions."""
    try:
        labels_array = np.asarray(labels, dtype=float)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("Labels must contain only numeric binary values 0 and 1.") from error
    if labels_array.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array.")
    if labels_array.size == 0:
        raise ValueError("Labels and probabilities must not be empty.")
    if not np.isfinite(labels_array).all() or not np.isin(labels_array, [0, 1]).all():
        raise ValueError("Labels must contain only finite binary values 0 and 1.")

    try:
        probabilities_array = np.asarray(probabilities, dtype=float)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("Probabilities must be a numeric one-dimensional array.") from error
    if probabilities_array.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional array.")
    if probabilities_array.size != labels_array.size:
        raise ValueError("Labels and probabilities must have matching lengths.")
    if not np.isfinite(probabilities_array).all():
        raise ValueError("Probabilities must contain only finite values.")
    if ((probabilities_array < 0) | (probabilities_array > 1)).any():
        raise ValueError("Probabilities must be within the interval [0, 1].")

    return labels_array.astype(np.int8), probabilities_array


def _validated_threshold(threshold: float) -> float:
    """Return a scalar threshold after checking its valid range."""
    try:
        threshold_array = np.asarray(threshold)
        if threshold_array.ndim != 0:
            raise ValueError
        threshold_value = float(threshold_array)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("Threshold must be a finite scalar within [0, 1].") from error
    if not np.isfinite(threshold_value) or not 0 <= threshold_value <= 1:
        raise ValueError("Threshold must be a finite scalar within [0, 1].")
    return threshold_value


def threshold_metrics(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels_array, probabilities_array = _validated_labels_and_probabilities(
        labels, probabilities
    )
    threshold_value = _validated_threshold(threshold)
    predictions = probabilities_array >= threshold_value
    tn, fp, fn, tp = confusion_matrix(
        labels_array, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": threshold_value,
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels_array, predictions, beta=2, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alert_volume": int(predictions.sum()),
    }


def select_f2_threshold(labels: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose the F2-maximizing threshold using validation labels only."""
    labels_array, probabilities = _validated_labels_and_probabilities(
        labels, probabilities
    )
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
    labels_array, probabilities_array = _validated_labels_and_probabilities(
        labels, probabilities
    )
    threshold_value = _validated_threshold(threshold)
    if np.unique(labels_array).size != 2:
        raise ValueError("Test metrics require both binary label values 0 and 1.")

    selected = threshold_metrics(labels_array, probabilities_array, threshold_value)
    selected.update(
        {
            "roc_auc": float(roc_auc_score(labels_array, probabilities_array)),
            "average_precision": float(
                average_precision_score(labels_array, probabilities_array)
            ),
            "fraud_prevalence": float(labels_array.mean()),
            "test_fraud_count": int(labels_array.sum()),
        }
    )
    predictions = probabilities_array >= threshold_value
    matrix = confusion_matrix(labels_array, predictions, labels=[0, 1])
    report = classification_report(
        labels_array,
        predictions,
        labels=[0, 1],
        target_names=["legitimate", "fraud"],
        output_dict=True,
        zero_division=0,
    )
    return selected, matrix, report
