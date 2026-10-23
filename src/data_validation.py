"""Dataset schema checks and a column-level data-quality audit."""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPECTED_FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
EXPECTED_COLUMNS = [*EXPECTED_FEATURES, "Class"]


def validate_schema(data: pd.DataFrame) -> None:
    """Raise a useful error when the input does not match the expected dataset."""
    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected schema. Expected the 31 standard ULB columns in order; "
            f"received {data.columns.tolist()}."
        )
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in data):
        raise TypeError("Every column in the expected ULB schema must be numeric.")
    if data["Class"].isna().any():
        raise ValueError("Target column 'Class' must not contain missing labels.")
    labels = set(data["Class"].dropna().unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected target labels {{0, 1}}, found {labels}.")


def build_quality_audit(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize values and flag anomalies without automatically removing them."""
    rows = []
    for column in data.columns:
        series = data[column]
        numeric = pd.to_numeric(series, errors="coerce")
        non_finite = int((~np.isfinite(numeric.dropna().to_numpy())).sum())
        unique = int(series.nunique(dropna=True))
        notes = []
        if unique == 1:
            notes.append("constant")
        if non_finite:
            notes.append(f"{non_finite} non-finite values")
        if column == "Amount" and (numeric.dropna() < 0).any():
            notes.append("negative amounts require investigation")
        if column == "Time" and (numeric.dropna() < 0).any():
            notes.append("negative elapsed times require investigation")
        if column == "Class" and set(series.dropna().unique()) != {0, 1}:
            notes.append("unexpected target labels")
        rows.append(
            {
                "column": column,
                "data_type": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_percentage": float(series.isna().mean() * 100),
                "unique_count": unique,
                "minimum": numeric.min(),
                "maximum": numeric.max(),
                "duplicate_or_anomaly_notes": "; ".join(notes) or "none detected",
                "action_taken": "none",
                "reason": "Values retained; no column-level cleaning indicated."
                if not notes
                else "Flagged for investigation; not removed automatically.",
            }
        )
    return pd.DataFrame(rows)
