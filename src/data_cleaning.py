"""Reproducible, minimal cleaning for the ULB benchmark dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_validation import validate_schema


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Validate, remove exact duplicate rows, and preserve original order."""
    validate_schema(data)
    before = len(data)
    missing_before = int(data.isna().sum().sum())
    non_finite = int(
        (~np.isfinite(data.select_dtypes(include=[np.number]).to_numpy())).sum()
    )
    if missing_before:
        raise ValueError(
            f"Found {missing_before} missing values; investigate rather than "
            "imputing blindly."
        )
    if non_finite:
        raise ValueError(f"Found {non_finite} non-finite numerical values.")
    if (data["Time"] < 0).any() or (data["Amount"] < 0).any():
        raise ValueError("Negative elapsed times or amounts require investigation.")

    duplicate_count = int(data.duplicated(keep="first").sum())
    cleaned = data.drop_duplicates(keep="first").copy().reset_index(drop=True)
    if not cleaned["Time"].is_monotonic_increasing:
        raise ValueError("Rows are not ordered by elapsed Time.")
    summary: dict[str, int | str] = {
        "original_rows": before,
        "duplicate_rows_identified": duplicate_count,
        "records_removed": before - len(cleaned),
        "final_rows": len(cleaned),
        "missing_values_before": missing_before,
        "missing_values_after": int(cleaned.isna().sum().sum()),
        "non_finite_values": non_finite,
        "cleaning_decision": (
            "Removed exact duplicate rows, keeping the first occurrence; no "
            "imputation, outlier removal, scaling, or target modification."
        ),
    }
    return cleaned, summary
