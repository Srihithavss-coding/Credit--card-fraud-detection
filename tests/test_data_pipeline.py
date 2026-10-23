from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_cleaning import clean_dataset
from data_validation import EXPECTED_COLUMNS, validate_schema
from train import chronological_split


def make_frame(times: list[float], labels: list[int]) -> pd.DataFrame:
    data = {column: [0.0] * len(times) for column in EXPECTED_COLUMNS}
    data["Time"] = times
    data["Class"] = labels
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


def test_validate_schema_accepts_expected_numeric_schema_and_binary_labels() -> None:
    validate_schema(make_frame([0.0, 1.0], [0, 1]))


def test_validate_schema_rejects_reordered_columns_and_invalid_target_labels() -> None:
    frame = make_frame([0.0, 1.0], [0, 1])
    with pytest.raises(ValueError, match="Unexpected schema"):
        validate_schema(frame[frame.columns[::-1]])

    frame.loc[1, "Class"] = 2
    with pytest.raises(ValueError, match="Expected target labels"):
        validate_schema(frame)


def test_validate_schema_rejects_missing_target_labels() -> None:
    frame = make_frame([0.0, 1.0, 2.0], [0, 1, 0])
    frame.loc[1, "Class"] = float("nan")
    with pytest.raises(ValueError, match="must not contain missing labels"):
        validate_schema(frame)


def test_clean_dataset_removes_exact_duplicates_without_mutating_input() -> None:
    frame = make_frame([0.0, 1.0, 1.0], [0, 1, 1])
    original = frame.copy(deep=True)

    cleaned, summary = clean_dataset(frame)

    assert_frame_equal(frame, original)
    assert len(cleaned) == 2
    assert summary["duplicate_rows_identified"] == 1
    assert summary["records_removed"] == 1
    assert cleaned["Class"].tolist() == [0, 1]


def test_chronological_split_is_ordered_and_separated() -> None:
    frame = make_frame(list(map(float, range(10))), [0, 1] * 5)

    splits, _ = chronological_split(frame)

    assert len(splits.X_train) == 6
    assert len(splits.X_validation) == 2
    assert len(splits.X_test) == 2
    assert splits.X_train["Time"].max() < splits.X_validation["Time"].min()
    assert splits.X_validation["Time"].max() < splits.X_test["Time"].min()
    assert splits.y_train.tolist() == [0, 1, 0, 1, 0, 1]
    assert splits.y_validation.tolist() == [0, 1]
    assert splits.y_test.tolist() == [0, 1]
