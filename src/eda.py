"""Small reusable summaries for exploratory analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def amount_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate amount distribution statistics separately by class."""
    rows = []
    for label, group in data.groupby("Class", sort=True):
        values = group["Amount"]
        rows.append(
            {
                "class": int(label),
                "class_name": "fraud" if label else "legitimate",
                "count": int(values.size),
                "mean": values.mean(),
                "median": values.median(),
                "standard_deviation": values.std(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "maximum": values.max(),
            }
        )
    return pd.DataFrame(rows)


def pca_feature_effects(data: pd.DataFrame) -> pd.DataFrame:
    """Rank anonymized PCA features by standardized mean difference."""
    fraud = data.loc[data["Class"] == 1]
    legitimate = data.loc[data["Class"] == 0]
    rows = []
    for column in (f"V{i}" for i in range(1, 29)):
        f_values = fraud[column]
        l_values = legitimate[column]
        pooled = np.sqrt(
            (
                (len(f_values) - 1) * f_values.var()
                + (len(l_values) - 1) * l_values.var()
            )
            / (len(f_values) + len(l_values) - 2)
        )
        statistic = ks_2samp(f_values, l_values)
        rows.append(
            {
                "feature": column,
                "legitimate_mean": l_values.mean(),
                "fraud_mean": f_values.mean(),
                "legitimate_median": l_values.median(),
                "fraud_median": f_values.median(),
                "standardized_mean_difference": (
                    (f_values.mean() - l_values.mean()) / pooled
                    if pooled
                    else 0.0
                ),
                "ks_statistic": statistic.statistic,
            }
        )
    return pd.DataFrame(rows).assign(
        absolute_effect=lambda table: table[
            "standardized_mean_difference"
        ].abs()
    ).sort_values("absolute_effect", ascending=False)


def time_bin_summary(data: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Return counts and fraud rate in equal-width elapsed-time intervals."""
    intervals = pd.cut(data["Time"], bins=bins, include_lowest=True)
    summary = (
        data.assign(time_interval=intervals)
        .groupby("time_interval", observed=False)["Class"]
        .agg(transaction_count="size", fraud_count="sum", fraud_rate="mean")
        .reset_index()
    )
    summary["time_start_seconds"] = summary["time_interval"].map(
        lambda interval: float(interval.left)
    )
    summary["time_end_seconds"] = summary["time_interval"].map(
        lambda interval: float(interval.right)
    )
    return summary
