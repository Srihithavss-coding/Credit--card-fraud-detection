# Final project summary

- Dataset: 284,807 raw rows, 31 columns; 492 raw fraud labels (0.1727%); after duplicate removal 473 (0.1667%).
- Cleaning: removed 1,081 exact duplicate rows (19 fraud, 1062 legitimate); no missing values imputed; no non-duplicate outliers removed; raw CSV preserved.
- Chronological split after cleaning: train 170,235 rows / 342 fraud; validation 56,745 / 57; test 56,746 / 74.
- Selected model: HistGradientBoosting; validation average precision 0.77432.
- Final test: ROC-AUC 0.9860; average precision 0.7764; precision 0.8462; recall 0.7432; F1 0.7914; specificity 0.9998.
- Validation-selected threshold: 0.90566246 (F2 objective; not cost- or capacity-optimized). Test confusion: TN=56,662, FP=10, FN=19, TP=55.
- Leading validation permutation-importance features: V14, V10, V4, V21, V3. These are model/sample-dependent signals, not causal explanations.
- EDA: fraud amount median 9.82 vs legitimate median 22.00; top three standardized PCA-feature differences: V17 (-8.09), V14 (-7.52), V12 (-6.35).

## Limitations

This is a historical benchmark with anonymized PCA features, sparse fraud cases, no documented calendar reference for Time, and no source/license documentation included alongside the local CSV. Chronological holdout results may be unstable and do not establish present-day banking effectiveness, calibrated operational risk, or business value. Threshold selection optimizes validation F2 in the absence of costs or investigation capacity.

## Suggested future work (not implemented)

1. **Temporal representativeness:** one chronological split is vulnerable to drift and small numbers of later fraud cases. Improve with repeated rolling-origin evaluation and uncertainty intervals.
2. **Decision economics:** F2 is only a proxy without costs or analyst capacity. Obtain validated cost and capacity assumptions, then compare expected utility and precision-at-capacity.
3. **Dataset validity and reproducibility:** local provenance/license and the reason for repeated records are not documented. Verify source terms and definitions, and validate on more recent, independently sampled data before operational claims.
