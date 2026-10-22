# Credit Card Fraud Detection

Reproducible analysis of the locally provided ULB / Worldline credit-card-fraud benchmark. The notebook is the primary deliverable. The dataset CSV is not committed; obtain it independently before running the project.

## Problem and dataset

The binary target is `Class` (1 fraud, 0 legitimate). The standard file contains `Time`, `V1`–`V28`, `Amount`, and `Class`. The commonly documented reference is [MLG-ULB Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). No source documentation or license accompanied the local CSV, so its provenance and redistribution rights are unverified. Obtain the data independently and do not redistribute it unless its terms permit it. `Time` is elapsed seconds relative to the dataset, not a calendar date or known time of day. `V1`–`V28` are anonymized transformed features with limited direct business interpretability.

## Observed results

- Raw data: 284,807 rows, 31 columns; fraud prevalence 492/284,807 (0.1727%); cleaned data has 473 fraud records (0.1667%).
- Cleaning: 1,081 exact duplicate rows were removed from the analysis copy (19 fraud, 1,062 legitimate); no missing or non-finite values, and no non-duplicate outliers were removed. This is an analysis assumption: conclusions depend on the deduplication policy. The original CSV remains unchanged.
- EDA: fraud amount median 9.82, legitimate median 22.00. The distributions overlap, so the difference in medians does not justify a simple amount cutoff. V17 (-8.09), V14 (-7.52), and V12 (-6.35) had the largest absolute standardized mean differences among PCA features; these descriptive effects do not confer business meaning on anonymized variables. The strongest observed pairwise PCA-feature correlation is small (V8–V21: 0.019); low pairwise correlation does not rule out nonlinear or interaction-based predictive value.
- Chronological splits (inclusive observed `Time` ranges in elapsed seconds): train 170,235 rows/342 fraud, 0–120,392; validation 56,745/57, 120,393–145,233; test 56,746/74, 145,234–172,792. These are dataset-relative elapsed seconds, not calendar dates or times of day.
- **HistGradientBoosting** had the highest observed validation average precision (0.77432), narrowly ahead of Logistic Regression (0.7737). This small difference on one validation interval does not establish consistent superiority.

### Validation model comparison

| Model | ROC-AUC | Average precision | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.9762 | 0.7743 | 0.2323 | 0.8070 | 0.3608 |
| Logistic Regression | 0.9706 | 0.7737 | 0.0257 | 0.8947 | 0.0499 |
| Random Forest | 0.9466 | 0.7662 | 0.9512 | 0.6842 | 0.7959 |
| Dummy (prior) | 0.5000 | 0.0010 | 0.0000 | 0.0000 | 0.0000 |

### Validation threshold trade-offs

These counts and metrics are measured on the 56,745-row validation interval only. The selected threshold maximizes validation F2; its validation result is 43 true positives, 7 false positives, 14 false negatives, and 50 alerts. This is a validation-based selection, not a guarantee of the same alert volume or performance on later data.

| Threshold | Precision | Recall | F2 | TP | FP | FN | Alerts |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.003 | 0.982 | 0.015 | 56 | 18344 | 1 | 18400 |
| 0.05 | 0.018 | 0.877 | 0.085 | 50 | 2660 | 7 | 2710 |
| 0.1 | 0.039 | 0.860 | 0.163 | 49 | 1222 | 8 | 1271 |
| 0.5 | 0.232 | 0.807 | 0.540 | 46 | 152 | 11 | 198 |
| 0.905662 | 0.860 | 0.754 | 0.773 | 43 | 7 | 14 | 50 |

- Final test on the untouched 56,746-row interval (74 fraud): ROC-AUC 0.9860; AP 0.7764; precision 0.8462; recall 0.7432; F1 0.7914; threshold 0.90566246; TN=56,662, FP=10, FN=19, TP=55 (65 alerts). The threshold was selected using validation F2, not test labels. Alert counts differ across the validation and test intervals; this threshold is not optimized for real-world review capacity or financial costs.

**Limitations:** The final test interval contains only 74 fraud cases, and results come from one chronological split. Statistical uncertainty and temporal stability have not been established. F2 does not incorporate actual operational costs or review capacity. Dataset provenance and redistribution rights are unverified, and benchmark results do not establish production banking effectiveness. See [the final summary](reports/final_summary.md) for a critical review and possible next experiments; rolling-origin evaluation, uncertainty intervals, calibration, and cost-based threshold optimization have not been performed.

## Setup and execution

1. Use Python 3.10 or newer.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Obtain the dataset independently from the source above and place it at `<project-root>/creditcard.csv`. The CSV is not committed, so cloning the repository alone is not sufficient to run the notebook.
4. If the file is elsewhere, edit `DATA_PATH` in the notebook's **Imports and configuration** cell to point to the local CSV, for example `Path(r"C:\path\to\creditcard.csv")`.
5. Run from the project root: `jupyter nbconvert --to notebook --execute --inplace notebooks/credit_card_fraud_detection.ipynb`. For interactive use: `jupyter lab notebooks/credit_card_fraud_detection.ipynb`.

## Project structure

- `notebooks/credit_card_fraud_detection.ipynb` — end-to-end executed analysis.
- `src/data_validation.py`, `src/data_cleaning.py`, `src/eda.py`, `src/train.py`, `src/evaluate.py` — reusable workflow modules.
- `reports/` — quality audit, model comparison, test metrics, permutation importance, and summary.
- `plots/` — generated EDA and evaluation figures.
- `models/` — fitted pipeline and JSON metadata including schema and decision threshold.
- `data/raw/` — reserved for data only if safely copied; this project leaves the input CSV in place to avoid duplication.

## Cleaning and modeling decisions

Exact duplicates are removed while keeping the first occurrence before splitting; this assumption reduces the cleaned fraud count by 19, and reported conclusions are conditional on that policy. No target changes, imputation, or automatic outlier deletion are made. A chronological 60/20/20 split is used without stratification. Logistic Regression scales within its training pipeline; tree models use unscaled inputs. Class weights/sample weights address imbalance in selected models. Model choice and the F2 decision threshold use validation only; the final test interval is reserved for reporting.

## Dependencies and future work

See `requirements.txt`. Potential follow-up work includes rolling-origin validation, uncertainty intervals, verified current data with documented provenance, probability-calibration checks on separate data, and threshold optimization using actual review capacity and error costs. None of these follow-up analyses is implemented in the current results. This benchmark alone is not sufficient for production decisions.
