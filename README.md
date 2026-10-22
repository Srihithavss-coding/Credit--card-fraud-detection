# Credit Card Fraud Detection

Reproducible analysis of the locally provided ULB / Worldline credit-card-fraud benchmark. The notebook is the primary deliverable; results are computed from the local dataset when it runs.

## Problem and dataset

The binary target is `Class` (1 fraud, 0 legitimate). The standard file contains `Time`, `V1`–`V28`, `Amount`, and `Class`. The commonly documented reference is [MLG-ULB Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). No source documentation or license accompanied the CSV, so provenance and redistribution permission cannot be verified. Do not commit or redistribute the dataset unless its terms permit it. `Time` is elapsed seconds, not a known wall-clock time. `V1`–`V28` are anonymized transformed values.

## Observed results

- Raw data: 284,807 rows, 31 columns; fraud prevalence 492/284,807 (0.1727%); cleaned fraud count 473 (0.1667%).
- Cleaning: 1,081 exact duplicate records removed from the analysis copy (19 fraud, 1062 legitimate); no missing or non-finite values, and no non-duplicate outliers removed. The original CSV remains untouched.
- EDA: fraud amount median 9.82, legitimate amount median 22.00; largest absolute standardized mean differences among PCA variables: V17 (-8.09), V14 (-7.52), V12 (-6.35). Descriptive only; PCA features have no assigned business meaning.
- Chronological splits: train 170,235 rows/342 fraud; validation 56,745/57; test 56,746/74.
- Best validation model by average precision: **HistGradientBoosting** (AP 0.77432).

### Validation model comparison

| Model | ROC-AUC | Average precision | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.9762 | 0.7743 | 0.2323 | 0.8070 | 0.3608 |
| Logistic Regression | 0.9706 | 0.7737 | 0.0257 | 0.8947 | 0.0499 |
| Random Forest | 0.9466 | 0.7662 | 0.9512 | 0.6842 | 0.7959 |
| Dummy (prior) | 0.5000 | 0.0010 | 0.0000 | 0.0000 | 0.0000 |

### Validation threshold trade-offs

| Threshold | Precision | Recall | F2 | TP | FP | FN | Alerts |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.003 | 0.982 | 0.015 | 56 | 18344 | 1 | 18400 |
| 0.05 | 0.018 | 0.877 | 0.085 | 50 | 2660 | 7 | 2710 |
| 0.1 | 0.039 | 0.860 | 0.163 | 49 | 1222 | 8 | 1271 |
| 0.5 | 0.232 | 0.807 | 0.540 | 46 | 152 | 11 | 198 |
| 0.905662 | 0.860 | 0.754 | 0.773 | 43 | 7 | 14 | 50 |

- Final test: ROC-AUC 0.9860; AP 0.7764; precision 0.8462; recall 0.7432; F1 0.7914; threshold 0.90566246; TN=56662, FP=10, FN=19, TP=55. Threshold selected on validation by F2, not on test and not by business costs.

Results are benchmark estimates with a small number of fraud examples in the chronological holdout; they do not prove live banking effectiveness. See [the final summary](reports/final_summary.md) for limitations and a critical review.

## Setup and execution

1. Use Python 3.10 or newer.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Place the downloaded `creditcard.csv` at the project root (or update `DATA_PATH` in the notebook).
4. Run from the project root: `jupyter nbconvert --to notebook --execute --inplace notebooks/credit_card_fraud_detection.ipynb`. For interactive use: `jupyter lab notebooks/credit_card_fraud_detection.ipynb`.

## Project structure

- `notebooks/credit_card_fraud_detection.ipynb` — end-to-end executed analysis.
- `src/data_validation.py`, `src/data_cleaning.py`, `src/eda.py`, `src/train.py`, `src/evaluate.py` — reusable workflow modules.
- `reports/` — quality audit, model comparison, test metrics, permutation importance, and summary.
- `plots/` — generated EDA and evaluation figures.
- `models/` — fitted pipeline and JSON metadata including schema and decision threshold.
- `data/raw/` — reserved for data only if safely copied; this project leaves the input CSV in place to avoid duplication.

## Cleaning and modeling decisions

Exact duplicates are removed while keeping the first occurrence before splitting; no target changes, imputation, or automatic outlier deletion are made. A chronological 60/20/20 split is used without stratification. Logistic Regression scales within its training pipeline; tree models use unscaled inputs. Class weights/sample weights address imbalance in selected models. Model choice and the F2 decision threshold use validation only; the test interval is reserved for final reporting.

## Dependencies and future work

See `requirements.txt`. Improve the evidence with rolling-origin validation, uncertainty intervals, verified current data with documented provenance, probability-calibration checks on separate data, and threshold optimization using actual review capacity and error costs. This benchmark alone is not sufficient for production decisions.
