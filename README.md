# Credit Card Fraud Detection

An end-to-end analysis of the ULB / Worldline credit-card-fraud benchmark. The notebook audits the data, explores class and feature patterns, compares classifiers, selects a decision threshold using validation data, and reports performance on a later chronological test interval.

## Dataset and method

The standard dataset has 284,807 rows and 31 columns: elapsed `Time`, anonymized features `V1`–`V28`, `Amount`, and binary target `Class` (1 = fraud). The commonly documented reference is [MLG-ULB Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). No source or license documentation accompanied the local CSV, so provenance and redistribution rights are unverified. The data is not included in this repository.

The analysis removes 1,081 exact duplicate rows, keeping the first occurrence (19 fraud and 1,062 legitimate). This is an explicit preprocessing assumption: results depend on treating identical rows as repeats. No missing values were imputed and no non-duplicate outliers were removed. A deterministic `log1p(Amount)` feature is added.

Rows are split chronologically by elapsed `Time`, without stratification:

| Partition | Rows | Fraud cases | Inclusive `Time` range (seconds) |
|---|---:|---:|---:|
| Train | 170,235 | 342 | 0–120,392 |
| Validation | 56,745 | 57 | 120,393–145,233 |
| Test | 56,746 | 74 | 145,234–172,792 |

Model configurations are fixed before fitting. Models are trained on the training interval and compared on validation average precision (AP), with ROC-AUC and threshold-0.5 scores also reported. AP is useful under the severe class imbalance (57 fraud cases in validation); the prior-only model's AP of 0.0010 is close to validation prevalence.

## Validation results

| Model | ROC-AUC | Average precision | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.9762 | 0.7743 | 0.2323 | 0.8070 | 0.3608 |
| Logistic Regression | 0.9706 | 0.7737 | 0.0257 | 0.8947 | 0.0499 |
| Random Forest | 0.9466 | 0.7662 | 0.9512 | 0.6842 | 0.7959 |
| Dummy (prior) | 0.5000 | 0.0010 | 0.0000 | 0.0000 | 0.0000 |

HistGradientBoosting was selected by validation AP. Its AP (0.77432) is only 0.000645 above Logistic Regression (0.77367); this small difference on one validation interval does not establish consistent superiority.

The threshold was selected by maximizing validation F2, which weights recall more heavily than precision. At the selected threshold, compared with the conventional 0.5 cutoff:

| Validation threshold | Precision | Recall | F2 | TP | FP | FN | Alerts |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.232 | 0.807 | 0.540 | 46 | 152 | 11 | 198 |
| 0.90566246 (selected) | 0.860 | 0.754 | 0.773 | 43 | 7 | 14 | 50 |

This reduces validation alerts while increasing missed fraud relative to 0.5. F2 is a proxy, not a business-optimal operating point: review capacity and error costs were not provided.

## Held-out test results

The final chronological test interval was not used for model or threshold selection. At the validation-selected threshold:

| ROC-AUC | Average precision | Precision | Recall | F1 | TN | FP | FN | TP | Alerts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9860 | 0.7764 | 0.8462 | 0.7432 | 0.7914 | 56,662 | 10 | 19 | 55 | 65 |

There are only 74 fraud cases in this test interval, and the evaluation uses one chronological split. Metric uncertainty and temporal stability have not been established. These benchmark results do not establish production banking effectiveness. `Time` is not a calendar timestamp, and the anonymized PCA features have limited business interpretability. See [the final summary](reports/final_summary.md) for additional limitations and suggested future work.

## Setup

1. Use Python 3.10 or newer. Install dependencies from the repository root:
   ```powershell
   python -m pip install -r requirements.txt
   ```
2. Obtain the dataset independently and place it at `<project-root>/creditcard.csv`. The notebook does not download it.
3. From the repository root, run the notebook with `jupyter lab notebooks/credit_card_fraud_detection.ipynb` or execute it with `jupyter nbconvert --to notebook --execute --inplace notebooks/credit_card_fraud_detection.ipynb`. It can also be launched from `notebooks/`; project-root discovery uses `README.md` and `src/`.
4. Run the dataset-independent tests from the repository root: `python -m pytest`.

The saved model metadata records scikit-learn 1.7.2, which is pinned in `requirements.txt` for compatibility with the serialized model.

## Repository contents

- `notebooks/credit_card_fraud_detection.ipynb` — end-to-end analysis and post-output interpretations.
- `src/` — schema validation, cleaning, exploratory summaries, model training, and evaluation.
- `tests/` — synthetic-data tests that do not require the benchmark CSV.
- `reports/`, `plots/`, `models/` — saved analysis outputs, fitted pipeline, and model metadata.
- `data/raw/` — empty placeholder; the notebook's default CSV location is the repository root.
