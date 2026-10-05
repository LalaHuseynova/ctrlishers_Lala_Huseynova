# Article Popularity Prediction

Predicts whether an online article will be popular (`target = 1`) or not (`target = 0`), using features such as title length, links, images, keywords, LDA topics, channel, and publish day.

Models are compared using **ROC-AUC** on a time-based validation set.

## Project structure

```
.
├── datasets/
│   ├── train.csv
│   └── test.csv
├── pca3_clean.ipynb
└── README.md
```

## Setup

```bash
pip install numpy pandas matplotlib scipy scikit-learn lightgbm catboost
```

## How to run

1. Put `train.csv` and `test.csv` in the `datasets/` folder.
2. Open `pca3_clean.ipynb` and run all cells from top to bottom.
3. Two submission files are created:
   - `submission_blend.csv`: blend of LightGBM, CatBoost and Logistic Regression
   - `submission_lgb.csv`: LightGBM only (backup)

## Approach

1. **EDA**: missing values, class balance, feature distributions, mean differences between popular and non-popular articles, skewness.
2. **Cleaning**: clips broken ratio values to 1 and drops `n_non_stop_words`, which carries no information.
3. **Feature engineering**: dominant LDA topic, weekend flag, channel × weekend, links and images per word, keyword ratio, publish month.
4. **Time-based split**: the oldest 80% of articles are used for training and the newest 20% for validation, so the model never sees the future.
5. **Models**:
   - Logistic Regression (with and without PCA)
   - Random Forest, Extra Trees, HistGradientBoosting
   - LightGBM and CatBoost with early stopping
6. **Rank blend**: a weighted average of model ranks (LightGBM 0.5, CatBoost 0.3, Logistic Regression 0.2).
7. **Time-series CV**: checks whether giving newer articles more weight helps LightGBM.
8. **Final training**: models are retrained on all training data and predictions are made for the test set.

## Notes

- Validation scores for LightGBM and CatBoost are slightly optimistic, because early stopping used the validation set.
- Recency weights are applied only to LightGBM, since that is the only model they were tested on.
