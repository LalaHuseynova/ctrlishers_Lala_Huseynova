# Article Popularity Prediction

A machine learning project that predicts whether an online article will become **popular** before it is published. It uses only information available at publish time: content length, links, images, keywords, topics, channel, and publish day.

The task is binary classification (`target = 1` for popular, `0` otherwise), and models are evaluated with **ROC-AUC**.

## Highlights

- **No time leakage**: validation always uses articles newer than the training data, just like in real use.
- **Feature engineering** grounded in EDA, such as link density, the dominant topic, and channel × weekend.
- **7 models compared**, from Logistic Regression to LightGBM and CatBoost.
- **A rank-based blend** of the strongest models.
- **Recency weighting**, tested with time-series cross-validation before it is used.

## Project structure

```
.
├── datasets/
│   ├── train.csv          # training data with target
│   └── test.csv           # data to predict
├── pca3_clean.ipynb       # full pipeline: EDA → models → submission
└── README.md
```


## How to run

1. Put `train.csv` and `test.csv` in the `datasets/` folder.
2. Open `pca3_clean.ipynb` and choose **Run All**.
3. The notebook writes two files:

| File | Description |
|---|---|
| `submission_blend.csv` | Blend of LightGBM, CatBoost and Logistic Regression (main submission) |
| `submission_lgb.csv` | LightGBM only (backup, in case the blend scores lower) |

## Pipeline

### 1. Exploratory data analysis
- Checks missing values and class balance.
- Compares feature means between popular and non-popular articles.
- Looks at popularity by channel and weekday.
- Finds heavily skewed features, which are later log-transformed for linear models.

### 2. Cleaning
- Clips the ratio features (`n_unique_tokens`, `n_non_stop_unique_tokens`) to 1, because some rows had impossible values.
- Drops `n_non_stop_words`, which is about 1 for every row and so carries no information.

### 3. Feature engineering

| Feature | Meaning |
|---|---|
| `lda_top` | Dominant LDA topic of the article |
| `lda_max` | How strongly the article belongs to that topic |
| `is_weekend` | Published on Saturday or Sunday |
| `chan_weekend` | Channel combined with the weekend flag |
| `hrefs_per_word` | Link density |
| `imgs_per_word` | Image density |
| `kw_avg_ratio` | Average keyword shares relative to the maximum |
| `month` | Publish month (the raw date is never used as a feature) |

### 4. Time-based validation
The data is sorted by `publish_date`. The oldest 80% of articles are used for training and the newest 20% for validation. A random split would let the model "see the future" and give scores that are too optimistic.

### 5. Models

| Model | Preprocessing |
|---|---|
| Logistic Regression | One-hot encoding + log transform + scaling |
| PCA + Logistic Regression | Same as above, then PCA keeping 95% of the variance |
| Random Forest | One-hot encoding |
| Extra Trees | One-hot encoding |
| HistGradientBoosting | One-hot encoding |
| LightGBM | Native categorical features, early stopping |
| CatBoost | Native categorical features, early stopping |

### 6. Rank blend
The final prediction is a weighted average of each model's **ranks**, not its raw probabilities:

```
LightGBM 0.5 + CatBoost 0.3 + Logistic Regression 0.2
```

Because ROC-AUC depends only on ordering, averaging ranks is safe even when the models' probabilities are on different scales.

### 7. Time-series cross-validation
A 4-fold `TimeSeriesSplit` checks whether giving newer articles more weight (an exponential decay with a one-year scale) improves LightGBM. The weights are kept only if the CV score improves.

### 8. Final training
All three blend models are retrained on the full training set. Boosting models use about 1.2× their best iteration count, since they now see 25% more data.

## Results

Fill this in after running the notebook (copy the numbers from `results_df`):

| Model | Validation ROC-AUC |
|---|---|
1	LightGBM |	0.755311
2	CatBoost	|	0.755160
3	Random Forest	|	0.750544
4	HistGradientBoosting	|	0.749052
5	Extra Trees	|	0.739851
6	Logistic Regression	|	0.735491
7	PCA + Logistic Regression	|	0.724215

## Limitations

- LightGBM and CatBoost validation scores are slightly optimistic, because early stopping used the validation set.
- The CV scores are also slightly optimistic, because the LightGBM iteration count was tuned on data that overlaps the later folds.
- Recency weights are tested and used only for LightGBM.
- The blend weights were chosen by hand, not optimized.

