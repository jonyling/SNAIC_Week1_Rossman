# Rossmann Store Sales — Time Series Forecasting Capstone

SNAIC (ICT6001) Week 1 capstone (Group 9): forecast daily sales for 1,115 Rossmann drug stores in Germany using historical sales, promotions, and store metadata, and turn the forecast into a business-ready recommendation.

## Problem

Predict daily `Sales` for each store using the classic [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle dataset — a three-table join (daily sales panel + per-store static features) with strong calendar/promo seasonality and the usual retail data-quality issues (store closures, missing competitor data, string-typed holiday flags).

## Approach

1. **Data quality audit** — identified missing competition-open dates, typo years (e.g. `1900`), stores with `Open=1 & Sales=0`, and inconsistent `StateHoliday` typing before modeling.
2. **Feature engineering** — calendar features, holiday/competition/Promo2 flags, and **train-only** lag/rolling features (`Sales_Lag7/14/28`, `Customers_Lag7`, rolling means) built with strict leakage controls (no same-day `Sales` leakage; test-time lags built from train history only).
3. **Modeling** — Ridge, Random Forest, and LightGBM compared via 5-fold `TimeSeriesSplit`, all inside leakage-safe `Pipeline`/`ColumnTransformer` (imputation + scaling per fold). The CV-best model (LightGBM) was then tuned with `GridSearchCV` (27-point grid, capped at the capstone's 50-fit budget).
4. **Evaluation** — RMSPE (Root Mean Square Percentage Error) on a temporal holdout split, plus 4 ablation experiments to check which feature families actually mattered.
5. **Business translation** — analyzed over-/under-prediction rates on the holdout and derived a safety margin to bias the deployed forecast toward avoiding stockouts.

## Results

| Model | CV RMSPE (tuned) | Val RMSPE |
|---|---|---|
| **LightGBM (champion)** | 0.3580 ± 0.2734 | **15.56%** |

- Best params: `learning_rate=0.05`, `max_depth=8`, `num_leaves=127`.
- Top drivers: `Sales_Lag14`, rolling sales means, `Promo`, `Store_MeanLogSales`.
- On the validation holdout: 48.6% of predictions were over-forecasts, 51.4% under-forecasts. Applying a **0.838 safety multiplier** to the champion's forecast caps the under-prediction rate at ~10% — trading a bit of over-stocking risk for fewer stockouts.
- Both a raw ("Kaggle") forecast and a margin-adjusted ("business") forecast are produced for the 41,088-row test set, since the two use cases (leaderboard score vs. inventory/staffing planning) call for different risk tolerances.

Full write-up, error analysis, and slides: [`Group 9 - Rossmann Store Sales.pdf`](./Group%209%20-%20Rossmann%20Store%20Sales.pdf).

## Repo structure

```
.
├── eda_JL_CPU.ipynb   # Full pipeline: EDA -> feature engineering -> modeling -> evaluation -> submission
├── submissions/       # Model outputs (Kaggle-format and business/inventory-format predictions)
└── Group 9 - Rossmann Store Sales.pdf   # Capstone slides / report
```

`data/` (the raw `train.csv`, `test.csv`, `store.csv`) is not included in this repo — see **Data** below.

## Data

This project uses the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle competition dataset, which isn't redistributed here. To reproduce:

1. Download `train.csv`, `test.csv`, and `store.csv` from the [competition page](https://www.kaggle.com/c/rossmann-store-sales/data) (requires a free Kaggle account and accepting the competition rules).
2. Place them in a `data/` folder at the repo root.
3. Run `eda_JL_CPU.ipynb` top to bottom.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook eda_JL_CPU.ipynb
```

The notebook runs on CPU (`lightgbm` with `device='cpu'`, `n_jobs=-1`) — no GPU required.

## Notes

- Predictions clamp `Open=0` days to `Sales=0`.
- Per the capstone constraint, the model was not re-tuned against the Kaggle leaderboard score — the holdout validation RMSPE was treated as the single deployment metric.
