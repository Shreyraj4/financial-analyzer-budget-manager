# Spending forecasting

Code: `backend/app/ml/forecasting/` (model, evaluate, predict, train),
features in `backend/app/features/monthly_features.py`,
serving in `backend/app/services/forecasting.py`.
Results: `docs/metrics/forecasting.json`.

## The problem

For each user and category, predict **next month's total spend**. This is
**time-series regression**: 121 series (12 users x their categories), 24 monthly
points each, many of them noisy or intermittent (Travel, Education).

## Terms

- **Trend**: long-run direction (here: ~6%/year inflation).
- **Seasonality**: repeating yearly pattern (festive shopping, summer electricity).
- **Lag feature**: last month's / two months ago value used as input.
- **Rolling mean**: average of the last few months.
- **Intermittent series**: many zero months and occasional big spikes; very hard.

## Features (all use only earlier months)

`lag_1, lag_2, lag_3`, `roll_mean_3`, `roll_std_3`, month-of-year as
`sin/cos` (so December is close to January), and `hist_mean`, an expanding mean
of everything before the month.

## The key modelling trick: one global model with scaling

Instead of a separate tiny model per series (too little data), train **one**
model across all users and categories. To make rupee amounts comparable, divide
every value by the series' own `hist_mean`. The model learns shapes ("this month
is ~1.3x the recent average"), not rupees; prediction = ratio x `hist_mean`.
Ratios are capped at 10x so intermittent spikes don't dominate.

## Models

- **Ridge regression**: linear model with an L2 penalty that shrinks weights to
  avoid overfitting. Category-specific seasonality is given via
  category one-hot x (sin, cos) interaction features.
- **Gradient boosting (HistGradientBoosting)**: many small trees, each fixing the
  previous ones' errors; handles interactions automatically.

## Baselines

Naive (repeat last month), 3-month moving average, seasonal naive (same month
last year), and the project's original per-series linear trend.

## Evaluation: rolling origin

12 forecast months (Jan-Dec 2025). For each, train on earlier months only and
predict. First 6 origins = validation (pick Ridge alpha / GBM depth), last 6 =
test (report).

## Results on the test months

| Method | MAE (Rs) | RMSE | WAPE | MASE |
|---|---|---|---|---|
| Naive | 8,643 | 21,624 | 0.820 | 1.73 |
| Moving avg 3 | 8,045 | 18,242 | 0.763 | 1.57 |
| Seasonal naive | 8,950 | 23,584 | 0.849 | 1.44 |
| Original linear trend | 7,756 | 16,937 | 0.736 | 1.50 |
| Ridge | 7,291 | 16,416 | 0.691 | 1.26 |
| Gradient boosting (chosen on validation) | 7,255 | 16,726 | 0.688 | 1.29 |

Both ML models beat every baseline (~16% lower MAE than naive, ~6% lower than the
old linear trend). The gain is modest because the data is noisy: intermittent
categories put a ceiling on accuracy.

## Model selection honesty

Validation chose gradient boosting; Ridge was marginally better on test. I kept
the validation choice: switching after seeing test results is cheating. They are
effectively tied, and Ridge is easier to explain.

## Bugs found while building (great interview stories)

1. First ML run had MAE in the millions. Cause: series with zero history had a
   scale of ~0, so ratios exploded. Fix: drop no-history rows when training,
   predict 0 for them, cap ratios.
2. A test showed I evaluated 9 forecast months instead of 12: my "first origin"
   counted from the first *feature* month, not the series start.

## Concepts to be able to explain

- Why time-series must not be shuffled; rolling-origin evaluation.
- Lag features and leakage; why `shift` matters.
- Ridge regularization; bias-variance trade-off.
- Global model vs one model per series; normalization by history.
- MAE vs RMSE vs WAPE vs MASE.
- Why baselines matter (naive is hard to beat on noisy data).

## Interview questions

**Q: Why a global model?**
A: Each series has only 24 points. Pooling users and categories gives thousands of
training rows, and scaling by each series' history makes them comparable.

**Q: How did you avoid leakage?**
A: All features use `shift`, so month t sees months < t; the scale is an expanding
mean of the past; evaluation is walk-forward; settings were chosen on validation
months, results reported on later test months.

**Q: Why is MASE above 1 for everything?**
A: MASE divides by a naive forecast's error measured on each series' first year;
the test year is harder, so values above 1 are normal. Only compare methods
against each other.

**Q: The improvement over the baseline looks small.**
A: It's real but modest; the series are noisy and intermittent. Reporting an
honest small gain is better than a suspicious large one. Next steps: prediction
intervals, exogenous features, more data.

**Q: What does the forecast do for a brand-new category?**
A: With under 3 months of history it falls back to an average or last value, and
the API's `method` field says which was used.

## Reading

- Forecasting: Principles and Practice (free, best resource): https://otexts.com/fpp3/
  - Benchmark methods (naive, seasonal naive): https://otexts.com/fpp3/simple-methods.html
  - Evaluating accuracy (incl. MASE): https://otexts.com/fpp3/accuracy.html
  - Time-series cross-validation: https://otexts.com/fpp3/tscv.html
- Hyndman & Koehler (2006), "Another look at measures of forecast accuracy" (MASE origin)
- scikit-learn: Ridge regression: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
- scikit-learn: Histogram gradient boosting: https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
- StatQuest: "Gradient Boost", "Ridge regression", "Time series basics"
