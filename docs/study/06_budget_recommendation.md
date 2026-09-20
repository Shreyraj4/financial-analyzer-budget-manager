# Budget recommendation

Code: `backend/app/ml/recommendation/` (budget, evaluate, train),
serving in `backend/app/services/recommendations.py`,
endpoint `GET /recommendations/budgets`.
Results: `docs/metrics/budget_recommendation.json`.

## The problem

Suggest, for each category, a budget for next month that a user can realistically
keep. A budget equal to the forecast is breached about half the time (the forecast
is the *expected* spend). We want a budget with a stated reliability: "you should
stay within this in about 80% of months".

## Idea: forecast + calibrated margin (split conformal prediction)

1. Forecast next month's spend (the ML forecaster from `03_forecasting.md`).
2. Look at how wrong that forecaster has been *in the past, out of sample*, scaled
   by the category's typical spend: `r = (actual - forecast) / typical_spend`.
3. Take the 80% quantile of those scaled errors (`q`), with the finite-sample
   correction: the `ceil(0.8 * (n + 1))`-th smallest.
4. `budget = forecast + q * typical_spend`.

Why it works: if future errors look like past errors, about 80% of future months
land at or below the budget, regardless of which forecaster you use. Margins are
learned per category when there are >= 30 past residuals (travel needs a much
larger margin than utilities), else pooled across categories.

Terms: **quantile** (the value below which a given share of data falls),
**calibration** (making stated probabilities match reality), **conformal
prediction** (a distribution-free way to turn any point predictor into
intervals/bounds with a coverage guarantee).

## Anomaly link

One-off events (a Rs 90,000 spike, a burst of duplicate charges) shouldn't set
next month's budget. The recommender can drop transactions flagged by the anomaly
detector from the history before forecasting. This is tested as an ablation
("clean history") and used only if validation says it helps. The user's *actual*
spending is still reported unchanged.

## Backtest design

12 rolling forecast months (second year). Margins for a month only use residuals
from strictly earlier months (tested by tampering with the future and checking the
budget doesn't change).

| Months | Role |
|---|---|
| Jan-Mar | calibration only |
| Apr-Jun | validation: choose which variant ships |
| Jul-Dec | test: reported, never used for choices |

Every method gets the *same* conformal procedure at 80% target coverage, so the
comparison isolates forecast quality: at the same breach rate, who wastes least?

## Metrics

| Metric | Meaning |
|---|---|
| Breach rate | share of category-months where actual > budget; target 0.20 |
| Slack | unspent budget / total actual spend: waste, lower is better |
| Shortfall | overspend beyond budget / total actual spend |
| Pinball loss (tau=0.8) | proper scoring rule for an upper-quantile forecast; punishes under-budgeting (weight 0.8) and over-budgeting (0.2); lower is better |

## Results (test months, target breach 0.20)

| Method | Breach | Slack | Shortfall | Pinball |
|---|---|---|---|---|
| Naive (last month) | 0.220 | 0.798 | 0.247 | 3839 |
| Moving average 3 | 0.224 | 0.687 | 0.248 | 3602 |
| Seasonal naive | 0.188 | 0.795 | 0.239 | 3759 |
| Original linear trend | 0.203 | 0.623 | 0.239 | 3393 |
| Ridge | 0.256 | 0.434 | 0.266 | 3214 |
| GBM | 0.238 | 0.422 | 0.273 | 3250 |
| Ridge, clean history | 0.208 | 0.462 | 0.247 | **3119** |
| **GBM, clean history (shipped)** | 0.228 | 0.450 | 0.251 | 3125 |
| Ridge, **no margin** | 0.351 | 0.360 | 0.307 | 3411 |
| GBM clean, **no margin** | 0.499 | 0.104 | 0.441 | 4012 |

Reading it:
- **Calibration works.** Every calibrated method breaches near 20%. Without a
  margin, breach rates are 35-50%.
- **Better forecasts give tighter budgets at the same reliability.** Ridge-clean
  has ~19% lower pinball loss than naive and ~8% lower than the old linear trend;
  slack falls from ~0.80 to ~0.45.
- **Cleaning history helps a bit** (pinball -3% for Ridge, -4% for GBM) and brings
  breach rates closer to 20% (raw ML models overshoot to 24-26%).
- **Calibration sweep** (target -> achieved coverage on test): 60% -> 55.8%,
  70% -> 68.1%, 80% -> 79.2%, 90% -> 91.4%. Close, which is the point of conformal.
- Validation picked GBM-clean; Ridge-clean is marginally better on test (3119 vs
  3125). Effectively tied; I kept the validation choice on purpose.

## Advice rules (deterministic; the LLM only narrates them)

Compared with the user's existing monthly budget for a category:
no budget -> `set_budget`; budget below the forecast -> `raise` (likely exceeded);
budget above 1.25 x recommendation -> `tighten` (potential saving); otherwise
`keep`. Each recommendation also carries last month's spend, the 3-month average,
the forecast change vs that average, and its `basis` (category margin, pooled
margin, or fallback for short histories).

## Honest limitations

1. **Slack is large (~45% of spend).** Category spend is noisy and intermittent, so
   an 80%-reliable budget must be generous. That's a real trade-off, not a bug;
   a lower target (e.g. 70%) gives tighter budgets.
2. **Coverage is "on average", not per user or category.** Conformal gives marginal
   coverage; individual categories can be over- or under-covered.
3. **It assumes tomorrow's errors resemble yesterday's.** A sudden lifestyle change
   breaks that.
4. **Cleaning relies on the anomaly detector** (precision ~0.83); wrongly removed
   normal transactions understate history. Synthetic labels make its accuracy
   optimistic.
5. **Sum of category budgets overstates a total 80% bound**, since not all
   categories overshoot at once.
6. Synthetic data, so absolute numbers are indicative only.

## Concepts to be able to explain

- Point forecast vs quantile/upper-bound forecast.
- Calibration and coverage; why an uncalibrated budget is breached ~50% of the time.
- Split conformal prediction and the `ceil((n+1) * coverage)` rank.
- Pinball (quantile) loss vs MAE.
- Exchangeability assumption and time-ordering (only earlier residuals are used).
- Trade-off between reliability and tightness (slack vs breach rate).
- Ablation: testing a component (anomaly cleaning) by turning it on and off.

## Interview questions

**Q: How do you turn a forecast into a budget?**
A: Add a margin equal to a high quantile of the forecaster's past scaled errors, so
the budget covers about 80% of months. This is split conformal prediction.

**Q: How do you know 80% really means 80%?**
A: Calibration sweep on unseen test months: target 60/70/80/90% achieved
55.8/68.1/79.2/91.4%. I also tested that margins never use the month being
predicted.

**Q: Why is pinball loss the right metric?**
A: It is the proper scoring rule for quantile forecasts: it penalizes budgets that
are too tight (breaches) and too loose (waste) with weights matching the target
quantile. MAE would treat both errors equally.

**Q: Did the anomaly detector actually help?**
A: Yes, modestly: 3-4% lower pinball loss and breach rates closer to target; I
tested it as an ablation and used validation to decide.

**Q: Why is the slack so high?**
A: Noisy, intermittent categories force wide margins for 80% reliability. Users
can pick a lower coverage for tighter budgets at the cost of more breaches.

**Q: What if a user has little history?**
A: With under 3 months the forecast falls back to a simple average and a pooled
margin, and the response's `basis` field says so.

## Reading

- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" (free on arXiv; read the intro and split-conformal section)
- Forecasting: Principles and Practice, prediction intervals and quantile scores: https://otexts.com/fpp3/prediction-intervals.html and https://otexts.com/fpp3/distaccuracy.html
- scikit-learn: Quantile / pinball loss (`mean_pinball_loss`): https://scikit-learn.org/stable/modules/model_evaluation.html#pinball-loss
- StatQuest (YouTube): "Quantiles and Percentiles", "Confidence Intervals"
