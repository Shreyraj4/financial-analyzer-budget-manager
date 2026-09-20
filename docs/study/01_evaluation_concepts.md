# Evaluation concepts (the foundation for every interview answer)

If you can explain this file, you can defend every result in the project.

## 1. Why we hold data out

A model can memorize its training data. To measure real performance you test
on data it never saw. Everything below is a variation on "how do we hold data
out *honestly*".

## 2. Overfitting

Great on training data, poor on new data. Symptom: training score >> test score.
Defences used here: held-out data, simple models, regularization (Ridge),
capping extreme values.

## 3. Data leakage (the classic silent killer)

Information from the future or from the answer sneaks into the features, making
scores fake. Where we guarded against it:

| Risk | Guard in our code |
|---|---|
| Lag/rolling features looking at the target month | built with `shift`, tested in `test_features.py` |
| Scale computed using the test month | `hist_mean` is an expanding mean of *earlier* months only |
| Random split on time series | chronological splits everywhere |
| Thresholds tuned on test data | chosen on a separate validation period |
| Memorizing merchants | "unseen merchant" evaluation (GroupKFold) |

Known remaining leakage (be honest about it): anomaly features such as
"duplicate within 1 day" look one day ahead, and per-user statistics use the
user's whole history.

## 4. Train / validation / test

- **Train:** fit the model.
- **Validation:** choose settings (thresholds, hyperparameters, which model).
- **Test:** report final numbers; *never* used to make a choice.
We followed this in forecasting (months 1-12 / 13-18 / 19-24 for anomalies;
first six forecast origins vs last six for forecasting).

## 5. Time-series evaluation: rolling origin

Random splits are invalid for time series (you'd train on the future).
Rolling-origin: for each month M, train on months < M, predict M, move forward.
Also called walk-forward validation. scikit-learn: `TimeSeriesSplit`.

## 6. Cross-validation variants used

- **Stratified k-fold** (categorization, known merchants): keeps class ratios.
- **GroupKFold** (categorization, unseen merchants): a merchant never appears in
  both train and test. Measures generalization, not memorization.

## 7. Baselines

A simple method your model must beat. If it doesn't, the model isn't worth its
complexity. Ours: rules / lookup table (categorization), naive / moving average /
seasonal naive / the old linear trend (forecasting), z-score / hand-written rules
(anomaly).

## 8. Metrics cheat sheet

**Classification (categorization, anomaly)**

| Metric | Meaning | When it misleads |
|---|---|---|
| Accuracy | share correct | Rare classes: "always normal" gets 94% |
| Precision | of what I flagged, share correct | ignores misses |
| Recall | of the real ones, share I found | ignores false alarms |
| F1 | balance of precision and recall | needs a threshold |
| Macro-F1 | F1 averaged equally over classes | small classes count as much as big |
| PR-AUC | quality across all thresholds; best for rare positives | harder to interpret |
| ROC-AUC | ranking quality | optimistic when positives are rare |

Worked example: flag 100, 40 correct, 60 real anomalies in total:
precision 0.40, recall 0.67, F1 = 2*0.4*0.67/(0.4+0.67) = 0.50.

**Regression / forecasting**

| Metric | Meaning |
|---|---|
| MAE | average absolute error, in rupees |
| RMSE | like MAE, big errors weigh more |
| WAPE | total absolute error / total actual (scale-free) |
| MASE | error relative to a naive forecast's error; compare methods against each other |
| sMAPE / MAPE | percent errors; unstable with zeros, so avoided |

## 9. Threshold vs ranking

A detector outputs a *score*. A *threshold* turns it into yes/no. PR-AUC judges
the ranking across all thresholds; precision/recall/F1 judge one chosen threshold.
Choose the threshold on validation data.

## 10. Class imbalance

Anomalies are ~6% of rows. Use PR-AUC / recall / F1, not accuracy.

## Interview questions

**Q: How do you know your model isn't overfitting?**
A: I test on data it never saw, split chronologically for time-based tasks and by
merchant for categorization, and I compare against baselines.

**Q: Why not use accuracy for anomaly detection?**
A: With 6% anomalies, predicting "normal" always gives 94% accuracy and is
useless. I use PR-AUC, precision and recall.

**Q: What is data leakage? Give an example from your project.**
A: Using information unavailable at prediction time. Example: a rolling mean that
includes the month being predicted. I built lag features with `shift` and wrote a
test that the target month is excluded.

**Q: Why did you choose the threshold on validation, not test?**
A: Tuning on test leaks information and inflates results; test must stay untouched.

## Reading

- scikit-learn: Cross-validation, incl. TimeSeriesSplit: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn: Common pitfalls (data leakage): https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn: Metrics and scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
- Forecasting: Principles and Practice, ch. 5.8 time-series cross-validation and ch. 5.8 accuracy: https://otexts.com/fpp3/accuracy.html and https://otexts.com/fpp3/tscv.html
- StatQuest (YouTube): "Cross Validation", "Precision and Recall", "ROC and AUC", "Bias and Variance"
