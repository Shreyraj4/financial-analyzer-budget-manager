# Anomaly detection

Code: `backend/app/ml/anomaly/` (detectors, evaluate, explain, train),
features in `backend/app/features/transaction_features.py`,
serving in `backend/app/services/anomalies.py`.
Results: `docs/metrics/anomaly_detection.json`.

## The problem

Flag unusual spending transactions. It is **unsupervised** in training (the
detector never sees anomaly labels while fitting), because on real data you
usually don't have labels. Labels exist here only to choose thresholds on
validation data and to grade results.

## Anomaly types injected into the synthetic data

| Type | Meaning |
|---|---|
| amount_spike | amount 5-15x larger than usual |
| duplicate_charge | same merchant and amount within a day |
| burst | 4-7 charges at one merchant on one day |
| unusual_high_value | Rs 25k-90k in a category the user rarely uses |

## Features (13, per transaction, computed per user)

log amount; robust z-score vs the user's history in that category; amount as a
share of typical monthly spend; merchant and category frequency; same-day merchant
count; merchant count in 3 days; duplicate-within-1-day flag; days since last
purchase in the category; day of week/month; weekend flag.

Robust z-score: `(x - median) / (1.4826 * MAD)`. Median and MAD (median absolute
deviation) aren't distorted by the outliers we're hunting, unlike mean/std.

## Detectors compared

1. **Robust z-score** (statistical baseline): amount only.
2. **Hand-written rules**: z-score, duplicate, burst, rare-large-purchase rules.
3. **Isolation Forest** (main ML): builds random trees that split features at
   random. Anomalies are few and different, so they get isolated in *fewer*
   splits; short average path = high anomaly score.
4. **Isolation Forest per user**.
5. **Local Outlier Factor**: compares a point's local density with its neighbours'.
6. **Isolation Forest + z-score union**.

## Protocol

Chronological: fit on months 1-12, choose thresholds on months 13-18, report on
months 19-24. Debits only. Anomaly rate in test: 6.2%.

## Results (test)

| Detector | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Robust z-score | 0.269 | 0.331 | 0.183 | 0.235 |
| Hand-written rules | 0.707 | 0.695 | 0.936 | 0.798 |
| **Isolation Forest** | **0.870** | 0.826 | 0.845 | **0.835** |
| IF per user | 0.835 | 0.760 | 0.854 | 0.804 |
| LOF | 0.393 | 0.599 | 0.361 | 0.450 |
| IF + z-score | 0.646 | 0.629 | 0.881 | 0.734 |

Recall by type (z-score vs Isolation Forest): amount spikes 1.00 vs 0.53;
duplicates 0.00 vs 0.71; bursts 0.02 vs 0.91; unusual high value 0.80 vs 0.92.

## What to conclude (and say in an interview)

- Isolation Forest is best overall on PR-AUC and F1.
- Detectors have complementary strengths. I tried the union; validation rejected
  it because precision fell from 0.83 to 0.63.
- Hand-written rules nearly match the forest, but they encode knowledge of how the
  synthetic anomalies were made (close to an oracle); the forest found the
  patterns without being told. Real data would be messier.
- Per-user models did not help: only ~600 training rows per user.

## Explanations without the LLM

`explain.py` produces structured reasons from feature values, e.g.
"Rs 25,625 is 35.0x your usual Health spend (typical Rs 733)" or
"7 charges at RAPIDO within a few days". If no rule explains a flag it says so
("unusual combination"). The LLM later narrates these; it never invents them.

## Known limitations (say them first)

- Synthetic labels made by my generator: optimistic.
- Duplicate detection looks one day ahead: in real time you can only flag the
  pair after the second charge.
- False positives exist: a legitimate rent payment was flagged at 3x usual
  Housing spend in a spot check (27 of 29 flags were true anomalies for that user).

## Concepts to be able to explain

- Supervised vs unsupervised; why anomalies are hard to label.
- Isolation Forest intuition (fewer splits = more anomalous), `contamination`.
- Robust statistics: median/MAD vs mean/std.
- PR-AUC vs ROC-AUC on imbalanced data; precision/recall trade-off; thresholds.
- Why per-type recall is more informative than a single number.

## Interview questions

**Q: How does Isolation Forest work?**
A: It builds random trees, each split picks a random feature and random value.
Unusual points are separated in few splits, so their average tree depth is short;
the score is derived from that depth.

**Q: Why not accuracy?**
A: Only ~6% are anomalies, so "always normal" gets 94%. I use PR-AUC and P/R/F1.

**Q: How did you pick the threshold?**
A: To maximize F1 on a validation period, then fixed it for the test period.

**Q: If it is unsupervised, why do you use labels?**
A: Only for choosing the threshold and evaluating. The model is fit without them.

**Q: Why not just use your hand-written rules?**
A: They only catch anomaly kinds someone anticipated. The forest generalizes to
unusual combinations, and here reaches similar quality without being told the
patterns. In production I'd combine both.

**Q: How would you deploy this to a real user?**
A: Compute features on their history, score, flag above the threshold, show the
reasons, and let the user mark false positives to improve the system.

## Reading

- Original paper: Liu, Ting, Zhou (2008), "Isolation Forest", IEEE ICDM
- scikit-learn: Novelty and outlier detection: https://scikit-learn.org/stable/modules/outlier_detection.html
- scikit-learn: Precision-recall curves: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
- StatQuest: "Isolation Forest" (search the channel), "Precision Recall curves"
- Chandola, Banerjee, Kumar (2009), "Anomaly Detection: A Survey" (skim intro)
