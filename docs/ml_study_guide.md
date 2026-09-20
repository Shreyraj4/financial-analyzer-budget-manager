# ML Study Guide for This Project (Beginner Friendly)

You don't need to learn all of machine learning. You need the ~15 ideas below,
in this order. Each one is tied to a file in this repo so you can read real code
right after reading the concept.

> **Interview prep lives in `docs/study/`**: start with `00_project_pitch_and_architecture.md`,
> then `01_evaluation_concepts.md`, then one file per model.

## 0. The one-paragraph big picture

A **model** is a function that learns a pattern from examples instead of being
hand-coded. **Training** = showing it examples and letting it adjust itself.
**Evaluation** = testing it on examples it has never seen, to check it really
learned the pattern and didn't just memorize. Every ML project is:
data -> features -> train -> evaluate -> use.

## 1. Concepts, in study order

| # | Concept | Plain-English meaning | Where it appears here |
|---|---|---|---|
| 1 | **Features and labels** | Features = the input numbers describing an example. Label = the right answer. | `features/`, dataset columns `category`, `is_anomaly` |
| 2 | **Supervised vs unsupervised** | Supervised = learn from labeled answers. Unsupervised = find structure with no answers. | Categorization/forecast = supervised. Isolation Forest, KMeans = unsupervised |
| 3 | **Train / test split** | Keep some data hidden from training to grade the model honestly. | `ml/dataset.py: time_split` |
| 4 | **Overfitting** | Model memorizes training data and fails on new data. The main enemy. | Why we hold out merchants and months |
| 5 | **Data leakage** | Accidentally letting the model see the future or the answer. Makes scores fake. | Lag features use `shift`, see `monthly_features.py` |
| 6 | **Time-series basics** | Order matters. Trend, seasonality, lags, rolling averages. Split by time, never randomly. | forecasting, `build_forecast_features` |
| 7 | **Baselines** | A dumb simple method your model must beat, or it isn't worth it. | "predict last month", rule-based categorizer, z-score |
| 8 | **Linear / Ridge regression** | Fit a weighted sum of features to predict a number. Ridge adds a penalty against overfitting. | Forecasting model |
| 9 | **Logistic regression** | Same idea, but outputs a probability for each class. | Categorization model |
| 10 | **TF-IDF and n-grams** | Turn text into numbers: how often pieces of text (letters/words) appear, down-weighting common ones. | `features/text_features.py` |
| 11 | **Isolation Forest** | Anomalies are easy to "isolate" with random splits, so they get a high score. | Anomaly detection |
| 12 | **KMeans clustering** | Group similar rows around k centers. Choose k with the silhouette score. | Spending profiles |
| 13 | **Classification metrics** | Accuracy can lie. Learn precision, recall, F1, confusion matrix. | Categorization, anomaly |
| 14 | **Regression metrics** | MAE, RMSE, sMAPE, MASE (error vs the naive baseline). | Forecasting |
| 15 | **Class imbalance** | Anomalies are 5% of rows, so "always say normal" gets 95% accuracy and is useless. | Why we use PR-AUC/recall, not accuracy |

## 2. The metrics, explained with anomalies as the example

Say the model flags 100 transactions; 40 are truly anomalies; there are 60 real
anomalies in total.

- **Precision** = of what I flagged, how much was right = 40/100 = 0.40.
- **Recall** = of all real anomalies, how many did I catch = 40/60 = 0.67.
- **F1** = one number balancing both (their harmonic mean).
- **PR-AUC** = quality across *all* possible thresholds. Better than accuracy for rare events.
- **Macro-F1** (categorization) = F1 averaged over every category equally, so small categories matter as much as big ones.
- **MAE** (forecasting) = average size of the error, in rupees. Easiest to explain.
- **MASE** = your error divided by a naive "repeat last month" forecast's error (measured on each series' first year). Lower is better; use it to compare methods against each other (values above 1 can still be normal if the test period is harder than the first year).

## 3. Words you will hear

- **Hyperparameter**: a setting you choose before training (e.g. number of clusters k). Not learned.
- **Cross-validation**: repeat train/test several times on different slices to get a steadier score. For time series use *rolling-origin* (always train on the past, test on the next chunk).
- **Feature engineering**: designing good input numbers. Usually matters more than picking a fancy model.
- **Contamination**: expected share of anomalies (we assume ~5%).
- **Inference**: using a trained model on new data.
- **Pipeline**: chaining preprocessing + model so they are trained and applied together.

## 4. Suggested learning path (about 3-4 weeks alongside the project)

1. **Week 1: foundations.** Andrew Ng's *Machine Learning Specialization* (Course 1: regression and classification). Or StatQuest on YouTube for each concept above; short and very clear. Also learn pandas basics (`groupby`, `merge`, `pivot_table`).
2. **Week 2: scikit-learn.** Its official "Getting Started" and "Common pitfalls (data leakage)" pages. Practice: train a `LogisticRegression` on a small dataset, print `classification_report`.
3. **Week 3: time series.** *Forecasting: Principles and Practice* (Hyndman, free online): chapters on baselines, decomposition, and evaluation. This is your forecasting theory.
4. **Week 4: anomaly detection and clustering.** scikit-learn docs for `IsolationForest`, `KMeans`, `silhouette_score`. Then re-read `features/transaction_features.py` and explain each feature out loud.

## 5. How to learn from this repo

- After each ML step I will explain what was trained, on what, and what the metric numbers mean.
- Read the tests in `backend/tests/test_features.py`; they are small examples of what each function does.
- Before I build each model, try to predict which baseline it must beat and why. Then check.
- Keep a notebook of terms you didn't understand; ask me and I'll explain with a concrete example from your data.

## 6. What to say in your viva/report

- Why baselines first (to prove ML adds value).
- Why time-based splits (future leakage).
- Why F1/PR-AUC/MASE and not accuracy.
- Honest limitation: the data is synthetic, so real-world scores would be lower.
- LLM only explains ML outputs; it never computes numbers.
