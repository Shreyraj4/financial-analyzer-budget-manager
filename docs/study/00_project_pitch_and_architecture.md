# Project pitch and architecture (start here)

## 30-second pitch (memorize this)

"I built a personal-finance analysis system where machine learning does the
analysis and an LLM only explains it. Users upload bank statements; the system
cleans them, categorizes every transaction with an ML classifier (asking the
user when it's unsure), forecasts next month's spending per category, and
flags anomalous transactions with reasons. Every model is compared against
simple baselines using time-aware evaluation, so the reported numbers are
honest. A FastAPI backend serves the results to a web dashboard."

## The one design principle (interviewers love this)

**The LLM is never the source of truth for numbers.** ML models produce
structured results (category, forecast, anomaly + reasons). The LLM only turns
those into readable sentences. Why: LLMs can hallucinate arithmetic; models can
be measured and validated.

## Architecture

```
CSV/PDF upload -> parse -> clean -> features
                                      |
        +-----------------------------+-----------------------------+
        v                             v                             v
 Categorizer (TF-IDF +         Forecaster (global Ridge/GBM   Anomaly detector
 Logistic Regression)          on lag features)               (Isolation Forest)
        |                             |                             |
        +---------------> PostgreSQL (Supabase) <-------------------+
                                      |
                              FastAPI endpoints
                                      |
                       LLM explanation layer (planned)
                                      |
                        Web dashboard (planned)
```

Code map:

| Concern | Where |
|---|---|
| Cleaning narrations | `backend/app/preprocessing/text.py` |
| Features | `backend/app/features/` |
| Categorization | `backend/app/ml/categorization/` |
| Forecasting | `backend/app/ml/forecasting/` |
| Anomaly detection | `backend/app/ml/anomaly/` |
| Serving (DB + API) | `backend/app/services/`, `backend/app/api/` |
| Recorded results | `docs/metrics/*.json` |

## Training vs inference (must be able to explain)

- **Training (offline):** `python -m app.ml.<task>.train` learns from data,
  evaluates, and saves a `.joblib` file.
- **Inference (online):** the API loads the saved file and predicts. It never
  trains during a request.

## Why synthetic data, and the honest caveat

Real bank data is private. I wrote a generator (`data/generate_synthetic_dataset.py`):
12 users x 24 months, ~14.6k transactions, 123 merchants, six spending personas,
inflation and seasonality, and injected labeled anomalies. **Caveat to state
proactively:** because I generated the labels, scores are optimistic compared
with real-world data. Saying this yourself makes you sound rigorous, not weak.

## Likely interview questions

**Q: Why not just ask an LLM to analyze the transactions?**
A: LLMs are unreliable at arithmetic and can't be validated the way a model can.
I use ML for numbers (measurable error) and the LLM only for explanation.

**Q: What's your ML contribution?**
A: Three models plus the evaluation methodology: a text classifier for merchant
categories with a confidence-gated human-in-the-loop, a global forecasting model
evaluated with rolling-origin validation, and an unsupervised anomaly detector
compared against rule and statistical baselines, with per-anomaly-type analysis.

**Q: What would you do with real data?**
A: Re-run the same pipeline; expect lower scores. Add user feedback loops
(already built for categories), and collect real labeled anomalies.

**Q: How is this different from a budgeting CRUD app?**
A: The value is in the models and how they're validated, not in storing data.

## Reading

- scikit-learn user guide (skim the table of contents): https://scikit-learn.org/stable/user_guide.html
- Google, "Rules of Machine Learning" (practical, short): https://developers.google.com/machine-learning/guides/rules-of-ml
