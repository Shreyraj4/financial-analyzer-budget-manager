"""Evaluate all categorization models, then train and save the production one.

Usage (from backend/): python -m app.ml.categorization.train
Writes: docs/metrics/categorization.json and backend/models/categorizer.joblib
"""
import json

import pandas as pd

from app.config import REPO_ROOT
from app.ml.categorization.evaluate import (
    confidence_tradeoff,
    cross_val_predict_df,
    detailed_report,
    evaluate_models,
)
from app.ml.categorization.model import (
    MODEL_PATH,
    MLCategorizer,
    MerchantLookupBaseline,
    SubstringRulesBaseline,
)
from app.ml.dataset import load_labeled_transactions
from app.scripts.seed_category_rules import RULES

METRICS_PATH = REPO_ROOT / "docs" / "metrics" / "categorization.json"
PRODUCTION = "logreg_text+amount_C10"


def candidate_models() -> dict[str, object]:
    return {
        "baseline_seed_rules": SubstringRulesBaseline([(p, c) for p, c, _ in RULES]),
        "baseline_merchant_lookup": MerchantLookupBaseline(),
        "logreg_text_C10": MLCategorizer("logreg", C=10, use_amount=False),
        "logreg_text+amount_C1": MLCategorizer("logreg", C=1),
        PRODUCTION: MLCategorizer("logreg", C=10),
        "logreg_text+amount_C100": MLCategorizer("logreg", C=100),
        "linearsvc_text+amount_C1": MLCategorizer("linearsvc", C=1),
    }


def main() -> None:
    df = load_labeled_transactions()
    models = candidate_models()
    results = evaluate_models(models, df)

    print(f"{'model':32s} {'known F1':>9s} {'known acc':>10s} {'unseen F1':>10s} {'unseen acc':>11s}")
    for name, r in results.items():
        k, u = r["known_merchants"], r["unseen_merchants"]
        print(f"{name:32s} {k['macro_f1']:9.3f} {k['accuracy']:10.3f} {u['macro_f1']:10.3f} {u['accuracy']:11.3f}")

    prod = models[PRODUCTION]
    report = {}
    for scenario in ("known_merchants", "unseen_merchants"):
        preds = cross_val_predict_df(prod, df, scenario)
        report[scenario] = detailed_report(df["category"], preds)

    # Confidence trade-off, using out-of-fold probabilities under the unseen-merchant scenario.
    from app.ml.categorization.evaluate import _folds
    conf = pd.Series(index=df.index, dtype=float)
    pred = pd.Series(index=df.index, dtype=object)
    for tr, te in _folds(df, "unseen_merchants"):
        m = MLCategorizer("logreg", C=10).fit(df.iloc[tr])
        out = m.predict_with_confidence(df.iloc[te])
        conf.iloc[te], pred.iloc[te] = out["confidence"].to_numpy(), out["category"].to_numpy()
    tradeoff = confidence_tradeoff(df["category"], conf, pred)
    print("\nConfidence trade-off (unseen merchants):", tradeoff)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {"dataset_rows": len(df), "n_merchants": int(df["merchant_key"].nunique()), "models": results,
             "production_model": PRODUCTION, "production_report": report, "confidence_tradeoff_unseen": tradeoff},
            indent=2,
        )
    )

    MLCategorizer("logreg", C=10).fit(df).save(MODEL_PATH)
    print(f"\nSaved metrics -> {METRICS_PATH}\nSaved model   -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
