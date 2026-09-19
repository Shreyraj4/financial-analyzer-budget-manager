"""Evaluate forecasters with rolling-origin validation, then train the final model.

Protocol
  1. Every candidate ML configuration is run over all rolling origins.
  2. The best configuration per family is chosen using the VALIDATION half only.
  3. Reported test numbers come from the later TEST half, which never
     influenced any choice.
  4. The chosen model is refit on all data and saved.

Usage (from backend/): python -m app.ml.forecasting.train
"""
import json

from app.config import REPO_ROOT
from app.features.monthly_features import build_forecast_features, monthly_category_panel
from app.ml.dataset import load_labeled_transactions
from app.ml.forecasting.evaluate import (
    BASELINES,
    compute_metrics,
    naive_scales,
    rolling_origin_predictions,
    split_validation_test,
)
from app.ml.forecasting.model import MODEL_PATH, GlobalForecaster

METRICS_PATH = REPO_ROOT / "docs" / "metrics" / "forecasting.json"

CANDIDATES = {
    "ridge_alpha0.1": lambda: GlobalForecaster("ridge", alpha=0.1),
    "ridge_alpha1": lambda: GlobalForecaster("ridge", alpha=1.0),
    "ridge_alpha10": lambda: GlobalForecaster("ridge", alpha=10.0),
    "ridge_alpha100": lambda: GlobalForecaster("ridge", alpha=100.0),
    "gbm_depth2": lambda: GlobalForecaster("gbm", max_depth=2),
    "gbm_depth3": lambda: GlobalForecaster("gbm", max_depth=3),
}


def main() -> None:
    df = load_labeled_transactions()
    panel = monthly_category_panel(df)
    features = build_forecast_features(panel, lags=(1, 2, 3))
    scales = naive_scales(panel)

    preds = rolling_origin_predictions(features, panel, CANDIDATES)
    validation, test = split_validation_test(preds)
    print(f"series={panel.groupby(['user_id', 'category']).ngroups} "
          f"validation_rows={len(validation)} test_rows={len(test)} "
          f"validation={validation['month'].min():%Y-%m}..{validation['month'].max():%Y-%m} "
          f"test={test['month'].min():%Y-%m}..{test['month'].max():%Y-%m}")

    fixed = ["naive_last_month", "moving_average_3", "seasonal_naive", "existing_linear_trend"]
    val_metrics = compute_metrics(validation, fixed + list(CANDIDATES), scales)
    best = {
        family: min((n for n in CANDIDATES if n.startswith(family)), key=lambda n: val_metrics[n]["mase"])
        for family in ("ridge", "gbm")
    }
    print("Chosen on validation:", best)

    reported = fixed + list(best.values())
    test_metrics = compute_metrics(test, reported, scales)

    print(f"\n{'method':24s} {'val MASE':>9s} | {'MAE':>8s} {'RMSE':>8s} {'WAPE':>7s} {'MASE':>7s}  (TEST)")
    for m in reported:
        t = test_metrics[m]
        print(f"{m:24s} {val_metrics[m]['mase']:9.3f} | {t['mae']:8.1f} {t['rmse']:8.1f} {t['wape']:7.3f} {t['mase']:7.3f}")

    per_category = {}
    for cat, g in test.groupby("category"):
        cat_scales = scales.loc[scales.index.isin(g.set_index(["user_id", "category"]).index)]
        per_category[cat] = compute_metrics(g, ["naive_last_month", best["ridge"]], cat_scales)

    winner = min(best.values(), key=lambda n: val_metrics[n]["mase"])
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "protocol": "rolling-origin; settings chosen on validation half, reported on test half",
        "n_series": int(panel.groupby(["user_id", "category"]).ngroups),
        "validation_months": [f"{m:%Y-%m}" for m in sorted(validation["month"].unique())],
        "test_months": [f"{m:%Y-%m}" for m in sorted(test["month"].unique())],
        "validation": val_metrics, "chosen": best, "test": test_metrics,
        "test_per_category": per_category, "production_model": winner,
    }, indent=2))

    CANDIDATES[winner]().fit(features).save(MODEL_PATH)
    print(f"\nProduction model: {winner}\nSaved metrics -> {METRICS_PATH}\nSaved model   -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
