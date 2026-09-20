"""Backtest budget recommenders, choose one on validation, report on test, save it.

Timeline (12 rolling forecast origins, Jan-Dec of the second year):
  Jan-Mar  calibration only (residuals used to set margins)
  Apr-Jun  validation  (choose which method/history variant to ship)
  Jul-Dec  test        (reported; never used to choose anything)
Margins for any month use only residuals from strictly earlier months.

Compared at the same target coverage (default 80% => ~20% breach rate):
  naive / moving average / seasonal naive / original linear trend / Ridge / GBM
  each with the SAME conformal margin procedure, plus versions trained on
  anomaly-cleaned history (flagged transactions removed).

Usage (from backend/): python -m app.ml.recommendation.train
"""
import json

import pandas as pd

from app.config import REPO_ROOT
from app.features.monthly_features import build_forecast_features, monthly_category_panel
from app.features.transaction_features import build_transaction_features
from app.ml.anomaly.detectors import IsolationForestDetector
from app.ml.anomaly.evaluate import best_f1_threshold, month_splits
from app.ml.dataset import load_labeled_transactions
from app.ml.forecasting.evaluate import rolling_origin_predictions
from app.ml.forecasting.model import GlobalForecaster
from app.ml.recommendation.budget import DEFAULT_COVERAGE, BudgetCalibrator, BudgetModel
from app.ml.recommendation.evaluate import (
    budget_metrics,
    conformal_budgets,
    split_months,
    uncalibrated_budgets,
)

METRICS_PATH = REPO_ROOT / "docs" / "metrics" / "budget_recommendation.json"
N_CALIBRATION, N_VALIDATION = 3, 3
KEYS = ["user_id", "category", "month"]
ML = {"ridge": lambda: GlobalForecaster("ridge", alpha=0.1), "gbm": lambda: GlobalForecaster("gbm", max_depth=2)}


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """Drops debits flagged by an Isolation Forest fitted ONLY on the first 12 months
    (threshold from the following 6), so later months are scored out-of-sample."""
    feats = build_transaction_features(df)
    debit = (feats["is_credit"] == 0).to_numpy()
    d, f = df[debit], feats[debit]
    train, val, _ = month_splits(d["date"])
    det = IsolationForestDetector().fit(f[train], d["user_id"][train])
    thr = best_f1_threshold(d["is_anomaly"][val].to_numpy(), det.score(f[val], d["user_id"][val]))
    flagged = det.score(f, d["user_id"]) >= thr
    print(f"anomaly cleaning: removed {flagged.sum()} of {len(d)} debits ({flagged.mean():.1%})")
    return df.drop(index=d.index[flagged])


def run_rolling(df: pd.DataFrame, ml_names: list[str]):
    panel = monthly_category_panel(df)
    features = build_forecast_features(panel, lags=(1, 2, 3))
    preds = rolling_origin_predictions(features, panel, {n: ML[n] for n in ml_names})
    return panel, features, preds


def main() -> None:
    df = load_labeled_transactions()
    panel_raw, feats_raw, raw = run_rolling(df, ["ridge", "gbm"])
    panel_clean, feats_clean, clean = run_rolling(clean_history(df), ["ridge", "gbm"])

    # Judge every budget against what was REALLY spent (raw actuals), with
    # forecasts built from raw or cleaned history.
    merged = raw.merge(
        clean[KEYS + ["ridge", "gbm", "hist_mean"]].rename(columns={"ridge": "ridge_clean", "gbm": "gbm_clean", "hist_mean": "hist_mean_clean"}),
        on=KEYS, how="inner",
    )
    origins = sorted(merged["month"].unique())
    first_val, first_test = split_months(origins, N_CALIBRATION, N_VALIDATION)
    print(f"origins={len(origins)} calibration<{first_val:%Y-%m} validation {first_val:%Y-%m}..<{first_test:%Y-%m} test>={first_test:%Y-%m} rows={len(merged)}")

    methods = {
        "naive_last_month": ("naive_last_month", "hist_mean"),
        "moving_average_3": ("moving_average_3", "hist_mean"),
        "seasonal_naive": ("seasonal_naive", "hist_mean"),
        "existing_linear_trend": ("existing_linear_trend", "hist_mean"),
        "ridge": ("ridge", "hist_mean"),
        "gbm": ("gbm", "hist_mean"),
        "ridge_clean_history": ("ridge_clean", "hist_mean_clean"),
        "gbm_clean_history": ("gbm_clean", "hist_mean_clean"),
    }

    def evaluate_all(first_month, last_month, coverage):
        res = {}
        for name, (col, hist) in methods.items():
            b = conformal_budgets(merged, col, coverage, first_month, hist_col=hist)
            b = b[b["month"] < last_month] if last_month is not None else b
            res[name] = budget_metrics(b, coverage)
        return res

    validation = evaluate_all(first_val, first_test, DEFAULT_COVERAGE)
    test = evaluate_all(first_test, None, DEFAULT_COVERAGE)
    no_margin = {}
    for name in ("ridge", "gbm_clean_history"):
        col, hist = methods[name]
        no_margin[name] = budget_metrics(uncalibrated_budgets(merged, col, first_test, hist), DEFAULT_COVERAGE)

    print(f"\nTarget coverage {DEFAULT_COVERAGE:.0%} (breach rate should be ~{1 - DEFAULT_COVERAGE:.0%})")
    print(f"{'method':24s} {'val pinball':>11s} | {'breach':>7s} {'slack':>7s} {'shortfall':>9s} {'pinball':>9s}  (TEST)")
    for name in methods:
        t = test[name]
        print(f"{name:24s} {validation[name]['pinball_loss']:11.1f} | {t['breach_rate']:7.3f} {t['slack']:7.3f} {t['shortfall']:9.3f} {t['pinball_loss']:9.1f}")
    for name, m in no_margin.items():
        print(f"{name + ' (NO margin)':24s} {'':11s} | {m['breach_rate']:7.3f} {m['slack']:7.3f} {m['shortfall']:9.3f} {m['pinball_loss']:9.1f}")

    # Calibration sweep for the shipped-candidate family: does 'coverage' mean what it says?
    sweep = {}
    for cov in (0.6, 0.7, 0.8, 0.9):
        b = conformal_budgets(merged, "ridge_clean", cov, first_test, hist_col="hist_mean_clean")
        sweep[str(cov)] = budget_metrics(b, cov)
    print("\nCalibration sweep (ridge_clean_history), target vs achieved coverage on test:")
    for cov, m in sweep.items():
        print(f"  target {float(cov):.0%} -> achieved {1 - m['breach_rate']:.1%}, slack {m['slack']:.3f}")

    # Choose the shipped variant on VALIDATION pinball loss among the ML methods.
    ml_methods = ["ridge", "gbm", "ridge_clean_history", "gbm_clean_history"]
    winner = min(ml_methods, key=lambda n: validation[n]["pinball_loss"])
    print(f"\nShipped variant (best validation pinball among ML): {winner}")

    # Refit the winner on all data; calibrate on ALL out-of-sample residuals.
    kind, use_clean = winner.split("_")[0], winner.endswith("clean_history")
    col, hist = methods[winner]
    feats = feats_clean if use_clean else feats_raw
    residuals = merged[merged[hist] > 0].assign(r=lambda d: (d["actual"] - d[col]) / d[hist])[["category", "r"]]
    model = BudgetModel(ML[kind]().fit(feats), BudgetCalibrator().fit(residuals, DEFAULT_COVERAGE), use_clean_history=use_clean, name=winner)
    model.save()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "protocol": "12 rolling origins: 3 calibration, 3 validation, 6 test; margins from strictly earlier months only",
        "target_coverage": DEFAULT_COVERAGE, "validation": validation, "test": test,
        "test_without_margin": no_margin, "calibration_sweep_test": sweep, "shipped": winner,
        "categories_with_own_margin": sorted(model.calibrator.by_category_), "global_margin_ratio": round(model.calibrator.global_q_, 4),
    }, indent=2))
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
