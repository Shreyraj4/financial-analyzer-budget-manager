"""Rolling-origin (walk-forward) evaluation of spend forecasters.

For each origin month M: train only on rows whose month is before M, then
predict month M. Repeating this over consecutive months mimics real use -
the model is always older than the data it forecasts - and never shuffles
time. Metrics:

  MAE    average absolute error, in rupees (easy to explain)
  RMSE   like MAE but punishes big misses harder
  WAPE   total absolute error / total actual spend (scale-free, safe with zeros)
  MASE   per-series MAE divided by the MAE a "repeat last month" forecast
         achieved on that series' first year; < 1 means better than naive
"""
from decimal import Decimal

import numpy as np
import pandas as pd

from app.analytics.monthly_totals import MonthlyTotal
from app.ml.forecast import forecast_next_month  # the original per-series linear-trend forecaster
from app.ml.forecasting import model as M

FIRST_ORIGIN_INDEX = 12   # need a full year of history before the first test month
SCALE_MONTHS = 12          # MASE scale is measured on each series' first year


def linear_trend_predictions(panel: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predictions from the project's original forecaster, using only history before each test month."""
    out = np.zeros(len(test))
    history = {k: g.sort_values("month") for k, g in panel.groupby(["user_id", "category"])}
    for i, row in enumerate(test.itertuples(index=False)):
        h = history[(row.user_id, row.category)]
        h = h[h["month"] < row.month]
        totals = [
            MonthlyTotal(category=row.category, year=m.year, month=m.month, total_spent=Decimal(str(s)))
            for m, s in zip(h["month"], h["spend"])
        ]
        out[i] = float(forecast_next_month(totals)[0].predicted_next_month_spend)
    return out


BASELINES = {
    "naive_last_month": M.naive,
    "moving_average_3": M.moving_average_3,
    "seasonal_naive": M.seasonal_naive,
}


def rolling_origin_predictions(features: pd.DataFrame, panel: pd.DataFrame, ml_models: dict[str, callable]) -> pd.DataFrame:
    """One row per (user, category, origin month) with actuals and every method's prediction.

    ``ml_models`` maps a name to a factory returning a fresh unfitted GlobalForecaster.
    """
    # ``t`` is the month's position within its series, so this is "after one full year of data".
    origins = sorted(features.loc[features["t"] >= FIRST_ORIGIN_INDEX, "month"].unique())
    parts = []
    for origin in origins:
        train = features[features["month"] < origin]
        test = features[features["month"] == origin].copy()
        keep = ["user_id", "category", "month", "spend"]
        rows = test[keep].rename(columns={"spend": "actual"}).reset_index(drop=True)
        for name, fn in BASELINES.items():
            rows[name] = fn(test)
        rows["existing_linear_trend"] = linear_trend_predictions(panel, test)
        for name, factory in ml_models.items():
            rows[name] = factory().fit(train).predict(test)
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def naive_scales(panel: pd.DataFrame) -> pd.Series:
    """MASE denominator per (user, category): mean |y_t - y_{t-1}| over the first year."""
    def scale(g: pd.DataFrame) -> float:
        first = g.sort_values("month")["spend"].to_numpy()[:SCALE_MONTHS]
        return float(np.abs(np.diff(first)).mean()) if len(first) > 1 else np.nan

    return panel.groupby(["user_id", "category"]).apply(scale, include_groups=False)


def compute_metrics(preds: pd.DataFrame, methods: list[str], scales: pd.Series) -> dict[str, dict[str, float]]:
    keyed = preds.set_index(["user_id", "category"])
    denom = scales.reindex(keyed.index).to_numpy()
    valid = np.isfinite(denom) & (denom > 0)
    results = {}
    for m in methods:
        err = (preds[m] - preds["actual"]).to_numpy()
        abs_err = np.abs(err)
        per_series = pd.Series(abs_err[valid]).groupby([keyed.index.get_level_values(0)[valid], keyed.index.get_level_values(1)[valid]]).mean()
        mase = float((per_series / scales.reindex(per_series.index)).mean())
        results[m] = {
            "mae": round(float(abs_err.mean()), 2),
            "rmse": round(float(np.sqrt((err**2).mean())), 2),
            "wape": round(float(abs_err.sum() / preds["actual"].sum()), 4),
            "mase": round(mase, 4),
        }
    return results


def split_validation_test(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First half of the origins -> validation (choose settings); second half -> test (report)."""
    origins = sorted(preds["month"].unique())
    cut = len(origins) // 2
    return preds[preds["month"].isin(origins[:cut])], preds[preds["month"].isin(origins[cut:])]
