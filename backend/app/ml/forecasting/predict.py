"""Next-month forecasts for a user's monthly spend panel."""
from decimal import Decimal

import pandas as pd

from app.analytics.monthly_totals import MonthlyTotal
from app.features.monthly_features import build_forecast_features
from app.ml.forecast import CategoryForecast, forecast_next_month
from app.ml.forecasting.model import GlobalForecaster

MIN_HISTORY_MONTHS = 3  # the feature set uses 3 lags


def _with_next_month_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Appends one unknown-spend row per series for the month after its last.
    Features only look backwards, so the placeholder never leaks into itself."""
    last = panel.sort_values("month").groupby(["user_id", "category"], as_index=False).tail(1)
    future = last.assign(month=last["month"] + pd.offsets.MonthBegin(1), spend=float("nan"))
    return pd.concat([panel, future], ignore_index=True)


def predict_next_month(panel: pd.DataFrame, model: GlobalForecaster | None) -> list[CategoryForecast]:
    """One forecast per category for a single user's panel (from ``monthly_category_panel``).

    Categories with at least MIN_HISTORY_MONTHS of history use the ML model;
    shorter ones fall back to the simple average/last-value forecaster, so
    every category always gets an answer and the ``method`` field says how.
    """
    if panel.empty:
        return []

    results: dict[str, CategoryForecast] = {}
    if model is not None:
        extended = _with_next_month_rows(panel)
        next_month = extended["month"].max()
        features = build_forecast_features(extended, lags=(1, 2, 3))
        rows = features[features["spend"].isna()]
        if not rows.empty:
            predictions = model.predict(rows)
            history = panel.groupby("category").size()
            for category, value in zip(rows["category"], predictions):
                results[category] = CategoryForecast(
                    category=category,
                    predicted_next_month_spend=Decimal(str(round(float(value), 2))),
                    method="ml_global",
                    history_months=int(history[category]),
                )

    missing = panel[~panel["category"].isin(results)]
    if not missing.empty:
        totals = [
            MonthlyTotal(category=r.category, year=r.month.year, month=r.month.month, total_spent=Decimal(str(r.spend)))
            for r in missing.itertuples(index=False)
        ]
        # The fallback ignores zero-fill months only implicitly, matching the original behavior.
        for f in forecast_next_month([t for t in totals if t.total_spent > 0]):
            results[f.category] = f

    return sorted(results.values(), key=lambda f: f.category)
