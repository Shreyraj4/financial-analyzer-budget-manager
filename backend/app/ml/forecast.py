from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from sklearn.linear_model import LinearRegression

from app.analytics.monthly_totals import MonthlyTotal, month_key

MIN_POINTS_FOR_REGRESSION = 3


@dataclass
class CategoryForecast:
    category: str
    predicted_next_month_spend: Decimal
    method: str  # "linear_regression" | "average" | "last_value"
    history_months: int


def _months_since_first(months: list) -> np.ndarray:
    first = months[0]
    return np.array([[(m.year - first.year) * 12 + (m.month - first.month)] for m in months], dtype=float)


def forecast_next_month(totals: list[MonthlyTotal]) -> list[CategoryForecast]:
    """Fits a simple linear trend per category over its monthly spend
    history and predicts next month's spend.

    Falls back to a plain average with too little history to fit a
    trend line meaningfully (2 points always yield a "perfect" line
    that's really just noise), and to the single known value with only
    one data point.
    """
    by_category: dict[str, list[MonthlyTotal]] = {}
    for t in totals:
        by_category.setdefault(t.category, []).append(t)

    forecasts = []
    for category, rows in by_category.items():
        rows = sorted(rows, key=lambda r: month_key(r.year, r.month))
        spends = [float(r.total_spent) for r in rows]

        if len(rows) == 1:
            forecasts.append(
                CategoryForecast(
                    category=category,
                    predicted_next_month_spend=Decimal(str(round(spends[0], 2))),
                    method="last_value",
                    history_months=1,
                )
            )
            continue

        if len(rows) < MIN_POINTS_FOR_REGRESSION:
            avg = sum(spends) / len(spends)
            forecasts.append(
                CategoryForecast(
                    category=category,
                    predicted_next_month_spend=Decimal(str(round(avg, 2))),
                    method="average",
                    history_months=len(rows),
                )
            )
            continue

        x = _months_since_first(rows)
        y = np.array(spends, dtype=float)
        model = LinearRegression().fit(x, y)
        next_month_index = np.array([[x[-1, 0] + 1]])
        prediction = float(model.predict(next_month_index)[0])
        prediction = max(prediction, 0.0)  # spend can't be negative

        forecasts.append(
            CategoryForecast(
                category=category,
                predicted_next_month_spend=Decimal(str(round(prediction, 2))),
                method="linear_regression",
                history_months=len(rows),
            )
        )

    return sorted(forecasts, key=lambda f: f.category)
