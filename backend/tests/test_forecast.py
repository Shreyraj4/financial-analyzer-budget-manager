from decimal import Decimal

from app.analytics.monthly_totals import MonthlyTotal
from app.ml.forecast import forecast_next_month


def _totals(category: str, monthly_spends: list[tuple[int, int, float]]) -> list[MonthlyTotal]:
    return [
        MonthlyTotal(category=category, year=y, month=m, total_spent=Decimal(str(spend)))
        for y, m, spend in monthly_spends
    ]


def test_single_month_uses_last_value():
    totals = _totals("Food & Dining", [(2026, 1, 1000.0)])
    forecasts = forecast_next_month(totals)
    assert len(forecasts) == 1
    assert forecasts[0].method == "last_value"
    assert forecasts[0].predicted_next_month_spend == Decimal("1000.00")


def test_two_months_uses_average():
    totals = _totals("Food & Dining", [(2026, 1, 1000.0), (2026, 2, 1200.0)])
    forecasts = forecast_next_month(totals)
    assert forecasts[0].method == "average"
    assert forecasts[0].predicted_next_month_spend == Decimal("1100.00")


def test_clear_upward_trend_extrapolates_with_regression():
    # Perfectly linear: +100 per month
    totals = _totals("Food & Dining", [(2026, 1, 1000.0), (2026, 2, 1100.0), (2026, 3, 1200.0)])
    forecasts = forecast_next_month(totals)
    assert forecasts[0].method == "linear_regression"
    assert forecasts[0].history_months == 3
    # Next month (month 4) should continue the trend to ~1300
    assert abs(float(forecasts[0].predicted_next_month_spend) - 1300.0) < 1.0


def test_flat_spend_predicts_same_value():
    totals = _totals("Groceries", [(2026, 1, 500.0), (2026, 2, 500.0), (2026, 3, 500.0)])
    forecasts = forecast_next_month(totals)
    assert abs(float(forecasts[0].predicted_next_month_spend) - 500.0) < 1.0


def test_multiple_categories_are_each_forecast_independently():
    totals = _totals("Food & Dining", [(2026, 1, 1000.0), (2026, 2, 1100.0), (2026, 3, 1200.0)])
    totals += _totals("Groceries", [(2026, 1, 500.0), (2026, 2, 500.0), (2026, 3, 500.0)])
    forecasts = forecast_next_month(totals)
    categories = {f.category for f in forecasts}
    assert categories == {"Food & Dining", "Groceries"}


def test_prediction_never_negative():
    # Sharp downward trend that would go negative if extrapolated naively
    totals = _totals("Entertainment", [(2026, 1, 300.0), (2026, 2, 150.0), (2026, 3, 0.0)])
    forecasts = forecast_next_month(totals)
    assert float(forecasts[0].predicted_next_month_spend) >= 0.0
