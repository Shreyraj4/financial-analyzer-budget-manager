from decimal import Decimal

from app.analytics.anomalies import detect_spend_anomalies
from app.analytics.monthly_totals import MonthlyTotal


def _totals(category, spends):
    return [
        MonthlyTotal(category=category, year=2026, month=i + 1, total_spent=Decimal(str(s)))
        for i, s in enumerate(spends)
    ]


def test_spike_is_flagged():
    anomalies = detect_spend_anomalies(_totals("Dining", [1000, 1100, 900, 1050, 5000]))
    assert [(a.category, a.month) for a in anomalies] == [("Dining", 5)]


def test_normal_variation_not_flagged():
    assert detect_spend_anomalies(_totals("Dining", [1000, 1100, 900, 1050, 1000])) == []


def test_too_little_history_skipped():
    assert detect_spend_anomalies(_totals("Dining", [1000, 9000])) == []


def test_flat_baseline_jump_flagged():
    anomalies = detect_spend_anomalies(_totals("Rent", [500, 500, 500, 900]))
    assert len(anomalies) == 1 and anomalies[0].month == 4


def test_dips_not_flagged():
    assert detect_spend_anomalies(_totals("Dining", [1000, 1100, 900, 1050, 10])) == []
