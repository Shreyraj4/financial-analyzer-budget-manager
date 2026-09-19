from decimal import Decimal

from app.analytics.budget_status import evaluate_budget


def test_under_threshold_is_ok():
    remaining, pct, status = evaluate_budget(Decimal("1000"), Decimal("500"))
    assert (remaining, pct, status) == (Decimal("500"), Decimal("50.00"), "ok")


def test_at_80_percent_is_warning():
    assert evaluate_budget(Decimal("1000"), Decimal("800"))[2] == "warning"


def test_exactly_at_budget_is_warning_not_exceeded():
    assert evaluate_budget(Decimal("1000"), Decimal("1000"))[2] == "warning"


def test_over_budget_is_exceeded_with_negative_remaining():
    remaining, pct, status = evaluate_budget(Decimal("1000"), Decimal("1250"))
    assert status == "exceeded"
    assert remaining == Decimal("-250")
    assert pct == Decimal("125.00")


def test_zero_spend_is_ok():
    assert evaluate_budget(Decimal("1000"), Decimal("0"))[2] == "ok"
