from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Budget, Transaction

WARNING_THRESHOLD = Decimal("0.8")


@dataclass
class BudgetStatus:
    budget_id: int
    category: str
    period: str
    start_date: date
    end_date: date
    budget_amount: Decimal
    spent: Decimal
    remaining: Decimal
    percent_used: Decimal
    status: str  # "ok" | "warning" | "exceeded"


def evaluate_budget(amount: Decimal, spent: Decimal) -> tuple[Decimal, Decimal, str]:
    """Pure budget math: returns (remaining, percent_used, status)."""
    remaining = amount - spent
    percent_used = (spent / amount * 100).quantize(Decimal("0.01")) if amount > 0 else Decimal("0.00")
    if spent > amount:
        status = "exceeded"
    elif amount > 0 and spent >= amount * WARNING_THRESHOLD:
        status = "warning"
    else:
        status = "ok"
    return remaining, percent_used, status


def spend_by_category(db: Session, user_id: int, start: date, end: date) -> dict[str, Decimal]:
    """Debit spend per category (positive numbers) within [start, end] inclusive."""
    stmt = (
        select(Transaction.category, func.sum(-Transaction.amount))
        .where(
            Transaction.user_id == user_id,
            Transaction.transaction_type == "debit",
            Transaction.category.is_not(None),
            Transaction.transaction_date >= start,
            Transaction.transaction_date <= end,
        )
        .group_by(Transaction.category)
    )
    return {category: total for category, total in db.execute(stmt)}


def budget_statuses(db: Session, budgets: list[Budget]) -> list[BudgetStatus]:
    statuses = []
    for b in budgets:
        spent = spend_by_category(db, b.user_id, b.start_date, b.end_date).get(b.category, Decimal("0"))
        remaining, percent_used, status = evaluate_budget(b.amount, spent)
        statuses.append(
            BudgetStatus(
                budget_id=b.id,
                category=b.category,
                period=b.period,
                start_date=b.start_date,
                end_date=b.end_date,
                budget_amount=b.amount,
                spent=spent,
                remaining=remaining,
                percent_used=percent_used,
                status=status,
            )
        )
    return statuses
