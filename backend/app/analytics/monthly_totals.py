from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import extract, func, select
from sqlalchemy.orm import Session

from app.models import Transaction


@dataclass
class MonthlyTotal:
    category: str
    year: int
    month: int
    total_spent: Decimal  # absolute value of debit spend, positive


def category_monthly_totals(db: Session, user_id: int) -> list[MonthlyTotal]:
    """Sums debit spend per (category, year, month) for a user.

    Only debits count as "spend" - credits (salary, refunds) are excluded
    since forecasting spend should predict money going out, not in.
    Uncategorized transactions (category is NULL) are excluded because a
    forecast per category is meaningless without a category.
    """
    year_col = extract("year", Transaction.transaction_date)
    month_col = extract("month", Transaction.transaction_date)

    stmt = (
        select(
            Transaction.category,
            year_col.label("year"),
            month_col.label("month"),
            func.sum(-Transaction.amount).label("total_spent"),
        )
        .where(
            Transaction.user_id == user_id,
            Transaction.transaction_type == "debit",
            Transaction.category.is_not(None),
        )
        .group_by(Transaction.category, year_col, month_col)
        .order_by(Transaction.category, year_col, month_col)
    )

    return [
        MonthlyTotal(category=row.category, year=int(row.year), month=int(row.month), total_spent=row.total_spent)
        for row in db.execute(stmt)
    ]


def month_key(year: int, month: int) -> date:
    """Sortable/plottable representation of a (year, month) pair."""
    return date(year, month, 1)
