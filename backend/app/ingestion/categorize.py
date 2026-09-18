from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CategoryRule


def load_rules(db: Session) -> list[CategoryRule]:
    return list(db.scalars(select(CategoryRule)))


def categorize(merchant: str | None, rules: list[CategoryRule]) -> tuple[str | None, str | None]:
    """Matches a normalized merchant string against known patterns.

    Longest-pattern-first so a specific rule (e.g. "SWIGGY BANGALORE")
    wins over a more general one (e.g. "SWIGGY") when both match.
    """
    if not merchant:
        return None, None

    for rule in sorted(rules, key=lambda r: len(r.merchant_pattern), reverse=True):
        if rule.merchant_pattern in merchant:
            return rule.category, rule.subcategory

    return None, None
