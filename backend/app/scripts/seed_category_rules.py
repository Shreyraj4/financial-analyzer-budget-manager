"""One-off script to seed CategoryRule rows matching the merchants used in
the synthetic sample data (see data/generate_sample_transactions.py).

Usage: python -m app.scripts.seed_category_rules
"""
from sqlalchemy import select

from app.database.session import SessionLocal
from app.models import CategoryRule

RULES = [
    ("SWIGGY", "Food & Dining", "Delivery"),
    ("ZOMATO", "Food & Dining", "Delivery"),
    ("DMART", "Groceries", "Supermarket"),
    ("NETFLIX", "Subscriptions", "Streaming"),
    ("UBER", "Transport", "Rideshare"),
    ("AMAZON", "Shopping", "Online"),
    ("ELECTRICITY BOARD", "Utilities", "Electricity"),
    ("SALARY", "Income", None),
]


def run() -> None:
    db = SessionLocal()
    try:
        existing = {r.merchant_pattern for r in db.scalars(select(CategoryRule))}
        created = 0
        for pattern, category, subcategory in RULES:
            if pattern in existing:
                continue
            db.add(CategoryRule(merchant_pattern=pattern, category=category, subcategory=subcategory))
            created += 1
        db.commit()
        print(f"Seeded {created} new category rule(s); {len(existing)} already existed.")
    finally:
        db.close()


if __name__ == "__main__":
    run()
