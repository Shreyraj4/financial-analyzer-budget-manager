from dataclasses import dataclass

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.ingestion.categorize import categorize, load_rules
from app.models import Transaction
from app.schemas.transaction import TransactionIn


@dataclass
class ImportResult:
    submitted_count: int
    imported_count: int
    duplicate_count: int


def import_transactions(db: Session, user_id: int, transactions: list[TransactionIn]) -> ImportResult:
    if not transactions:
        return ImportResult(submitted_count=0, imported_count=0, duplicate_count=0)

    rules = load_rules(db)
    values = []
    for t in transactions:
        category, subcategory = categorize(t.merchant, rules)
        values.append(
            {
                "user_id": user_id,
                "transaction_date": t.transaction_date,
                "description": t.description,
                "merchant": t.merchant,
                "amount": t.amount,
                "transaction_type": t.transaction_type,
                "category": category,
                "subcategory": subcategory,
                "source": t.source,
            }
        )

    stmt = (
        pg_insert(Transaction)
        .values(values)
        .on_conflict_do_nothing(constraint="uq_transaction_dedupe")
        .returning(Transaction.id)
    )
    inserted_ids = db.execute(stmt).scalars().all()
    db.commit()

    imported_count = len(inserted_ids)
    return ImportResult(
        submitted_count=len(transactions),
        imported_count=imported_count,
        duplicate_count=len(transactions) - imported_count,
    )
