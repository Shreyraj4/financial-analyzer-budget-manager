from dataclasses import dataclass

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models import Transaction
from app.schemas.transaction import TransactionIn
from app.services.categorization import build_categorizer


@dataclass
class ImportResult:
    submitted_count: int
    imported_count: int
    duplicate_count: int


def import_transactions(db: Session, user_id: int, transactions: list[TransactionIn]) -> ImportResult:
    if not transactions:
        return ImportResult(submitted_count=0, imported_count=0, duplicate_count=0)

    categorizer = build_categorizer(db, user_id)
    auto = categorizer.categorize_rows([(t.description, t.amount) for t in transactions])
    values = []
    for t, guess in zip(transactions, auto):
        if t.category:  # explicit user label wins over any automatic decision
            category, subcategory, source, confidence = t.category.strip(), t.subcategory, "user", None
        else:
            category, subcategory, source = guess.category, guess.subcategory, guess.source
            confidence = guess.confidence if guess.category else None
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
                "category_source": source,
                "category_confidence": confidence,
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
