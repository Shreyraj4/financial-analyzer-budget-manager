"""Retrain the categorizer using users' own labels as extra training data.

Every transaction a user labeled by hand is a verified (description, amount,
category) example. Combining them with the base dataset teaches the model the
merchants and custom categories it previously got wrong or didn't know.

Usage (from backend/): python -m app.ml.categorization.retrain
"""
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.categorization.model import MODEL_PATH, MLCategorizer
from app.ml.dataset import load_labeled_transactions
from app.models import Transaction

TRAIN_COLUMNS = ["description", "amount", "category"]


def user_labeled_frame(db: Session) -> pd.DataFrame:
    stmt = select(Transaction.description, Transaction.amount, Transaction.category).where(
        Transaction.category_source == "user", Transaction.category.is_not(None)
    )
    frame = pd.DataFrame(db.execute(stmt).all(), columns=TRAIN_COLUMNS)
    frame["amount"] = frame["amount"].astype(float)
    return frame


def retrain(db: Session, base: pd.DataFrame | None = None, save_to=MODEL_PATH) -> dict[str, int]:
    base = (load_labeled_transactions() if base is None else base)[TRAIN_COLUMNS]
    user = user_labeled_frame(db)
    combined = pd.concat([base, user], ignore_index=True)
    MLCategorizer("logreg", C=10).fit(combined).save(save_to)

    from app.services.categorization import reload_model  # avoid import cycle at module load

    reload_model()
    return {"base_rows": len(base), "user_labeled_rows": len(user), "categories": int(combined["category"].nunique())}


if __name__ == "__main__":
    from app.database.session import SessionLocal

    with SessionLocal() as session:
        print(retrain(session))
