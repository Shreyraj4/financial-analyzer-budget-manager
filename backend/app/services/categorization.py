"""Decides a category for each transaction, and learns from the user.

Priority (first match wins):
  1. user   - the user already labeled this merchant (their word is final)
  2. rule   - curated CategoryRule table
  3. model  - ML prediction, only if confidence >= CONFIDENCE_THRESHOLD
  4. none   - left uncategorized for the user to label; the model's best
              guess is still attached as ``suggested_category``
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ingestion.categorize import categorize as rule_categorize
from app.ml.categorization.model import MODEL_PATH, MLCategorizer
from app.models import CategoryRule, Transaction
from app.preprocessing.text import extract_merchant

logger = logging.getLogger(__name__)

# Chosen from docs/metrics/categorization.json: at 0.7 the model is ~67% accurate
# even on never-seen merchants, and far higher on known ones.
CONFIDENCE_THRESHOLD = 0.7


@dataclass
class Categorization:
    category: str | None
    subcategory: str | None
    source: str | None  # "user" | "rule" | "model" | None
    confidence: float | None
    suggested_category: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.category is None


@lru_cache
def get_model() -> MLCategorizer | None:
    """The trained categorizer, or None if it hasn't been trained yet
    (the app then falls back to rules + user labels only)."""
    if not MODEL_PATH.exists():
        return None
    try:
        return MLCategorizer.load(MODEL_PATH)
    except Exception:
        logger.exception("Could not load categorizer model at %s", MODEL_PATH)
        return None


def reload_model() -> None:
    get_model.cache_clear()


def load_user_labels(db: Session, user_id: int) -> dict[str, tuple[str, str | None]]:
    """merchant -> (category, subcategory) from this user's own labels; the most recent wins."""
    stmt = (
        select(Transaction.description, Transaction.category, Transaction.subcategory)
        .where(Transaction.user_id == user_id, Transaction.category_source == "user", Transaction.category.is_not(None))
        .order_by(Transaction.id)
    )
    return {extract_merchant(d): (c, s) for d, c, s in db.execute(stmt)}


class Categorizer:
    def __init__(
        self,
        rules: list[CategoryRule],
        user_labels: dict[str, tuple[str, str | None]],
        model: MLCategorizer | None,
        threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.rules = rules
        self.user_labels = user_labels
        self.model = model
        self.threshold = threshold

    def categorize_rows(self, rows: list[tuple[str, Decimal | float | None]]) -> list[Categorization]:
        """``rows`` = (description, amount) pairs; returns one result per row, in order."""
        results: list[Categorization | None] = [None] * len(rows)
        needs_model: list[int] = []

        for i, (description, _) in enumerate(rows):
            merchant = extract_merchant(description)
            if merchant in self.user_labels:
                category, subcategory = self.user_labels[merchant]
                results[i] = Categorization(category, subcategory, "user", None)
                continue
            category, subcategory = rule_categorize(merchant, self.rules)
            if category is not None:
                results[i] = Categorization(category, subcategory, "rule", 1.0)
                continue
            needs_model.append(i)

        if needs_model and self.model is not None:
            frame = pd.DataFrame(
                {
                    "description": [rows[i][0] for i in needs_model],
                    "amount": [float(rows[i][1] or 0) for i in needs_model],
                }
            )
            predictions = self.model.predict_with_confidence(frame)
            for i, (category, confidence) in zip(needs_model, predictions[["category", "confidence"]].itertuples(index=False)):
                confidence = round(float(confidence), 3)
                if confidence >= self.threshold:
                    results[i] = Categorization(category, None, "model", confidence)
                else:
                    results[i] = Categorization(None, None, None, confidence, suggested_category=category)
        for i in needs_model:
            if results[i] is None:
                results[i] = Categorization(None, None, None, None)

        return results  # type: ignore[return-value]


def build_categorizer(db: Session, user_id: int) -> Categorizer:
    rules = list(db.scalars(select(CategoryRule)))
    return Categorizer(rules, load_user_labels(db, user_id), get_model())
