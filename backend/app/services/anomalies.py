import logging
from functools import lru_cache

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.features.transaction_features import build_transaction_features
from app.ml.anomaly.detectors import MODEL_PATH, AnomalyDetector, default_rule_detector
from app.ml.anomaly.explain import explain_flags
from app.models import Transaction

logger = logging.getLogger(__name__)

MIN_DEBITS_FOR_DETECTION = 30  # per-user statistics are meaningless below this


@lru_cache
def get_detector() -> AnomalyDetector:
    """The trained detector, or the rule-based one if none has been trained."""
    if MODEL_PATH.exists():
        try:
            return AnomalyDetector.load(MODEL_PATH)
        except Exception:
            logger.exception("Could not load anomaly detector at %s; using rules", MODEL_PATH)
    return default_rule_detector()


def detect_user_anomalies(db: Session, user_id: int, months: int = 6, limit: int = 50) -> list[dict]:
    """Flagged debit transactions from the user's last ``months`` months of data.

    Features are computed over the user's whole history (a spike is only a
    spike relative to what is normal for them), then only recent flags are returned.
    """
    stmt = select(
        Transaction.id, Transaction.transaction_date, Transaction.description, Transaction.amount, Transaction.category
    ).where(Transaction.user_id == user_id)
    df = pd.DataFrame(db.execute(stmt).all(), columns=["id", "date", "description", "amount", "category"])
    if df.empty:
        return []
    df["amount"] = df["amount"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["user_id"] = user_id
    df = df.reset_index(drop=True)

    feats = build_transaction_features(df)
    debit = feats["is_credit"] == 0
    if debit.sum() < MIN_DEBITS_FOR_DETECTION:
        return []

    result = get_detector().flag(feats[debit], df.loc[debit, "user_id"])
    cutoff = df["date"].max() - pd.DateOffset(months=months)
    flagged = result["is_anomaly"] & (df.loc[debit, "date"] >= cutoff)
    reasons = explain_flags(df.loc[debit], feats[debit], flagged)

    rows = [
        {
            "transaction_id": int(df.at[i, "id"]),
            "transaction_date": df.at[i, "date"].date(),
            "description": df.at[i, "description"],
            "amount": df.at[i, "amount"],
            "category": df.at[i, "category"],
            "score": round(float(result.at[i, "score"]), 4),
            "reasons": reasons[i],
        }
        for i in reasons
    ]
    return sorted(rows, key=lambda r: r["score"], reverse=True)[:limit]
