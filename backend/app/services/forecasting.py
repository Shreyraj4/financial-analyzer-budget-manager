import logging
from functools import lru_cache

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.features.monthly_features import monthly_category_panel
from app.ml.forecast import CategoryForecast
from app.ml.forecasting.model import MODEL_PATH, GlobalForecaster
from app.ml.forecasting.predict import predict_next_month
from app.models import Transaction

logger = logging.getLogger(__name__)


@lru_cache
def get_forecaster() -> GlobalForecaster | None:
    """The trained forecaster, or None if not trained (callers then use the simple fallback)."""
    if not MODEL_PATH.exists():
        return None
    try:
        return GlobalForecaster.load(MODEL_PATH)
    except Exception:
        logger.exception("Could not load forecaster at %s", MODEL_PATH)
        return None


def user_spend_panel(db: Session, user_id: int) -> pd.DataFrame:
    stmt = select(Transaction.transaction_date, Transaction.amount, Transaction.category).where(
        Transaction.user_id == user_id, Transaction.transaction_type == "debit", Transaction.category.is_not(None)
    )
    frame = pd.DataFrame(db.execute(stmt).all(), columns=["date", "amount", "category"])
    if frame.empty:
        return monthly_category_panel(frame.assign(user_id=user_id))
    frame["amount"] = frame["amount"].astype(float)
    frame["user_id"] = user_id
    return monthly_category_panel(frame)


def forecast_user(db: Session, user_id: int) -> list[CategoryForecast]:
    return predict_next_month(user_spend_panel(db, user_id), get_forecaster())
