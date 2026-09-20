from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.models import User
from app.services.anomalies import detect_user_anomalies

router = APIRouter(prefix="/analytics", tags=["analytics"])


class TransactionAnomaly(BaseModel):
    transaction_id: int
    transaction_date: date
    description: str
    amount: Decimal
    category: str | None
    score: float
    reasons: list[dict]  # [{code, text, ...}] computed in code, not by the LLM


@router.get("/transaction-anomalies", response_model=list[TransactionAnomaly])
def transaction_anomalies(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    months: int = Query(default=6, ge=1, le=60),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict]:
    """Unusual spending transactions (most anomalous first), each with the reasons it was flagged."""
    return detect_user_anomalies(db, current_user.id, months=months, limit=limit)
