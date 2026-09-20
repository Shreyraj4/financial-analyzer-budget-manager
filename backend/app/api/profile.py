from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.models import User
from app.services.profile import profile_user

router = APIRouter(prefix="/analytics", tags=["analytics"])


class CategoryShare(BaseModel):
    category: str
    share: float


class SpendingProfile(BaseModel):
    cluster: int
    name: str
    window_months: int
    is_full_window: bool
    assignment_margin: float  # 0 = on the border between two profiles, 1 = clearly in one
    your_typical_monthly_spend: float
    profile_typical_monthly_spend: float
    your_top_categories: list[CategoryShare]
    profile_top_categories: list[CategoryShare]


@router.get("/spending-profile", response_model=SpendingProfile | None)
def spending_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict | None:
    """Which spending-behaviour profile the user falls into; null if there isn't enough history."""
    return profile_user(db, current_user.id)
