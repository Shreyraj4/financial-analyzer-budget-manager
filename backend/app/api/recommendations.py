from datetime import date

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.models import User
from app.services.recommendations import recommend_budgets

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class CategoryBudgetRecommendation(BaseModel):
    category: str
    forecast: float
    recommended_budget: float
    margin: float
    basis: str  # ml_conformal_category | ml_conformal_pooled | fallback_pooled_margin
    history_months: int
    last_month_spend: float
    avg_3_month_spend: float
    forecast_vs_avg_3_month_pct: float | None
    existing_budget: float | None
    action: str  # set_budget | raise | tighten | keep
    action_amount: float


class BudgetRecommendations(BaseModel):
    for_month: date
    target_coverage: float
    history_variant: str
    excluded_anomalous_transactions: int
    excluded_anomalous_spend: float
    total_forecast: float
    total_recommended: float
    recommendations: list[CategoryBudgetRecommendation]


@router.get("/budgets", response_model=BudgetRecommendations | None)
def budget_recommendations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict | None:
    """Per-category budgets for next month: an ML forecast plus a margin calibrated so that
    the user stays within budget in about ``target_coverage`` of months. Null if not enough data."""
    return recommend_budgets(db, current_user.id)
