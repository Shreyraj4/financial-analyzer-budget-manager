from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analytics.anomalies import detect_spend_anomalies
from app.analytics.budget_status import budget_statuses
from app.analytics.monthly_totals import category_monthly_totals
from app.api.deps import get_current_user
from app.database.session import get_db
from app.models import Budget, User
from app.schemas.budget import (
    BudgetCreate,
    BudgetResponse,
    BudgetStatusResponse,
    SpendAnomalyResponse,
)

router = APIRouter(tags=["budgets"])


def _user_budgets(db: Session, user_id: int) -> list[Budget]:
    stmt = select(Budget).where(Budget.user_id == user_id).order_by(Budget.start_date.desc(), Budget.id.desc())
    return list(db.scalars(stmt).all())


@router.post("/budgets", response_model=BudgetResponse, status_code=status.HTTP_201_CREATED)
def create_budget(
    body: BudgetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Budget:
    budget = Budget(user_id=current_user.id, **body.model_dump())
    db.add(budget)
    db.commit()
    db.refresh(budget)
    return budget


@router.get("/budgets", response_model=list[BudgetResponse])
def list_budgets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Budget]:
    return _user_budgets(db, current_user.id)


@router.get("/budgets/status", response_model=list[BudgetStatusResponse])
def get_budget_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[BudgetStatusResponse]:
    return [BudgetStatusResponse(**vars(s)) for s in budget_statuses(db, _user_budgets(db, current_user.id))]


@router.delete("/budgets/{budget_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_budget(
    budget_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    budget = db.get(Budget, budget_id)
    if budget is None or budget.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Budget not found")
    db.delete(budget)
    db.commit()


@router.get("/analytics/anomalies", response_model=list[SpendAnomalyResponse])
def get_anomalies(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SpendAnomalyResponse]:
    totals = category_monthly_totals(db, current_user.id)
    return [SpendAnomalyResponse(**vars(a)) for a in detect_spend_anomalies(totals)]
