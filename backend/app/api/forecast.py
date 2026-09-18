from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.analytics.monthly_totals import category_monthly_totals
from app.api.deps import get_current_user
from app.database.session import get_db
from app.ml.forecast import forecast_next_month
from app.models import User
from app.schemas.forecast import CategoryForecastResponse, ForecastResponse

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("", response_model=ForecastResponse)
def get_forecast(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ForecastResponse:
    totals = category_monthly_totals(db, current_user.id)
    forecasts = forecast_next_month(totals)
    return ForecastResponse(
        forecasts=[
            CategoryForecastResponse(
                category=f.category,
                predicted_next_month_spend=f.predicted_next_month_spend,
                method=f.method,
                history_months=f.history_months,
            )
            for f in forecasts
        ]
    )
