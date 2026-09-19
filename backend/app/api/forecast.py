from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.models import User
from app.schemas.forecast import CategoryForecastResponse, ForecastResponse
from app.services.forecasting import forecast_user

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("", response_model=ForecastResponse)
def get_forecast(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ForecastResponse:
    forecasts = forecast_user(db, current_user.id)
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
