from decimal import Decimal

from pydantic import BaseModel


class CategoryForecastResponse(BaseModel):
    category: str
    predicted_next_month_spend: Decimal
    method: str
    history_months: int


class ForecastResponse(BaseModel):
    forecasts: list[CategoryForecastResponse]
