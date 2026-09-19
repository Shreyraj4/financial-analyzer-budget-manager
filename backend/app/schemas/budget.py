from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BudgetCreate(BaseModel):
    category: str = Field(min_length=1, max_length=100)
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    period: Literal["weekly", "monthly"]
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def end_after_start(self) -> "BudgetCreate":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class BudgetResponse(BaseModel):
    id: int
    category: str
    amount: Decimal
    period: str
    start_date: date
    end_date: date

    model_config = {"from_attributes": True}


class BudgetStatusResponse(BaseModel):
    budget_id: int
    category: str
    period: str
    start_date: date
    end_date: date
    budget_amount: Decimal
    spent: Decimal
    remaining: Decimal
    percent_used: Decimal
    status: str


class SpendAnomalyResponse(BaseModel):
    category: str
    year: int
    month: int
    total_spent: Decimal
    baseline_mean: Decimal
    z_score: float
