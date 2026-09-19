from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class TransactionPreviewRow(BaseModel):
    row_number: int
    transaction_date: date | None
    description: str
    merchant: str | None
    amount: Decimal | None
    transaction_type: str | None
    category: str | None = None
    subcategory: str | None = None
    category_source: str | None = None
    confidence: float | None = None
    suggested_category: str | None = None
    needs_review: bool = False
    valid: bool
    errors: list[str]


class UploadPreviewResponse(BaseModel):
    filename: str
    total_rows: int
    valid_count: int
    error_count: int
    rows: list[TransactionPreviewRow]


class TransactionIn(BaseModel):
    transaction_date: date
    description: str
    merchant: str | None = None
    amount: Decimal
    transaction_type: str
    source: str = "csv_upload"
    # Send only for rows the user labeled or changed: it is stored as a user label
    # (and remembered for that merchant). Rows without it are categorized server-side.
    category: str | None = Field(default=None, min_length=1, max_length=100)
    subcategory: str | None = Field(default=None, max_length=100)


class ImportRequest(BaseModel):
    transactions: list[TransactionIn]


class ImportResponse(BaseModel):
    submitted_count: int
    imported_count: int
    duplicate_count: int


class TransactionResponse(BaseModel):
    id: int
    transaction_date: date
    description: str
    merchant: str | None
    amount: Decimal
    transaction_type: str
    category: str | None
    subcategory: str | None
    source: str | None
    category_source: str | None
    category_confidence: Decimal | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CategoryUpdateRequest(BaseModel):
    category: str = Field(min_length=1, max_length=100)
    subcategory: str | None = Field(default=None, max_length=100)
    # Also label every other not-yet-user-labeled transaction from the same merchant.
    apply_to_merchant: bool = True


class CategoryUpdateResponse(BaseModel):
    updated_count: int
    transaction: TransactionResponse
