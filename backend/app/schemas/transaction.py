from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel


class TransactionPreviewRow(BaseModel):
    row_number: int
    transaction_date: date | None
    description: str
    merchant: str | None
    amount: Decimal | None
    transaction_type: str | None
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
    created_at: datetime

    model_config = {"from_attributes": True}
