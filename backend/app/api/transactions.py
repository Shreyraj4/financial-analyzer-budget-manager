from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.ingestion.cleaning import clean_transactions
from app.ingestion.csv_parser import parse_csv
from app.models import Transaction, User
from app.schemas.transaction import (
    ImportRequest,
    ImportResponse,
    TransactionPreviewRow,
    TransactionResponse,
    UploadPreviewResponse,
)
from app.services.transaction_import import import_transactions

router = APIRouter(prefix="/transactions", tags=["transactions"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@router.post("/upload", response_model=UploadPreviewResponse)
async def upload(
    file: UploadFile,
    current_user: User = Depends(get_current_user),
) -> UploadPreviewResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .csv files are supported")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File exceeds 5 MB limit")

    try:
        df = parse_csv(content)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    cleaned_rows = clean_transactions(df)
    preview_rows = [
        TransactionPreviewRow(
            row_number=r.row_number,
            transaction_date=r.transaction_date,
            description=r.description,
            merchant=r.merchant,
            amount=r.amount,
            transaction_type=r.transaction_type,
            valid=r.valid,
            errors=r.errors,
        )
        for r in cleaned_rows
    ]
    valid_count = sum(1 for r in preview_rows if r.valid)

    return UploadPreviewResponse(
        filename=file.filename,
        total_rows=len(preview_rows),
        valid_count=valid_count,
        error_count=len(preview_rows) - valid_count,
        rows=preview_rows,
    )


@router.post("/import", response_model=ImportResponse)
def import_csv(
    body: ImportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ImportResponse:
    result = import_transactions(db, current_user.id, body.transactions)
    return ImportResponse(
        submitted_count=result.submitted_count,
        imported_count=result.imported_count,
        duplicate_count=result.duplicate_count,
    )


@router.get("", response_model=list[TransactionResponse])
def list_transactions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[Transaction]:
    stmt = (
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .order_by(Transaction.transaction_date.desc(), Transaction.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt).all())
