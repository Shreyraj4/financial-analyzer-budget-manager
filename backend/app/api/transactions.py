from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy import delete as sa_delete, func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.ingestion.cleaning import clean_transactions
from app.ingestion.csv_parser import parse_csv
from app.ingestion.pdf_parser import parse_pdf
from app.models import CategoryRule, Transaction, User
from app.preprocessing.text import extract_merchant
from app.services.categorization import build_categorizer, get_model
from app.schemas.transaction import (
    DeleteRangeRequest,
    DeleteRangeResponse,
    ImportRequest,
    CategoryUpdateRequest,
    CategoryUpdateResponse,
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
    db: Session = Depends(get_db),
) -> UploadPreviewResponse:
    filename = (file.filename or "").lower()
    if filename.endswith(".csv"):
        file_kind = "csv"
    elif filename.endswith(".pdf"):
        file_kind = "pdf"
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .csv or .pdf files are supported")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File exceeds 5 MB limit")

    try:
        df = parse_csv(content) if file_kind == "csv" else parse_pdf(content)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    cleaned_rows = clean_transactions(df)
    categorizer = build_categorizer(db, current_user.id)
    guesses = categorizer.categorize_rows([(r.description, r.amount) for r in cleaned_rows])
    preview_rows = []
    for r, guess in zip(cleaned_rows, guesses):
        preview_rows.append(
            TransactionPreviewRow(
                row_number=r.row_number,
                transaction_date=r.transaction_date,
                description=r.description,
                merchant=r.merchant,
                amount=r.amount,
                transaction_type=r.transaction_type,
                category=guess.category,
                subcategory=guess.subcategory,
                category_source=guess.source,
                confidence=guess.confidence,
                suggested_category=guess.suggested_category,
                needs_review=guess.needs_review,
                valid=r.valid,
                errors=r.errors,
            )
        )
    valid_count = sum(1 for r in preview_rows if r.valid)

    return UploadPreviewResponse(
        filename=file.filename,
        source="pdf_upload" if file_kind == "pdf" else "csv_upload",
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


@router.delete("", response_model=DeleteRangeResponse)
def delete_transactions(
    body: DeleteRangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeleteRangeResponse:
    """Removes this user's own transactions in a date range (inclusive), e.g. to undo one upload.
    Does not touch other users' data, budgets, or previously generated reports (those keep the
    numbers they reported at the time; generate a new report to reflect the change)."""
    conditions = [
        Transaction.user_id == current_user.id,
        Transaction.transaction_date >= body.start_date,
        Transaction.transaction_date <= body.end_date,
    ]
    if body.source:
        conditions.append(Transaction.source == body.source)
    deleted_count = db.execute(sa_delete(Transaction).where(*conditions)).rowcount
    db.commit()
    return DeleteRangeResponse(deleted_count=deleted_count)


@router.get("/count", response_model=int)
def count_transactions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    start_date: date | None = None,
    end_date: date | None = None,
) -> int:
    """How many of this user's transactions fall in a date range; used to preview a delete before it runs."""
    stmt = select(func.count()).select_from(Transaction).where(Transaction.user_id == current_user.id)
    if start_date:
        stmt = stmt.where(Transaction.transaction_date >= start_date)
    if end_date:
        stmt = stmt.where(Transaction.transaction_date <= end_date)
    return db.scalar(stmt) or 0


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


@router.get("/review", response_model=list[TransactionResponse])
def transactions_needing_review(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[Transaction]:
    """Transactions the system was not confident enough to categorize."""
    stmt = (
        select(Transaction)
        .where(Transaction.user_id == current_user.id, Transaction.category.is_(None))
        .order_by(Transaction.transaction_date.desc(), Transaction.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt).all())


@router.get("/categories", response_model=list[str])
def list_categories(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[str]:
    """Categories to offer in the labeling UI: model classes, rule categories,
    and any custom category this user has already created."""
    names: set[str] = set()
    model = get_model()
    if model is not None:
        names.update(str(c) for c in model.classes_)
    names.update(db.scalars(select(CategoryRule.category).distinct()))
    names.update(
        db.scalars(select(Transaction.category).where(Transaction.user_id == current_user.id, Transaction.category.is_not(None)).distinct())
    )
    return sorted(names)


@router.patch("/{transaction_id}/category", response_model=CategoryUpdateResponse)
def set_transaction_category(
    transaction_id: int,
    body: CategoryUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CategoryUpdateResponse:
    """User labels a transaction. By default the label is applied to every other
    transaction from the same merchant that the user hasn't labeled themselves,
    and future imports from that merchant are labeled automatically."""
    txn = db.get(Transaction, transaction_id)
    if txn is None or txn.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Transaction not found")

    category = body.category.strip()
    if not category:
        raise HTTPException(422, "Category cannot be blank")

    targets = [txn]
    if body.apply_to_merchant:
        merchant = extract_merchant(txn.description)
        if merchant:
            candidates = db.scalars(
                select(Transaction).where(
                    Transaction.user_id == current_user.id,
                    Transaction.id != txn.id,
                    or_(Transaction.category_source.is_(None), Transaction.category_source != "user"),
                )
            )
            targets += [t for t in candidates if extract_merchant(t.description) == merchant]

    for t in targets:
        t.category = category
        t.subcategory = body.subcategory
        t.category_source = "user"
        t.category_confidence = None
    db.commit()
    db.refresh(txn)
    return CategoryUpdateResponse(updated_count=len(targets), transaction=txn)
