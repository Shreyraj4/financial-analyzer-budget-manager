import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import pandas as pd

from app.ingestion.personal import mask_long_numbers
from app.preprocessing.text import upi_counterparty

DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
    "%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%d-%B-%Y",  # 15 Jun 2026, 15-Jun-2026
    "%d/%m/%y", "%d-%m-%y", "%d %b %y", "%d-%b-%y",
]


@dataclass
class CleanedRow:
    row_number: int
    transaction_date: date | None
    description: str
    merchant: str | None
    amount: Decimal | None
    transaction_type: str | None
    source: str
    valid: bool
    errors: list[str] = field(default_factory=list)


def _parse_date(raw: str) -> date | None:
    raw = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _parse_amount(raw: str) -> Decimal | None:
    # Strip thousands separators and currency symbols banks sometimes include.
    cleaned = re.sub(r"[,₹$\s]", "", raw)
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _normalize_merchant(description: str) -> str:
    counterparty = upi_counterparty(description)
    if counterparty:  # UPI narrations: just the other party, not the bank code / reference / remark
        return re.sub(r"\s+", " ", counterparty.upper())
    normalized = re.sub(r"\s+", " ", description.strip().upper())
    return normalized


def clean_transactions(df: pd.DataFrame, source: str = "csv_upload") -> list[CleanedRow]:
    """Deterministically normalizes each raw CSV row into a typed,
    validated CleanedRow. Positive amount = money in (credit), negative =
    money out (debit) - if a source uses the opposite convention, this is
    the single place to add a sign-flip option, not scattered downstream."""
    rows: list[CleanedRow] = []

    for i, raw in enumerate(df.to_dict(orient="records")):
        row_number = i + 2  # +1 for 0-index, +1 for the header row
        errors: list[str] = []

        raw_date = str(raw.get("date", "")).strip()
        raw_description = mask_long_numbers(str(raw.get("description", "")).strip())
        raw_amount = str(raw.get("amount", "")).strip()

        transaction_date = _parse_date(raw_date) if raw_date else None
        if not raw_date:
            errors.append("Missing date")
        elif transaction_date is None:
            errors.append(f"Unrecognized date format: '{raw_date}'")

        if not raw_description:
            errors.append("Missing description")

        amount = _parse_amount(raw_amount) if raw_amount else None
        if not raw_amount:
            errors.append("Missing amount")
        elif amount is None:
            errors.append(f"Unrecognized amount format: '{raw_amount}'")

        transaction_type = None
        if amount is not None:
            if amount == 0:
                errors.append("Amount cannot be zero")
            else:
                transaction_type = "credit" if amount > 0 else "debit"

        merchant = _normalize_merchant(raw_description) if raw_description else None

        rows.append(
            CleanedRow(
                row_number=row_number,
                transaction_date=transaction_date,
                description=raw_description,
                merchant=merchant,
                amount=amount,
                transaction_type=transaction_type,
                source=source,
                valid=len(errors) == 0,
                errors=errors,
            )
        )

    return rows
