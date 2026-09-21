import io
import re

import pandas as pd
import pdfplumber

# Header cells are compared after lowercasing and replacing punctuation with spaces, so
# "Withdrawal (Dr.)" -> "withdrawal dr" and "Amount (INR)" -> "amount inr".
COLUMN_ALIASES = {
    "date": {"date", "txn date", "transaction date", "value date", "tran date", "trans date", "posting date"},
    "description": {
        "description", "narration", "narrations", "particulars", "details", "transaction details",
        "transaction remarks", "remarks", "transaction description",
    },
    # One signed (or Dr/Cr-suffixed) amount column...
    "amount": {"amount", "amount inr", "amount in inr", "value", "debit credit", "transaction amount"},
    # ...or separate money-out / money-in columns (the usual layout of Indian bank statements).
    "withdrawal": {
        "withdrawal", "withdrawal dr", "withdrawals", "withdrawal amt", "withdrawal amount", "debit",
        "debit amount", "debits", "dr", "dr amount", "paid out", "money out",
    },
    "deposit": {
        "deposit", "deposit cr", "deposits", "deposit amt", "deposit amount", "credit", "credit amount",
        "credits", "cr", "cr amount", "paid in", "money in",
    },
}

# Only these three columns leave the parser. Everything else in the PDF (account holder, account number,
# address, nominee, running balance, cheque/reference number) is never read into the result.
OUTPUT_COLUMNS = ["date", "description", "amount"]


def _norm(cell: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (cell or "").lower())).strip()


def _match_column(header_cell: str) -> str | None:
    normalized = _norm(header_cell)
    for canonical, aliases in COLUMN_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def _column_map(row: list) -> dict[int, str]:
    """{column index: canonical name} if this row is a transactions header, else {}."""
    mapping: dict[int, str] = {}
    for i, cell in enumerate(row):
        canonical = _match_column(cell or "")
        if canonical and canonical not in mapping.values():
            mapping[i] = canonical
    names = set(mapping.values())
    has_money = "amount" in names or "withdrawal" in names or "deposit" in names
    return mapping if {"date", "description"} <= names and has_money else {}


def _number(cell: str | None) -> str:
    """'1,234.50' -> '1234.50'; '' for empty cells and placeholder dashes."""
    cleaned = re.sub(r"[,₹\s]", "", cell or "")
    return "" if cleaned in {"", "-", "--"} else cleaned


def _signed_amount(values: dict[str, str]) -> str:
    """Debits are negative, credits positive (the convention the rest of the pipeline uses)."""
    withdrawal, deposit = _number(values.get("withdrawal")), _number(values.get("deposit"))
    if withdrawal or deposit:
        return f"-{withdrawal.lstrip('-')}" if withdrawal else deposit
    raw = re.sub(r"[,₹\s]", "", values.get("amount") or "")
    suffix = re.search(r"(dr|cr)\.?$", raw, re.IGNORECASE)  # "1200.00Dr" style
    if suffix:
        number = raw[: suffix.start()]
        return f"-{number.lstrip('-')}" if suffix.group(1).lower() == "dr" else number
    return _number(values.get("amount"))


def _extract_rows_from_tables(pdf: "pdfplumber.PDF") -> list[dict[str, str]]:
    """Extracts transaction rows from PDF tables (bank statements are almost always laid out as one).

    The header row can be anywhere in a table (statements often put a title row above it), and a table
    that continues on the next page without repeating the header reuses the previous page's columns."""
    rows: list[dict[str, str]] = []
    column_map: dict[int, str] = {}
    width = 0

    for page in pdf.pages:
        for table in page.extract_tables():
            body = None
            for index, candidate in enumerate(table):
                found = _column_map(candidate)
                if found:
                    column_map, width, body = found, len(candidate), table[index + 1:]
                    break
            if body is None:
                if not column_map or not table or len(table[0]) != width:
                    continue  # not a transactions table
                body = table

            for raw_row in body:
                values = {name: (raw_row[i] if i < len(raw_row) else "") or "" for i, name in column_map.items()}
                if not re.search(r"\d", values.get("date", "")):
                    continue  # opening-balance line, totals, a repeated header, notes
                rows.append({
                    "date": values["date"],
                    "description": values.get("description", ""),
                    "amount": _signed_amount(values),
                })
    return rows


def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    """Parses a PDF bank statement into the same shape parse_csv() produces:
    a string-typed DataFrame with date/description/amount columns.

    PDFs vary in layout, so this handles the common case of a genuine table whose header row has
    recognizable column names (see COLUMN_ALIASES), including separate Withdrawal/Deposit columns.
    Anything else raises ValueError, same as an unparseable CSV. Statements with no ruled table
    (plain text columns) are not supported."""
    if not file_bytes.strip():
        raise ValueError("The uploaded file is empty")

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            rows = _extract_rows_from_tables(pdf)
    except Exception as exc:
        raise ValueError(f"Could not parse file as PDF: {exc}") from exc

    if not rows:
        raise ValueError(
            "No transaction table found in PDF. Expected a table with "
            "recognizable columns for date, description, and amount (or withdrawal/deposit)."
        )

    df = pd.DataFrame(rows, dtype=str, columns=OUTPUT_COLUMNS).fillna("")
    # Table cells carry stray whitespace/newlines from PDF text extraction (wrapped narrations).
    for col in df.columns:
        df[col] = df[col].apply(lambda v: re.sub(r"\s+", " ", str(v)).strip())
    return df
