import io
import re

import pandas as pd
import pdfplumber

REQUIRED_COLUMNS = {"date", "description", "amount"}

# Matches header cells like "Date", "Txn Date", "Description", "Amount (INR)"
COLUMN_ALIASES = {
    "date": {"date", "txn date", "transaction date", "value date"},
    "description": {"description", "narration", "particulars", "details"},
    "amount": {"amount", "amount (inr)", "debit/credit", "value"},
}


def _match_column(header_cell: str) -> str | None:
    normalized = header_cell.strip().lower()
    for canonical, aliases in COLUMN_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def _extract_rows_from_tables(pdf: "pdfplumber.PDF") -> list[dict[str, str]]:
    """Extracts transaction rows from PDF tables (the common case for bank
    statements, which are almost always laid out as a table)."""
    rows: list[dict[str, str]] = []

    for page in pdf.pages:
        for table in page.extract_tables():
            if not table or len(table) < 2:
                continue

            header = table[0]
            column_map = {}
            for i, cell in enumerate(header):
                canonical = _match_column(cell or "")
                if canonical:
                    column_map[i] = canonical

            if not REQUIRED_COLUMNS.issubset(set(column_map.values())):
                continue  # this table isn't a transactions table - skip it

            for raw_row in table[1:]:
                row = {}
                for i, canonical in column_map.items():
                    row[canonical] = raw_row[i] if i < len(raw_row) else ""
                rows.append(row)

    return rows


def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    """Parses a PDF bank statement into the same shape parse_csv() produces:
    a string-typed DataFrame with at least date/description/amount columns.

    PDFs vary wildly in layout, so this only handles the common case of a
    genuine table with a header row containing recognizable column names
    (see COLUMN_ALIASES). Anything else raises ValueError, same as an
    unparseable CSV - the cleaning/categorization pipeline downstream
    doesn't need to know whether the source was a CSV or a PDF.
    """
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
            "recognizable columns for date, description, and amount."
        )

    df = pd.DataFrame(rows, dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    # Table cells sometimes carry stray whitespace/newlines from PDF text extraction.
    for col in df.columns:
        df[col] = df[col].apply(lambda v: re.sub(r"\s+", " ", str(v)).strip())

    return df
