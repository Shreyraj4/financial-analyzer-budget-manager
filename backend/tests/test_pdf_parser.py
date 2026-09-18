import io

import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from app.ingestion.pdf_parser import parse_pdf


def _make_statement_pdf(rows: list[list[str]], header: list[str] | None = None) -> bytes:
    """Builds a minimal PDF with a real table (mimicking a bank statement)
    so parse_pdf() has genuine table structure to extract, not just text."""
    header = header or ["Date", "Description", "Amount"]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    data = [header] + rows
    table = Table(data)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    doc.build([table])
    return buffer.getvalue()


def test_extracts_rows_from_table_with_standard_headers():
    pdf_bytes = _make_statement_pdf(
        [
            ["2026-01-01", "SWIGGY BANGALORE", "-450.00"],
            ["2026-01-04", "SALARY CREDIT", "55000.00"],
        ]
    )
    df = parse_pdf(pdf_bytes)
    assert list(df.columns) == ["date", "description", "amount"]
    assert len(df) == 2
    assert df.iloc[0]["description"] == "SWIGGY BANGALORE"
    assert df.iloc[1]["amount"] == "55000.00"


def test_recognizes_alias_headers():
    pdf_bytes = _make_statement_pdf(
        [["2026-01-01", "DMART RETAIL", "-2300.50"]],
        header=["Txn Date", "Narration", "Amount (INR)"],
    )
    df = parse_pdf(pdf_bytes)
    assert set(df.columns) == {"date", "description", "amount"}
    assert df.iloc[0]["description"] == "DMART RETAIL"


def test_empty_pdf_bytes_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_pdf(b"")


def test_pdf_without_recognizable_table_raises():
    # A table with unrelated columns should not be mistaken for a transaction table.
    pdf_bytes = _make_statement_pdf(
        [["Q1", "10", "20"]],
        header=["Quarter", "Widgets", "Gadgets"],
    )
    with pytest.raises(ValueError, match="No transaction table found"):
        parse_pdf(pdf_bytes)


def test_garbage_bytes_raise_value_error():
    with pytest.raises(ValueError):
        parse_pdf(b"this is not a pdf at all, just random bytes 12345")
