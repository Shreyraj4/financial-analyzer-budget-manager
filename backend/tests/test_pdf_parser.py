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


# --- Indian bank statement layout: title row, Withdrawal/Deposit columns, wrapped narrations ---

from reportlab.platypus import PageBreak, Paragraph  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402

STATEMENT_HEADER = ["#", "Date", "Description", "Chq/Ref. No.", "Withdrawal (Dr.)", "Deposit (Cr.)", "Balance"]
GRID = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)])


def _bank_statement_pdf() -> bytes:
    styles = getSampleStyleSheet()
    page1 = [
        ["Current Account Transactions", "", "", "", "", "", ""],
        STATEMENT_HEADER,
        ["-", "-", "Opening Balance", "-", "-", "-", "30,754.72"],
        ["1", "03 Jun 2026", "UPI/Test Person/YESB/123456789012/Payment\nfrom PhonePe", "UPI-1", "219.33", "", "30,535.39"],
        ["2", "06 Jun 2026", "UPI/Another Person/123456789012/Payment", "UPI-2", "", "1,500.00", "32,035.39"],
    ]
    page2 = [  # continuation: no title, no repeated header
        ["3", "18 Jun 2026", "UPI/ZOMATO LIMITED/HDFC/123456789012/Zomato", "UPI-3", "2,484.00", "", "29,551.39"],
    ]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    doc.build([
        Paragraph("Account Statement", styles["Title"]),
        Paragraph("HOLDER NAME ENTERPRISES Account No. 5949538126", styles["Normal"]),
        Paragraph("Nominee Name Someone Private, 12 Private Street", styles["Normal"]),
        Table(page1, style=GRID),
        PageBreak(),
        Table(page2, style=GRID),
    ])
    return buffer.getvalue()


def test_bank_statement_with_title_row_and_withdrawal_deposit_columns():
    df = parse_pdf(_bank_statement_pdf())
    assert list(df.columns) == ["date", "description", "amount"]
    assert list(df["amount"]) == ["-219.33", "1500.00", "-2484.00"]  # debits negative, credits positive
    assert list(df["date"]) == ["03 Jun 2026", "06 Jun 2026", "18 Jun 2026"]
    assert df.iloc[0]["description"].startswith("UPI/Test Person/YESB/123456789012/Payment from")  # wrapped line joined


def test_opening_balance_and_header_rows_are_not_transactions():
    df = parse_pdf(_bank_statement_pdf())
    assert not df["description"].str.contains("Opening Balance|Description").any()


def test_private_statement_details_never_reach_the_output():
    df = parse_pdf(_bank_statement_pdf())
    blob = " ".join(df.astype(str).values.ravel()) + " ".join(df.columns)
    for secret in ("5949538126", "HOLDER NAME", "Nominee", "Private Street", "30,754", "Balance"):
        assert secret not in blob


def test_single_amount_column_with_dr_cr_suffix():
    df = parse_pdf(_make_statement_pdf(
        [["01/06/2026", "ATM WDL", "1,200.00 Dr"], ["02/06/2026", "SALARY", "55,000.00 Cr"]],
        header=["Txn Date", "Particulars", "Amount"],
    ))
    assert list(df["amount"]) == ["-1200.00", "55000.00"]
