from decimal import Decimal

import pandas as pd

from app.ingestion.cleaning import clean_transactions


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_valid_row_is_cleaned_correctly():
    df = _df([{"date": "2026-09-01", "description": " swiggy bangalore ", "amount": "-450.00"}])
    rows = clean_transactions(df)
    assert len(rows) == 1
    row = rows[0]
    assert row.valid is True
    assert row.errors == []
    assert row.transaction_date.isoformat() == "2026-09-01"
    assert row.merchant == "SWIGGY BANGALORE"
    assert row.amount == Decimal("-450.00")
    assert row.transaction_type == "debit"


def test_positive_amount_is_credit():
    df = _df([{"date": "2026-09-01", "description": "SALARY", "amount": "55000.00"}])
    row = clean_transactions(df)[0]
    assert row.transaction_type == "credit"
    assert row.valid is True


def test_zero_amount_is_invalid():
    df = _df([{"date": "2026-09-01", "description": "ZERO TXN", "amount": "0"}])
    row = clean_transactions(df)[0]
    assert row.valid is False
    assert "zero" in row.errors[0].lower()


def test_missing_fields_produce_errors():
    df = _df([{"date": "", "description": "", "amount": ""}])
    row = clean_transactions(df)[0]
    assert row.valid is False
    assert any("date" in e.lower() for e in row.errors)
    assert any("description" in e.lower() for e in row.errors)
    assert any("amount" in e.lower() for e in row.errors)


def test_unrecognized_date_format_is_flagged():
    df = _df([{"date": "not-a-date", "description": "X", "amount": "-10"}])
    row = clean_transactions(df)[0]
    assert row.valid is False
    assert "date format" in row.errors[0].lower()


def test_amount_with_currency_symbol_and_commas_is_parsed():
    df = _df([{"date": "2026-09-01", "description": "X", "amount": "₹1,200.50"}])
    row = clean_transactions(df)[0]
    assert row.valid is True
    assert row.amount == Decimal("1200.50")


def test_alternate_date_formats_are_accepted():
    df = _df([
        {"date": "01-09-2026", "description": "X", "amount": "-1"},
        {"date": "09/01/2026", "description": "Y", "amount": "-1"},
    ])
    rows = clean_transactions(df)
    assert all(r.valid for r in rows)


def test_row_numbers_account_for_header_row():
    df = _df([
        {"date": "2026-09-01", "description": "X", "amount": "-1"},
        {"date": "2026-09-02", "description": "Y", "amount": "-1"},
    ])
    rows = clean_transactions(df)
    assert rows[0].row_number == 2
    assert rows[1].row_number == 3
