import pytest

from app.ingestion.csv_parser import parse_csv


def test_parses_valid_csv():
    content = b"date,description,amount\n2026-09-01,SWIGGY,-450.00\n"
    df = parse_csv(content)
    assert list(df.columns) == ["date", "description", "amount"]
    assert len(df) == 1


def test_rejects_empty_file():
    with pytest.raises(ValueError, match="empty"):
        parse_csv(b"")


def test_rejects_missing_required_columns():
    content = b"date,notes\n2026-09-01,something\n"
    with pytest.raises(ValueError, match="missing required column"):
        parse_csv(content)


def test_rejects_headers_with_no_data_rows():
    content = b"date,description,amount\n"
    with pytest.raises(ValueError, match="no data rows"):
        parse_csv(content)


def test_column_names_are_normalized_case_and_whitespace():
    content = b" Date , Description , Amount \n2026-09-01,SWIGGY,-450.00\n"
    df = parse_csv(content)
    assert list(df.columns) == ["date", "description", "amount"]
