import pytest

from app.ml.dataset import load_labeled_transactions
from app.preprocessing.text import extract_merchant, normalize_description


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("UPI/412345678901/SWIGGY/paytm", "SWIGGY"),
        ("POS 4512XXXXXX1234 ZARA MUMBAI", "ZARA MUMBAI"),
        ("ACH D- NETFLIX-998877", "NETFLIX"),
        ("UPI-DELHI METRO-869598675@YBL", "DELHI METRO"),
        ("UPI/123456789/PAYTM INSIDER/paytm", "PAYTM INSIDER"),
        ("H&M PUNE", "H&M PUNE"),
        ("BOX8 *40753", "BOX8"),
        ("CLIENT PAYMENT NEFT", "CLIENT PAYMENT NEFT"),
    ],
)
def test_normalize_description(raw, expected):
    assert normalize_description(raw) == expected


def test_extract_merchant_drops_trailing_city_only():
    assert extract_merchant("POS 4512XXXXXX1234 ZARA MUMBAI") == "ZARA"
    assert extract_merchant("DELHI METRO BANGALORE") == "DELHI METRO"
    assert extract_merchant("DELHI") == "DELHI"  # never reduce to empty


def test_handles_missing_and_empty_input():
    assert normalize_description(None) == ""
    assert extract_merchant("") == ""


def test_merchant_extraction_recovers_most_generated_merchants():
    df = load_labeled_transactions()
    recovered = df["description"].map(extract_merchant) == df["merchant_key"]
    # Truncated narrations ('STARBU 123456') legitimately can't be recovered.
    assert recovered.mean() > 0.8
