from datetime import date

import pytest

from app.agent.facts import anomaly_fact
from app.ingestion.cleaning import clean_transactions
from app.ingestion.personal import PERSONAL_CATEGORY, mask_long_numbers, person_name
from app.preprocessing.text import extract_merchant, upi_counterparty
from app.services.categorization import Categorizer

import pandas as pd


@pytest.mark.parametrize(
    "narration, name",
    [
        ("UPI/Asha Verma/YESB/123456789012/Payment from PhonePe", "Asha Verma"),
        ("UPI/Mr RAVI KUMAR SIN/123456789012/Payment from Ph", "Ravi Kumar Sin"),
        ("UPI/DR/123456789012/MEERA K NAIR/YESB/meera@ybl", "Meera K Nair"),
    ],
)
def test_person_to_person_payments_are_recognized(narration, name):
    assert person_name(narration) == name


@pytest.mark.parametrize(
    "narration",
    [
        "UPI/ZOMATO LIMITED/HDFC/123456789012/Zomato Payme",   # business wording
        "UPI/PhonePe/YESB/123456789012/Payment",               # payment app
        "UPI/AJIO/123456789012/PayviaRazorpay",                # single word: usually a brand
        "UPI/15 AD BAKERY/123456789012/Payment",               # digits in the name
        "UPI/Kumar Sweets/123456789012/Payment",               # business word
        "SWIGGY BANGALORE",                                    # not a UPI narration at all
        "SALARY CREDIT",
        None,
    ],
)
def test_businesses_and_brands_are_not_people(narration):
    assert person_name(narration) is None


def test_upi_counterparty_handles_both_common_layouts():
    assert upi_counterparty("UPI/412345678901/SWIGGY/paytm") == "SWIGGY"
    assert upi_counterparty("UPI/Asha Verma/YESB/123456789012/x") == "Asha Verma"
    assert extract_merchant("UPI/Asha Verma/YESB/123456789012/Payment from PhonePe") == "ASHA VERMA"


def test_long_numbers_keep_only_last_four_digits():
    assert mask_long_numbers("UPI/Asha/123456789012/pay 4567") == "UPI/Asha/XXXXXXXX9012/pay 4567"


def test_personal_transfer_is_labelled_with_the_persons_name_before_the_model():
    class ConfidentButWrong:
        def predict_with_confidence(self, frame):
            return pd.DataFrame({"category": ["Shopping"] * len(frame), "confidence": [1.0] * len(frame)})

    cat = Categorizer([], {}, ConfidentButWrong())
    person, brand = cat.categorize_rows([
        ("UPI/Asha Verma/YESB/123456789012/Payment from PhonePe", -100),
        ("UPI/Blinkit/123456789012/order", -300),
    ])
    assert (person.category, person.subcategory, person.source) == (PERSONAL_CATEGORY, "Asha Verma", "person")
    assert not person.needs_review
    assert brand.category == "Shopping"  # not a person: falls through to the model


def test_user_label_beats_the_person_rule():
    labels = {"ASHA VERMA": ("Rent", None)}
    (result,) = Categorizer([], labels, None).categorize_rows([("UPI/Asha Verma/YESB/123456789012/x", -100)])
    assert (result.category, result.source) == ("Rent", "user")


def test_cleaning_parses_indian_dates_masks_numbers_and_uses_the_counterparty_as_merchant():
    df = pd.DataFrame({
        "date": ["15 Jun 2026", "16-Jun-2026", "17/06/26"],
        "description": ["UPI/Asha Verma/YESB/123456789012/Payment", "ATM CASH", "NEFT 98765432101234 RENT"],
        "amount": ["-219.33", "-500", "1,000.00"],
    })
    rows = clean_transactions(df)
    assert [r.transaction_date for r in rows] == [date(2026, 6, 15), date(2026, 6, 16), date(2026, 6, 17)]
    assert rows[0].merchant == "ASHA VERMA"
    assert "123456789012" not in rows[0].description and "9012" in rows[0].description
    assert "98765432101234" not in rows[2].description


def test_llm_facts_never_contain_a_persons_name():
    fact = anomaly_fact({
        "transaction_id": 7, "transaction_date": date(2026, 6, 18), "amount": -2484,
        "description": "UPI/Asha Verma/YESB/XXXXXXXX9012/Payment from PhonePe", "category": PERSONAL_CATEGORY,
        "reasons": [{"text": "3 charges at ASHA VERMA within a few days."}],
    })
    assert "Asha" not in fact.statement and "ASHA" not in fact.statement
    assert "a person" in fact.statement and "₹2,484" in fact.statement


def test_business_anomalies_still_name_the_merchant():
    fact = anomaly_fact({
        "transaction_id": 8, "transaction_date": date(2026, 6, 18), "amount": -547,
        "description": "SWIGGY BANGALORE", "category": "Food & Dining", "reasons": [{"text": "Unusual amount."}],
    })
    assert '"SWIGGY BANGALORE"' in fact.statement
