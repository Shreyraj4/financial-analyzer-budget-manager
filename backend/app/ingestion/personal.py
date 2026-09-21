"""Recognises person-to-person UPI payments and masks digits that could identify an account.

Bank statements are full of transfers to individuals ("UPI/<name>/<bank>/<ref>/<remark>"). Those have no
merchant category, so they are labelled "Personal Transfers" with the person's name as the subcategory,
instead of being sent to the review queue or guessed by the merchant model.

The check is a heuristic: a UPI counterparty that is 1-4 alphabetic words and contains no business
wording. It can be wrong (a shop named after its owner); the user's own label always wins.
"""
import re

from app.preprocessing.text import upi_counterparty

PERSONAL_CATEGORY = "Personal Transfers"

# Words that mean "this is a business or a payment app", not a person.
BUSINESS_WORDS = {
    "LIMITED", "LTD", "PVT", "PRIVATE", "LLP", "INC", "CORP", "CORPORATION", "COMPANY", "CO", "ENTERPRISES",
    "ENTERPRISE", "INDUSTRIES", "TRADERS", "TRADING", "TRADERS", "STORE", "STORES", "MART", "SHOP", "SHOPPE",
    "SERVICES", "SERVICE", "SOLUTIONS", "TECHNOLOGIES", "TECHNOLOGY", "TECH", "SYSTEMS", "FOODS", "FOOD",
    "RESTAURANT", "CAFE", "HOTEL", "BAKERS", "BAKERY", "SWEETS", "KIRANA", "GENERAL", "DAIRY", "MEDICAL",
    "MEDICALS", "PHARMA", "PHARMACY", "CLINIC", "HOSPITAL", "AGENCY", "AGENCIES", "SUPERMARKET", "PETROL",
    "FUELS", "BANK", "PAYMENTS", "PAYMENT", "PAY", "WALLET", "RECHARGE", "TRAVELS", "TOURS", "TELECOM",
    "ELECTRICALS", "ELECTRONICS", "FASHION", "GARMENTS", "TEXTILES", "JEWELLERS", "OPTICALS", "STUDIO",
    "ACADEMY", "SCHOOL", "COLLEGE", "INSTITUTE", "UNIVERSITY", "MUNICIPAL", "BOARD", "GOVT", "DEPARTMENT",
    "PHONEPE", "PAYTM", "GPAY", "GOOGLE", "BHARATPE", "AMAZON", "FLIPKART", "RAZORPAY", "CRED", "MOBIKWIK",
    "SWIGGY", "ZOMATO", "UBER", "OLA", "NETFLIX", "SPOTIFY", "JIO", "AIRTEL", "BLINKIT", "ZEPTO", "DMART",
}

_NAME = re.compile(r"^[A-Za-z][A-Za-z.' ]*$")
_LONG_NUMBER = re.compile(r"\d{9,}")


TITLES = {"MR", "MRS", "MS", "MISS", "SHRI", "SMT", "SRI", "SHRIMATI"}


def person_name(description: str | None, min_words: int = 2) -> str | None:
    """The person's name (Title Case, honorific dropped) if this narration looks like a payment to an
    individual, else None. Single-word names are not accepted by default: they are far more often brands
    ("AJIO", "EMITRA") than people, so they are left for the user to label."""
    name = upi_counterparty(description)
    if not name or not _NAME.match(name):
        return None
    words = [w for w in re.split(r"[ .]+", name.upper()) if w]
    if words and words[0] in TITLES:
        words = words[1:]
        name = " ".join(name.split()[1:])
    if not min_words <= len(words) <= 4:
        return None
    if any(w in BUSINESS_WORDS for w in words) or any(len(w) > 20 for w in words):
        return None
    return " ".join(w.capitalize() for w in name.split())


def mask_long_numbers(text: str) -> str:
    """Account, card and reference numbers (9+ digits) keep only their last four digits: ``XXXXXXXX1234``."""
    return _LONG_NUMBER.sub(lambda m: "X" * (len(m.group()) - 4) + m.group()[-4:], text)
