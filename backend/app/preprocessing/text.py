"""Normalizes noisy bank-statement narrations into clean merchant text.

Real narrations look like ``UPI/412345678901/SWIGGY/paytm``,
``POS 4512XXXXXX1234 ZARA MUMBAI`` or ``ACH D- NETFLIX-998877``. Everything
except the merchant name is noise for categorization, so it is stripped
here once and every downstream model sees the same text.
"""
import re

# Payment-rail markers, only removed when they lead the narration
# ("NEFT" at the end of "CLIENT PAYMENT NEFT" is meaningful).
LEADING_PREFIXES = {"UPI", "POS", "ACH", "NEFT", "IMPS", "RTGS", "ATM", "ECS", "NACH", "DR", "CR"}

# UPI virtual-payment-address suffixes; never part of a merchant name.
VPA_HANDLES = {"YBL", "OKHDFC", "OKSBI", "AXL", "ICICI", "OKAXIS", "IBL"}

CITIES = {
    "BANGALORE", "BENGALURU", "MUMBAI", "DELHI", "PUNE", "HYDERABAD",
    "CHENNAI", "GURGAON", "GURUGRAM", "NOIDA", "KOLKATA", "AHMEDABAD",
    "INDORE"
}

_TOKEN_SPLIT = re.compile(r"[^A-Z0-9&.]+")
_VPA = re.compile(r"@\w+")


def _is_reference_token(token: str) -> bool:
    """Reference numbers, masked card numbers and other id-like tokens.
    Short alphanumerics such as 'BOX8' are kept; 3+ digits or an X-mask is not."""
    return sum(c.isdigit() for c in token) >= 3 or "XXX" in token


_RAIL_MARKERS = {"UPI", "DR", "CR"}


def upi_counterparty(description: str | None) -> str | None:
    """The party on the other side of a slash-separated UPI narration, e.g. ``UPI/<name>/<bank>/<ref>/<remark>``
    (also ``UPI/<ref>/<name>/<handle>`` and ``UPI/DR/<ref>/<name>/...``): the first segment that is not the rail
    marker or a reference number. Returns None for anything that is not shaped like that."""
    text = str(description or "").strip()
    if not text.upper().startswith("UPI") or "/" not in text:
        return None
    for segment in text.split("/"):
        segment = _VPA.sub(" ", segment.strip()).strip()
        upper = segment.upper()
        if not segment or upper in _RAIL_MARKERS or _is_reference_token(upper.replace(" ", "")):
            continue
        return re.sub(r"\s+", " ", segment)
    return None


def _tokens(description: str | None) -> list[str]:
    text = str(description or "").upper()
    is_upi = text.startswith("UPI")
    counterparty = upi_counterparty(text)
    if counterparty:  # the bank code, reference and remark around the name are noise
        text = counterparty.upper()
    text = _VPA.sub(" ", text)
    tokens = [t.strip(".") for t in _TOKEN_SPLIT.split(text)]
    tokens = [t for t in tokens if t and not _is_reference_token(t) and t not in VPA_HANDLES]

    while tokens and (tokens[0] in LEADING_PREFIXES or len(tokens[0]) == 1 and tokens[0] != "&"):
        tokens.pop(0)

    # "UPI/<ref>/MERCHANT/paytm": the trailing handle is the app, not the merchant.
    if is_upi and len(tokens) > 1 and tokens[-1] == "PAYTM":
        tokens.pop()
    return tokens


def normalize_description(description: str | None) -> str:
    """Uppercase text with rails, references and VPA handles removed."""
    return " ".join(_tokens(description))


def extract_merchant(description: str | None) -> str:
    """Best-effort stable merchant key: normalized text minus trailing city names,
    so 'ZARA MUMBAI' and 'ZARA DELHI' group together."""
    tokens = _tokens(description)
    while len(tokens) > 1 and tokens[-1] in CITIES:
        tokens.pop()
    return " ".join(tokens)
