from app.ingestion.categorize import categorize
from app.models import CategoryRule


def _rule(pattern: str, category: str, subcategory: str | None = None) -> CategoryRule:
    return CategoryRule(merchant_pattern=pattern, category=category, subcategory=subcategory)


def test_matches_substring_pattern():
    rules = [_rule("SWIGGY", "Food & Dining", "Delivery")]
    category, subcategory = categorize("SWIGGY BANGALORE", rules)
    assert category == "Food & Dining"
    assert subcategory == "Delivery"


def test_no_match_returns_none():
    rules = [_rule("SWIGGY", "Food & Dining")]
    category, subcategory = categorize("UNKNOWN MERCHANT", rules)
    assert category is None
    assert subcategory is None


def test_more_specific_pattern_wins_over_general():
    rules = [
        _rule("DMART", "Groceries"),
        _rule("DMART RETAIL", "Groceries", "Supermarket"),
    ]
    category, subcategory = categorize("DMART RETAIL", rules)
    assert category == "Groceries"
    assert subcategory == "Supermarket"


def test_none_merchant_returns_none():
    category, subcategory = categorize(None, [_rule("SWIGGY", "Food")])
    assert category is None
    assert subcategory is None
