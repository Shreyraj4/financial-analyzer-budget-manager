import pandas as pd
import pytest

from app.ml.categorization.evaluate import cross_val_predict_df, score
from app.ml.categorization.model import (
    UNKNOWN,
    MerchantLookupBaseline,
    MLCategorizer,
    SubstringRulesBaseline,
)
from app.ml.dataset import load_labeled_transactions


@pytest.fixture(scope="module")
def sample():
    df = load_labeled_transactions()
    return df.sample(2500, random_state=0).reset_index(drop=True)


def test_ml_categorizer_learns_known_merchants(sample):
    train, test = sample.iloc[:2000], sample.iloc[2000:]
    model = MLCategorizer("logreg", C=10).fit(train)
    assert score(test["category"], model.predict(test))["accuracy"] > 0.95


def test_confidence_is_a_probability_and_aligned_with_prediction(sample):
    model = MLCategorizer("logreg").fit(sample.iloc[:2000])
    out = model.predict_with_confidence(sample.iloc[2000:])
    assert out["confidence"].between(0, 1).all()
    assert (out["category"] == model.predict(sample.iloc[2000:])).all()
    assert list(out.index) == list(sample.iloc[2000:].index)


def test_linearsvc_has_no_confidence(sample):
    model = MLCategorizer("linearsvc").fit(sample.iloc[:500])
    with pytest.raises(ValueError):
        model.predict_with_confidence(sample.iloc[500:510])


def test_save_and_load_roundtrip(sample, tmp_path):
    model = MLCategorizer("logreg").fit(sample.iloc[:1000])
    model.save(tmp_path / "m.joblib")
    loaded = MLCategorizer.load(tmp_path / "m.joblib")
    test = sample.iloc[1000:1050]
    assert (loaded.predict(test) == model.predict(test)).all()


def test_lookup_baseline_knows_seen_merchants_and_gives_up_on_new_ones():
    train = pd.DataFrame({"description": ["SWIGGY PUNE", "SWIGGY DELHI", "UBER"], "amount": [-1, -1, -1],
                          "category": ["Food", "Food", "Transport"]})
    model = MerchantLookupBaseline().fit(train)
    test = pd.DataFrame({"description": ["UPI/123456789/SWIGGY/ybl", "ZARA MUMBAI"], "amount": [-1, -1]})
    assert model.predict(test).tolist() == ["Food", UNKNOWN]


def test_substring_rules_baseline():
    model = SubstringRulesBaseline([("SWIGGY", "Food"), ("SWIGGY INSTAMART", "Groceries")])
    df = pd.DataFrame({"description": ["SWIGGY INSTAMART PUNE", "SWIGGY", "OTHER"]})
    assert model.predict(df).tolist() == ["Groceries", "Food", UNKNOWN]


def test_unseen_merchant_scenario_never_leaks_merchants(sample):
    """Under GroupKFold, the lookup baseline can't know any test merchant, so it
    must score ~0 - proof the split really holds merchants out. (Not exactly
    Unknown everywhere: a truncated narration like 'SWIGGY' from 'SWIGGY
    INSTAMART' can collide with another merchant's key, and is then wrong.)"""
    preds = cross_val_predict_df(MerchantLookupBaseline(), sample, "unseen_merchants")
    assert score(sample["category"], preds)["accuracy"] < 0.05
