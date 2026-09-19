import numpy as np
import pandas as pd

from app.features.monthly_features import (
    build_forecast_features,
    monthly_category_panel,
    monthly_share_vectors,
)
from app.features.text_features import build_text_vectorizer
from app.features.transaction_features import FEATURE_COLUMNS, build_transaction_features
from app.ml.dataset import load_labeled_transactions, time_split


def _txns(rows):
    return pd.DataFrame(rows, columns=["user_id", "date", "description", "amount", "category"])


def test_transaction_features_shape_and_no_nans():
    df = _txns([
        (1, "2026-01-01", "SWIGGY", -400.0, "Food"),
        (1, "2026-01-05", "SWIGGY", -450.0, "Food"),
        (1, "2026-01-09", "SWIGGY", -420.0, "Food"),
        (1, "2026-01-10", "SALARY", 50000.0, "Income"),
    ])
    feats = build_transaction_features(df)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(feats) == len(df)
    assert not feats.isna().any().any()
    assert feats.loc[3, "is_credit"] == 1


def test_amount_spike_gets_high_robust_z():
    rows = [(1, f"2026-01-{d:02d}", "SWIGGY", -400.0 - d, "Food") for d in range(1, 11)]
    rows.append((1, "2026-01-20", "SWIGGY", -9000.0, "Food"))
    feats = build_transaction_features(_txns(rows))
    assert feats["amount_robust_z"].iloc[-1] > 10
    assert feats["amount_robust_z"].iloc[:-1].abs().max() < 3


def test_duplicate_and_burst_signals():
    df = _txns([
        (1, "2026-03-01", "AMAZON", -999.0, "Shopping"),
        (1, "2026-03-02", "AMAZON", -999.0, "Shopping"),   # same amount next day -> duplicate
        (1, "2026-03-10", "UBER", -200.0, "Transport"),
        (1, "2026-03-10", "UBER", -210.0, "Transport"),
        (1, "2026-03-10", "UBER", -190.0, "Transport"),    # 3 same-day rides -> burst, not duplicate
        (1, "2026-03-20", "DMART", -1500.0, "Groceries"),
    ])
    feats = build_transaction_features(df)
    assert feats["duplicate_within_1d"].tolist() == [1, 1, 0, 0, 0, 0]
    assert feats["same_day_merchant_count"].iloc[2:5].tolist() == [3, 3, 3]
    assert feats["same_day_merchant_count"].iloc[5] == 1


def test_features_are_computed_per_user():
    df = _txns([
        (1, "2026-01-01", "SWIGGY", -400.0, "Food"),
        (1, "2026-01-02", "SWIGGY", -410.0, "Food"),
        (1, "2026-01-03", "SWIGGY", -420.0, "Food"),
        (2, "2026-01-01", "SWIGGY", -4000.0, "Food"),
        (2, "2026-01-02", "SWIGGY", -4100.0, "Food"),
        (2, "2026-01-03", "SWIGGY", -4200.0, "Food"),
    ])
    feats = build_transaction_features(df)
    # 4,000 is normal for user 2, so it must not look like a spike.
    assert feats["amount_robust_z"].abs().max() < 3


def test_monthly_panel_zero_fills_gap_months():
    df = _txns([
        (1, "2026-01-10", "SWIGGY", -100.0, "Food"),
        (1, "2026-03-10", "SWIGGY", -300.0, "Food"),
    ])
    panel = monthly_category_panel(df)
    assert panel["spend"].tolist() == [100.0, 0.0, 300.0]


def test_forecast_features_only_use_past_months():
    months = pd.date_range("2026-01-01", periods=6, freq="MS")
    panel = pd.DataFrame({"user_id": 1, "category": "Food", "month": months,
                          "spend": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    feats = build_forecast_features(panel, lags=(1, 2, 3))
    assert len(feats) == 3  # first 3 months lack a 3-month history
    row = feats.iloc[0]     # month index 3 (spend 40)
    assert (row["lag_1"], row["lag_2"], row["lag_3"]) == (30.0, 20.0, 10.0)
    assert row["roll_mean_3"] == 20.0  # mean of 10,20,30 - excludes the target month
    assert row["spend"] == 40.0


def test_share_vectors_sum_to_one():
    df = _txns([
        (1, "2026-01-05", "SWIGGY", -300.0, "Food"),
        (1, "2026-01-06", "UBER", -100.0, "Transport"),
    ])
    shares = monthly_share_vectors(monthly_category_panel(df))
    assert np.isclose(shares[["Food", "Transport"]].sum(axis=1).iloc[0], 1.0)
    assert shares.loc[0, "Food"] == 0.75


def test_text_vectorizer_generalizes_across_narration_noise():
    vec = build_text_vectorizer()
    docs = ["UPI/111111111/SWIGGY/ybl", "POS 1234XXXXXX5678 SWIGGY PUNE", "ACH D- NETFLIX-123456", "NETFLIX DELHI"]
    x = vec.fit_transform(docs)
    sim = (x @ x.T).toarray()
    assert sim[0, 1] > sim[0, 2]  # both SWIGGY rows closer than SWIGGY vs NETFLIX


def test_time_split_is_chronological_and_disjoint():
    df = load_labeled_transactions()
    train, test = time_split(df, holdout_months=3)
    assert train["date"].max() < test["date"].min()
    assert len(train) + len(test) == len(df)
    assert test["date"].dt.to_period("M").nunique() == 3


def test_features_on_real_dataset_have_no_nans():
    df = load_labeled_transactions()
    feats = build_transaction_features(df)
    assert len(feats) == len(df)
    assert not feats.isna().any().any()
