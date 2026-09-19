"""Per-transaction numeric features (used mainly by anomaly detection).

Input frame needs: user_id, date, description, amount; ``category`` is used
when present. Statistics are computed per user, so a 5,000 purchase is judged
against that user's own habits, not a global average. Output rows are aligned
with the input index.
"""
import numpy as np
import pandas as pd

from app.preprocessing.text import extract_merchant

FEATURE_COLUMNS = [
    "log_amount",
    "is_credit",
    "amount_robust_z",
    "amount_share_of_monthly_spend",
    "merchant_freq",
    "category_freq",
    "same_day_merchant_count",
    "merchant_count_3d",
    "duplicate_within_1d",
    "days_since_last_in_category",
    "day_of_week",
    "day_of_month",
    "is_weekend",
]

MAD_TO_SIGMA = 1.4826
MAX_DAYS_SINCE = 365


def _window_counts(days: np.ndarray, before: int, after: int) -> np.ndarray:
    """For each day value, how many entries in ``days`` lie in [d-before, d+after]
    (including itself)."""
    order = np.sort(days)
    hi = np.searchsorted(order, days + after, side="right")
    lo = np.searchsorted(order, days - before, side="left")
    return hi - lo


def _grouped_window_counts(df: pd.DataFrame, keys: list[str], before: int, after: int) -> pd.Series:
    out = pd.Series(0, index=df.index, dtype=float)
    for _, idx in df.groupby(keys, sort=False).groups.items():
        out.loc[idx] = _window_counts(df.loc[idx, "day_num"].to_numpy(), before, after)
    return out


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["merchant"] = d["description"].map(extract_merchant)
    if "category" not in d.columns:
        d["category"] = "Unknown"
    d["category"] = d["category"].fillna("Unknown")
    d["abs_amount"] = d["amount"].abs().astype(float)
    d["is_credit"] = (d["amount"] > 0).astype(int)
    d["day_num"] = (d["date"] - pd.Timestamp("1970-01-01")).dt.days.astype(int)

    feats = pd.DataFrame(index=d.index)
    feats["log_amount"] = np.log1p(d["abs_amount"])
    feats["is_credit"] = d["is_credit"]

    # Robust z-score against the user's history in the same category and direction.
    grp = d.groupby(["user_id", "category", "is_credit"])["abs_amount"]
    median = grp.transform("median")
    mad = (d["abs_amount"] - median).abs().groupby([d["user_id"], d["category"], d["is_credit"]]).transform("median")
    # Floor the scale so near-constant series (rent, subscriptions) don't yield huge z-scores.
    scale = np.maximum(MAD_TO_SIGMA * mad, np.maximum(0.05 * median, 1.0))
    feats["amount_robust_z"] = ((d["abs_amount"] - median) / scale).clip(-20, 100)

    # Size relative to what this user typically spends in a month.
    debits = d[d["is_credit"] == 0]
    monthly_spend = debits.groupby([debits["user_id"], debits["date"].dt.to_period("M")])["abs_amount"].sum()
    typical_month = monthly_spend.groupby(level=0).median().reindex(d["user_id"]).to_numpy()
    feats["amount_share_of_monthly_spend"] = d["abs_amount"] / np.maximum(typical_month, 1.0)

    n_user = d.groupby("user_id")["user_id"].transform("size")
    feats["merchant_freq"] = d.groupby(["user_id", "merchant"])["user_id"].transform("size") / n_user
    feats["category_freq"] = d.groupby(["user_id", "category"])["user_id"].transform("size") / n_user

    # Burst and duplicate signals.
    feats["same_day_merchant_count"] = _grouped_window_counts(d, ["user_id", "merchant"], 0, 0)
    feats["merchant_count_3d"] = _grouped_window_counts(d, ["user_id", "merchant"], 2, 0)
    same_amount = d.assign(amount_key=d["abs_amount"].round(2))
    feats["duplicate_within_1d"] = (
        _grouped_window_counts(same_amount, ["user_id", "merchant", "amount_key"], 1, 1) - 1
    ).clip(lower=0).clip(upper=1)

    # Recency within the category (first transaction gets the cap).
    ordered = d.sort_values(["user_id", "category", "date"], kind="stable")
    gap = ordered.groupby(["user_id", "category"])["date"].diff().dt.days
    feats["days_since_last_in_category"] = gap.reindex(d.index).fillna(MAX_DAYS_SINCE).clip(upper=MAX_DAYS_SINCE)

    feats["day_of_week"] = d["date"].dt.dayofweek
    feats["day_of_month"] = d["date"].dt.day
    feats["is_weekend"] = (feats["day_of_week"] >= 5).astype(int)

    return feats[FEATURE_COLUMNS].astype(float)
