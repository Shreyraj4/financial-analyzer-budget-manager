"""Monthly aggregates for forecasting and clustering.

Everything here is derived from debit transactions only ("spend"). Lag and
rolling features are built with ``shift`` so a row for month *t* only ever
sees months before *t* - no target leakage, which is what makes the
time-based evaluation honest.
"""
import numpy as np
import pandas as pd


def monthly_category_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Spend per (user, category, month), with zero-filled gap months.

    Columns: user_id, category, month (month-start Timestamp), spend (positive).
    A category's series runs from the user's first to last month in the data,
    so a month with no purchases is an explicit 0 rather than a missing row
    (which would silently distort lags and rolling means).
    """
    d = df[(df["amount"] < 0) & df["category"].notna()].copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    d["spend"] = -d["amount"].astype(float)
    grouped = d.groupby(["user_id", "category", "month"], as_index=False)["spend"].sum()

    frames = []
    for user_id, user_rows in grouped.groupby("user_id"):
        months = pd.date_range(user_rows["month"].min(), user_rows["month"].max(), freq="MS")
        for category, rows in user_rows.groupby("category"):
            series = rows.set_index("month")["spend"].reindex(months, fill_value=0.0)
            frames.append(
                pd.DataFrame(
                    {"user_id": user_id, "category": category, "month": months, "spend": series.to_numpy()}
                )
            )
    if not frames:
        return pd.DataFrame(columns=["user_id", "category", "month", "spend"])
    return pd.concat(frames, ignore_index=True)


def build_forecast_features(panel: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Adds lag / rolling / calendar features; ``spend`` stays as the target.

    Rows without enough history for the largest lag are dropped, since a
    model can't be trained or scored on features that don't exist.
    """
    p = panel.sort_values(["user_id", "category", "month"]).reset_index(drop=True)
    g = p.groupby(["user_id", "category"])["spend"]

    for k in lags:
        p[f"lag_{k}"] = g.shift(k)
    shifted = g.shift(1)
    grp_keys = [p["user_id"], p["category"]]
    p["roll_mean_3"] = shifted.groupby(grp_keys).transform(lambda s: s.rolling(3, min_periods=1).mean())
    p["roll_std_3"] = shifted.groupby(grp_keys).transform(lambda s: s.rolling(3, min_periods=2).std()).fillna(0.0)

    p["month_of_year"] = p["month"].dt.month
    p["month_sin"] = np.sin(2 * np.pi * p["month_of_year"] / 12)
    p["month_cos"] = np.cos(2 * np.pi * p["month_of_year"] / 12)
    p["t"] = p.groupby(["user_id", "category"]).cumcount()

    lag_cols = [f"lag_{k}" for k in lags]
    return p.dropna(subset=lag_cols + ["roll_mean_3"]).reset_index(drop=True)


FORECAST_FEATURE_COLUMNS = ["roll_mean_3", "roll_std_3", "month_sin", "month_cos", "t"]


def monthly_share_vectors(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (user, month): each category's share of that month's spend,
    plus log total spend. Input to spending-behaviour clustering; shares make
    users comparable regardless of income level, and log total keeps scale
    as a separate signal."""
    wide = panel.pivot_table(index=["user_id", "month"], columns="category", values="spend", aggfunc="sum", fill_value=0.0)
    total = wide.sum(axis=1)
    shares = wide.div(total.where(total > 0, 1.0), axis=0)
    shares["log_total_spend"] = np.log1p(total)
    return shares.reset_index()
