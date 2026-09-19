"""Loading the labeled synthetic dataset and time-based splitting."""
from pathlib import Path

import pandas as pd

from app.config import REPO_ROOT

DEFAULT_DATASET = REPO_ROOT / "data" / "synthetic" / "transactions_labeled.csv"


def load_labeled_transactions(path: Path | str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DEFAULT_DATASET, parse_dates=["date"])
    return df.sort_values(["user_id", "date"], kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, holdout_months: int, date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the last ``holdout_months`` calendar months become
    the test set. Never shuffled - models must not see the future."""
    months = df[date_col].dt.to_period("M")
    cutoff = months.max() - holdout_months
    return df[months <= cutoff], df[months > cutoff]
