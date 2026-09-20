import logging
from functools import lru_cache

import numpy as np
from sqlalchemy.orm import Session

from app.features.monthly_features import monthly_share_vectors
from app.ml.clustering.model import MODEL_PATH, TOTAL_COL, SpendingProfiler
from app.services.forecasting import user_spend_panel

logger = logging.getLogger(__name__)

FULL_WINDOW = 12   # the window the profiler was trained on
MIN_MONTHS = 3     # below this a "behaviour" is just noise


@lru_cache
def get_profiler() -> SpendingProfiler | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return SpendingProfiler.load(MODEL_PATH)
    except Exception:
        logger.exception("Could not load spending profiler at %s", MODEL_PATH)
        return None


def profile_user(db: Session, user_id: int) -> dict | None:
    """The user's spending profile over their most recent months (up to 12).

    Returns None when there is no trained profiler or fewer than MIN_MONTHS of
    history. With under a full year the profile is still returned but flagged
    ``is_full_window = False``: the model was trained on 12-month windows, so
    a shorter view is less reliable.
    """
    profiler = get_profiler()
    panel = user_spend_panel(db, user_id)
    months = panel["month"].nunique() if not panel.empty else 0
    if profiler is None or months < MIN_MONTHS:
        return None

    window = min(FULL_WINDOW, months)
    vectors = monthly_share_vectors(panel, window=window)
    latest = vectors.sort_values("month").tail(1)
    assigned = profiler.assign(latest).iloc[0]
    cluster = int(assigned["cluster"])
    description = profiler.describe()[cluster]

    shares = latest.iloc[0].drop(labels=["user_id", "month", TOTAL_COL]).sort_values(ascending=False)
    return {
        "cluster": cluster,
        "name": description["name"],
        "window_months": int(window),
        "is_full_window": window == FULL_WINDOW,
        "assignment_margin": round(float(assigned["margin"]), 3),
        "your_typical_monthly_spend": round(float(np.expm1(latest.iloc[0][TOTAL_COL])), 2),
        "profile_typical_monthly_spend": description["typical_monthly_spend"],
        "your_top_categories": [{"category": c, "share": round(float(v), 3)} for c, v in shares.head(3).items()],
        "profile_top_categories": description["top_categories"],
    }
