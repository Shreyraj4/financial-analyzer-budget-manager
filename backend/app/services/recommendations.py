import logging
from functools import lru_cache

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.recommendation.budget import MODEL_PATH, BudgetModel
from app.models import Budget
from app.services.anomalies import flagged_transaction_ids
from app.services.forecasting import user_spend_panel

logger = logging.getLogger(__name__)

TIGHTEN_RATIO = 1.25   # an existing budget this far above the recommendation has room to be tightened
MIN_MONTHS = 1


@lru_cache
def get_budget_model() -> BudgetModel | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return BudgetModel.load(MODEL_PATH)
    except Exception:
        logger.exception("Could not load budget model at %s", MODEL_PATH)
        return None


def _existing_monthly_budgets(db: Session, user_id: int) -> dict[str, float]:
    rows = db.scalars(
        select(Budget).where(Budget.user_id == user_id, Budget.period == "monthly").order_by(Budget.end_date, Budget.id)
    )
    return {b.category: float(b.amount) for b in rows}  # later (most recent) budgets overwrite earlier ones


def advise(recommended: float, forecast: float, existing: float | None) -> dict:
    """Deterministic advice comparing a user's current budget with the recommendation."""
    if existing is None:
        return {"action": "set_budget", "action_amount": round(recommended, 2)}
    if existing < forecast:
        return {"action": "raise", "action_amount": round(recommended - existing, 2)}
    if existing > recommended * TIGHTEN_RATIO:
        return {"action": "tighten", "action_amount": round(existing - recommended, 2)}
    return {"action": "keep", "action_amount": 0.0}


def recommend_budgets(db: Session, user_id: int) -> dict | None:
    model = get_budget_model()
    if model is None:
        return None

    flagged = flagged_transaction_ids(db, user_id) if model.use_clean_history else set()
    raw_panel = user_spend_panel(db, user_id)
    if raw_panel.empty or raw_panel["month"].nunique() < MIN_MONTHS:
        return None
    panel = user_spend_panel(db, user_id, exclude_ids=flagged) if flagged else raw_panel
    recs = model.recommend(panel)
    if recs.empty:
        return None

    last_month = raw_panel["month"].max()
    by_cat = raw_panel.set_index(["category", "month"])["spend"]
    existing = _existing_monthly_budgets(db, user_id)

    items = []
    for r in recs.itertuples(index=False):
        recent = [float(by_cat.get((r.category, last_month - pd.DateOffset(months=i)), 0.0)) for i in range(3)]
        avg3 = sum(recent) / 3
        items.append({
            "category": r.category,
            "forecast": round(r.forecast, 2),
            "recommended_budget": round(r.recommended_budget, 2),
            "margin": round(r.margin, 2),
            "basis": r.basis,
            "history_months": int(r.history_months),
            "last_month_spend": round(recent[0], 2),
            "avg_3_month_spend": round(avg3, 2),
            "forecast_vs_avg_3_month_pct": round((r.forecast / avg3 - 1) * 100, 1) if avg3 > 0 else None,
            "existing_budget": existing.get(r.category),
            **advise(r.recommended_budget, r.forecast, existing.get(r.category)),
        })
    items.sort(key=lambda i: i["recommended_budget"], reverse=True)

    return {
        "for_month": (last_month + pd.DateOffset(months=1)).date(),
        "target_coverage": model.coverage,
        "history_variant": model.name,
        "excluded_anomalous_transactions": len(flagged),
        "excluded_anomalous_spend": round(float(raw_panel["spend"].sum() - panel["spend"].sum()), 2),
        "total_forecast": round(sum(i["forecast"] for i in items), 2),
        "total_recommended": round(sum(i["recommended_budget"] for i in items), 2),
        "recommendations": items,
    }
