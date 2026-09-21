"""Turns the ML outputs into a numbered list of short factual sentences.

This is the ONLY place figures are decided. Every number the narrator may
mention is written into a statement here; derived figures (changes, ratios,
savings) are computed here, never by the LLM.
"""
import re
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.verify import extract_numbers
from app.models import Transaction

MAX_TEXT = 60


@dataclass
class Fact:
    id: str
    statement: str

    @property
    def numbers(self) -> list[float]:
        return extract_numbers(self.statement)


@dataclass
class FactSet:
    facts: list[Fact]
    period_start: date
    period_end: date

    def numbers_by_id(self) -> dict[str, list[float]]:
        return {f.id: f.numbers for f in self.facts}

    def by_id(self) -> dict[str, Fact]:
        return {f.id: f for f in self.facts}


def money(x: float) -> str:
    return f"₹{round(x):,}"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def clean_text(text: str | None, limit: int = MAX_TEXT) -> str:
    """User-supplied text (merchant names) is untrusted: keep printable characters only, single line, short."""
    cleaned = "".join(c if c.isprintable() else " " for c in str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().replace('"', "'")
    return cleaned[:limit]


def _month_label(ts: pd.Timestamp) -> str:
    return f"{ts:%b} {ts.year}"


def _budget_statement(item: dict) -> str:
    parts = [
        f"{item['category']}: forecast {money(item['forecast'])}, recommended budget {money(item['recommended_budget'])}",
        f"latest month {money(item['last_month_spend'])}, 3-month average {money(item['avg_3_month_spend'])}",
    ]
    change = item.get("forecast_vs_avg_3_month_pct")
    if change is not None:
        parts.append(f"forecast is {change:+.1f}% versus that average")
    action, amount, existing = item["action"], item["action_amount"], item.get("existing_budget")
    if action == "set_budget":
        parts.append("no budget set yet")
    elif action == "raise":
        parts.append(f"current budget {money(existing)} is below the forecast; suggest raising it to {money(item['recommended_budget'])} (by {money(amount)})")
    elif action == "tighten":
        parts.append(f"current budget {money(existing)} has room; could tighten it to {money(item['recommended_budget'])} (saving {money(amount)})")
    else:
        parts.append(f"current budget {money(existing)} is about right")
    return "; ".join(parts) + "."


def build_facts_from_parts(
    panel: pd.DataFrame,
    profile: dict | None,
    budget: dict | None,
    anomalies: list[dict],
    period_end: date,
) -> FactSet | None:
    """Pure function: all inputs are plain data, so it is easy to test."""
    if panel.empty:
        return None
    facts: list[Fact] = []
    monthly = panel.groupby("month")["spend"].sum().sort_index()
    last = monthly.index[-1]
    label = _month_label(last)
    facts.append(Fact("data.coverage", f"The data covers {len(monthly)} months of spending, up to {label}."))
    facts.append(Fact("spend.latest_total", f"Total spending in the latest month ({label}) was {money(monthly.iloc[-1])}."))
    if len(monthly) >= 2 and monthly.iloc[-2] > 0:
        prev, cur = monthly.iloc[-2], monthly.iloc[-1]
        facts.append(Fact(
            "spend.change_vs_previous",
            f"Spending changed {(cur / prev - 1) * 100:+.1f}% versus the previous month ({money(prev)} to {money(cur)}).",
        ))

    latest = panel[panel["month"] == last].sort_values("spend", ascending=False)
    total = latest["spend"].sum()
    for rank, r in enumerate(latest.head(3).itertuples(index=False), start=1):
        if total > 0 and r.spend > 0:
            facts.append(Fact(f"spend.top_{rank}", f"{r.category} was spending category number {rank} in {label}: {money(r.spend)}, {r.spend / total * 100:.1f}% of the month."))

    if profile:
        yours, typical = profile["your_typical_monthly_spend"], profile["profile_typical_monthly_spend"]
        facts.append(Fact(
            "profile.type",
            f"Spending profile: \"{profile['name']}\", based on the last {profile['window_months']} months; "
            f"your typical monthly spend is {money(yours)} versus {money(typical)} for this profile.",
        ))
        if not profile["is_full_window"]:
            facts.append(Fact("profile.limited", f"The profile uses only {profile['window_months']} months of data, so it is less reliable than a full year."))

    if budget:
        facts.append(Fact(
            "budget.total",
            f"Recommended budgets for {budget['for_month']:%b %Y} total {money(budget['total_recommended'])} against a total forecast of "
            f"{money(budget['total_forecast'])}; each budget is set so spending stays within it in about {budget['target_coverage'] * 100:.0f}% of months.",
        ))
        for item in budget["recommendations"]:
            facts.append(Fact(f"budget.{slug(item['category'])}", _budget_statement(item)))
        if budget["excluded_anomalous_transactions"]:
            facts.append(Fact(
                "budget.excluded_anomalies",
                f"{budget['excluded_anomalous_transactions']} unusual transactions totalling {money(budget['excluded_anomalous_spend'])} "
                "were left out when computing budgets, so one-off spikes do not inflate them.",
            ))

    for a in anomalies:
        reasons = " ".join(r["text"] for r in a["reasons"])
        facts.append(Fact(
            f"anomaly.{a['transaction_id']}",
            f"On {a['transaction_date']:%d %b %Y} a payment of {money(abs(float(a['amount'])))} to \"{clean_text(a['description'])}\" "
            f"({a['category'] or 'uncategorized'}) was flagged as unusual: {clean_text(reasons, 220)}",
        ))

    period_start = last.date().replace(day=1)
    return FactSet(facts=facts, period_start=period_start, period_end=period_end)


def build_facts(db: Session, user_id: int) -> FactSet | None:
    from app.services.anomalies import detect_user_anomalies
    from app.services.forecasting import user_spend_panel
    from app.services.profile import profile_user
    from app.services.recommendations import recommend_budgets

    panel = user_spend_panel(db, user_id)
    if panel.empty:
        return None
    period_end = db.scalar(select(func.max(Transaction.transaction_date)).where(Transaction.user_id == user_id))
    return build_facts_from_parts(
        panel=panel,
        profile=profile_user(db, user_id),
        budget=recommend_budgets(db, user_id),
        anomalies=detect_user_anomalies(db, user_id, months=3, limit=5),
        period_end=period_end,
    )
