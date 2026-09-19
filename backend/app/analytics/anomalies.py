from dataclasses import dataclass
from decimal import Decimal
from statistics import mean, pstdev

from app.analytics.monthly_totals import MonthlyTotal, month_key

MIN_HISTORY_MONTHS = 3
Z_THRESHOLD = 2.0
MATERIAL_INCREASE = 1.25
FLAT_BASELINE_JUMP = 1.5


@dataclass
class SpendAnomaly:
    category: str
    year: int
    month: int
    total_spent: Decimal
    baseline_mean: Decimal
    z_score: float


def detect_spend_anomalies(totals: list[MonthlyTotal], z_threshold: float = Z_THRESHOLD) -> list[SpendAnomaly]:
    """Flags months whose category spend is unusually high versus the
    same category's other months (z-score against the rest of the history).

    Categories with fewer than MIN_HISTORY_MONTHS months are skipped: a
    baseline built from 1-2 points is noise. The month under test is left
    out of its own baseline so a large outlier can't hide itself by
    inflating the standard deviation. Only spikes are flagged, not dips.
    """
    by_category: dict[str, list[MonthlyTotal]] = {}
    for t in totals:
        by_category.setdefault(t.category, []).append(t)

    anomalies = []
    for category, rows in by_category.items():
        rows = sorted(rows, key=lambda r: month_key(r.year, r.month))
        if len(rows) < MIN_HISTORY_MONTHS:
            continue
        for i, row in enumerate(rows):
            others = [float(r.total_spent) for j, r in enumerate(rows) if j != i]
            mu = mean(others)
            sigma = pstdev(others)
            spent = float(row.total_spent)
            if sigma == 0:
                # Flat baseline: any material jump is an anomaly.
                if mu > 0 and spent > mu * FLAT_BASELINE_JUMP:
                    z = 999.0
                else:
                    continue
            else:
                z = (spent - mu) / sigma
                # With few months the std dev can be tiny, so also require a
                # materially higher amount, not just a statistical one.
                if z < z_threshold or spent < mu * MATERIAL_INCREASE:
                    continue
            anomalies.append(
                SpendAnomaly(
                    category=category,
                    year=row.year,
                    month=row.month,
                    total_spent=row.total_spent,
                    baseline_mean=Decimal(str(round(mu, 2))),
                    z_score=round(z, 2),
                )
            )
    return sorted(anomalies, key=lambda a: (a.year, a.month, a.category))
