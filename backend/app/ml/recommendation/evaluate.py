"""Backtesting budgets.

Each budget is judged against what was actually spent. Because every method is
calibrated to the same target coverage, the fair comparison is: at the SAME
breach rate, whose budgets waste the least money (smallest slack / pinball loss)?

  breach rate     share of category-months where actual > budget (target = 1 - coverage)
  slack           budgeted-but-unspent money / total actual spend (waste; lower is better)
  shortfall       overspend beyond budget / total actual spend
  pinball loss    the proper scoring rule for an upper-quantile forecast at level tau;
                  lower is better, and it rewards being both well-calibrated and tight
"""
import numpy as np
import pandas as pd

from app.ml.recommendation.budget import MIN_POOL_PER_CATEGORY, conformal_quantile


def conformal_budgets(
    preds: pd.DataFrame,
    method: str,
    coverage: float,
    first_eval_month: pd.Timestamp,
    hist_col: str = "hist_mean",
    per_category: bool = True,
    min_pool: int = MIN_POOL_PER_CATEGORY,
) -> pd.DataFrame:
    """Budgets for every month >= ``first_eval_month``, each calibrated only on strictly earlier months.

    ``preds`` needs user_id, category, month, actual, ``method`` and ``hist_col`` columns.
    """
    d = preds[preds[hist_col] > 0].copy()
    d["r"] = (d["actual"] - d[method]) / d[hist_col]
    out = []
    for month in sorted(d["month"].unique()):
        if month < first_eval_month:
            continue
        pool, test = d[d["month"] < month], d[d["month"] == month].copy()
        if pool.empty:
            continue
        q_global = conformal_quantile(pool["r"].to_numpy(), coverage)
        if per_category:
            q_cat = {c: conformal_quantile(g["r"].to_numpy(), coverage) for c, g in pool.groupby("category") if len(g) >= min_pool}
            test["q"] = test["category"].map(q_cat).fillna(q_global)
        else:
            test["q"] = q_global
        test["budget"] = np.maximum(test[method] + test["q"] * test[hist_col], 0.0)
        out.append(test)
    return pd.concat(out, ignore_index=True)


def uncalibrated_budgets(preds: pd.DataFrame, method: str, first_eval_month: pd.Timestamp, hist_col: str = "hist_mean") -> pd.DataFrame:
    d = preds[(preds[hist_col] > 0) & (preds["month"] >= first_eval_month)].copy()
    d["budget"] = d[method].clip(lower=0)
    return d


def budget_metrics(rows: pd.DataFrame, coverage: float) -> dict[str, float]:
    a, b = rows["actual"].to_numpy(), rows["budget"].to_numpy()
    diff = a - b
    pinball = np.maximum(coverage * diff, (coverage - 1) * diff)
    return {
        "n": int(len(rows)),
        "breach_rate": round(float((a > b).mean()), 4),
        "slack": round(float(np.maximum(b - a, 0).sum() / a.sum()), 4),
        "shortfall": round(float(np.maximum(a - b, 0).sum() / a.sum()), 4),
        "budget_to_spend": round(float(b.sum() / a.sum()), 4),
        "pinball_loss": round(float(pinball.mean()), 2),
    }


def split_months(months, n_calibration: int, n_validation: int):
    """Ordered origins -> (first validation month, first test month)."""
    ordered = sorted(months)
    return ordered[n_calibration], ordered[n_calibration + n_validation]
