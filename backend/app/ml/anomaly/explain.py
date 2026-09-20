"""Turns a flagged transaction into human-readable reasons, computed in code.

The LLM narrates these structured reasons; it never invents them. A flag
can have several reasons (e.g. a large duplicate charge). If the model
flagged a row but no rule below explains it, the fallback says so honestly.
"""
import pandas as pd

from app.ml.anomaly.detectors import BURST_3D, BURST_SAME_DAY, LARGE_SHARE, RARE_CATEGORY, Z_THRESHOLD
from app.preprocessing.text import extract_merchant


def explain_flags(txns: pd.DataFrame, feats: pd.DataFrame, flagged: pd.Series) -> dict[int, list[dict]]:
    """``txns``: user_id, description, amount, category (aligned with ``feats``).
    Returns {row index: [ {code, text, ...}, ... ]} for rows where ``flagged`` is True."""
    category = txns["category"].fillna("Unknown")
    abs_amount = txns["amount"].abs().astype(float)
    median = abs_amount.groupby([txns["user_id"], category]).transform("median")
    merchant = txns["description"].map(extract_merchant)

    reasons: dict[int, list[dict]] = {}
    for idx in feats.index[flagged.reindex(feats.index).fillna(False).to_numpy()]:
        f = feats.loc[idx]
        out: list[dict] = []
        if f["amount_robust_z"] >= Z_THRESHOLD and median[idx] > 0:
            ratio = abs_amount[idx] / median[idx]
            out.append({
                "code": "amount_spike",
                "text": f"₹{abs_amount[idx]:,.0f} is {ratio:.1f}× your usual {category[idx]} spend (typical ₹{median[idx]:,.0f}).",
                "ratio_to_typical": round(float(ratio), 2),
            })
        if f["duplicate_within_1d"] >= 1:
            out.append({"code": "duplicate_charge",
                        "text": f"The same amount was charged at {merchant[idx]} more than once within a day."})
        if f["same_day_merchant_count"] >= BURST_SAME_DAY or f["merchant_count_3d"] >= BURST_3D:
            n = int(max(f["same_day_merchant_count"], f["merchant_count_3d"]))
            out.append({"code": "burst", "text": f"{n} charges at {merchant[idx]} within a few days.", "count": n})
        if f["amount_share_of_monthly_spend"] >= LARGE_SHARE and f["category_freq"] < RARE_CATEGORY:
            out.append({
                "code": "rare_large_purchase",
                "text": f"A large purchase ({f['amount_share_of_monthly_spend']:.0%} of your typical monthly spend) "
                        f"in {category[idx]}, a category you rarely use.",
            })
        if not out:
            out.append({"code": "unusual_pattern",
                        "text": "The combination of amount, timing and merchant is unusual compared with your history."})
        reasons[int(idx)] = out
    return reasons
