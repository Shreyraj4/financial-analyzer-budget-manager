"""Generates a labeled synthetic bank-transaction dataset for the ML pipeline.

Unlike generate_sample_transactions.py (7 merchants, for pipeline smoke
tests), this is built to be a meaningful ML benchmark:

  * ~100 merchants across 11 categories, with messy Indian bank-style
    narrations (UPI/POS/ACH prefixes, ref numbers, city suffixes, truncation)
    so the categorizer has to learn from text, not exact-match names.
  * 6 user personas with different spending mixes -> ground truth for
    clustering.
  * 24 months per user with inflation drift, seasonality (festive shopping,
    summer electricity, December travel) and noise -> forecastable series.
  * Injected, labeled anomalies (amount spikes, duplicate charges, bursts,
    unusual high-value merchants) -> ground truth for anomaly detection.

Outputs (data/synthetic/):
  transactions_labeled.csv   all users, with ground-truth columns
  user_<id>_<persona>.csv    upload-compatible (date,description,amount)

Usage: python generate_synthetic_dataset.py [--seed 42] [--months 24]
"""
import argparse
import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

OUT_DIR = Path(__file__).parent / "synthetic"
START = date(2024, 1, 1)

CITIES = ["BANGALORE", "MUMBAI", "DELHI", "PUNE", "HYDERABAD", "CHENNAI", "GURGAON", "NOIDA"]

# category -> subcategory -> merchants
CATALOG: dict[str, dict[str, list[str]]] = {
    "Food & Dining": {
        "Delivery": ["SWIGGY", "ZOMATO", "EATSURE", "DOMINOS PIZZA", "PIZZA HUT", "BOX8"],
        "Restaurant": ["BARBEQUE NATION", "MCDONALDS", "KFC", "BURGER KING", "HALDIRAMS", "PUNJABI DHABA", "SAGAR RATNA"],
        "Cafe": ["STARBUCKS", "CAFE COFFEE DAY", "THIRD WAVE COFFEE", "CHAAYOS", "BLUE TOKAI"],
    },
    "Groceries": {
        "Supermarket": ["DMART", "BIG BAZAAR", "RELIANCE FRESH", "MORE SUPERMARKET", "SPENCERS"],
        "Online": ["BIGBASKET", "BLINKIT", "ZEPTO", "JIOMART", "SWIGGY INSTAMART"],
        "Local": ["SHARMA KIRANA", "FRESH VEGETABLES MART", "MILK DAIRY BOOTH", "GUPTA PROVISION STORE"],
    },
    "Transport": {
        "Rideshare": ["UBER", "OLA CABS", "RAPIDO", "BLUSMART"],
        "Fuel": ["INDIAN OIL PETROL", "HP PETROL PUMP", "BHARAT PETROLEUM", "SHELL FUEL"],
        "Public": ["DELHI METRO", "BMTC BUS PASS", "IRCTC", "NAMMA METRO", "MUMBAI LOCAL PASS"],
    },
    "Shopping": {
        "Online": ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "MEESHO", "NYKAA"],
        "Electronics": ["CROMA", "RELIANCE DIGITAL", "VIJAY SALES", "APPLE STORE"],
        "Fashion": ["ZARA", "H&M", "WESTSIDE", "PANTALOONS", "LIFESTYLE STORES", "DECATHLON"],
    },
    "Utilities": {
        "Electricity": ["BESCOM", "TATA POWER", "ADANI ELECTRICITY", "BSES RAJDHANI"],
        "Telecom": ["JIO PREPAID", "AIRTEL POSTPAID", "VI RECHARGE", "ACT FIBERNET"],
        "Gas & Water": ["INDANE GAS", "BHARAT GAS", "BWSSB WATER"],
    },
    "Subscriptions": {
        "Streaming": ["NETFLIX", "AMAZON PRIME", "HOTSTAR", "SPOTIFY", "SONYLIV", "YOUTUBE PREMIUM"],
        "Software": ["GOOGLE ONE", "MICROSOFT 365", "ICLOUD", "CHATGPT PLUS"],
    },
    "Health": {
        "Pharmacy": ["APOLLO PHARMACY", "MEDPLUS", "NETMEDS", "PHARMEASY"],
        "Clinic": ["FORTIS HOSPITAL", "MANIPAL HOSPITAL", "LAL PATHLABS", "PRACTO CONSULT"],
        "Fitness": ["CULT FIT", "GOLDS GYM"],
    },
    "Entertainment": {
        "Movies": ["PVR CINEMAS", "INOX", "BOOKMYSHOW"],
        "Gaming": ["STEAM GAMES", "PLAYSTATION STORE", "DREAM11"],
        "Events": ["PAYTM INSIDER", "COMEDY CLUB TICKETS"],
    },
    "Travel": {
        "Flights": ["INDIGO AIRLINES", "AIR INDIA", "MAKEMYTRIP", "CLEARTRIP"],
        "Stay": ["OYO ROOMS", "BOOKING.COM", "TAJ HOTELS", "AIRBNB"],
    },
    "Education": {
        "Courses": ["UDEMY", "COURSERA", "UNACADEMY", "BYJUS"],
        "Fees": ["COLLEGE FEE PAYMENT", "EXAM FEE NTA", "BOOKS PUBLISHER"],
    },
    "Housing": {
        "Rent": ["RENT TRANSFER LANDLORD", "NOBROKER RENT"],
        "Maintenance": ["SOCIETY MAINTENANCE", "URBAN COMPANY", "PLUMBER HOME SERVICE"],
    },
}

# Persona -> per-category monthly expected transaction count and typical
# (median) ticket size in INR. Missing category => that persona rarely spends there.
PERSONAS: dict[str, dict] = {
    "student": {
        "income": ("ALLOWANCE FROM PARENTS", 15000, 1),
        "mix": {"Food & Dining": (12, 220), "Groceries": (4, 300), "Transport": (10, 120),
                "Shopping": (1, 800), "Utilities": (1, 400), "Subscriptions": (2, 200),
                "Entertainment": (3, 350), "Education": (1, 1500), "Health": (1, 300)},
    },
    "young_professional": {
        "income": ("SALARY CREDIT", 55000, 1),
        "mix": {"Food & Dining": (18, 420), "Groceries": (6, 900), "Transport": (14, 200),
                "Shopping": (4, 2200), "Utilities": (3, 1100), "Subscriptions": (4, 300),
                "Entertainment": (4, 700), "Health": (1, 800), "Travel": (0.4, 5000), "Housing": (1, 14000)},
    },
    "family": {
        "income": ("SALARY CREDIT", 110000, 1),
        "mix": {"Food & Dining": (10, 600), "Groceries": (10, 1500), "Transport": (10, 900),
                "Shopping": (4, 2500), "Utilities": (5, 1800), "Subscriptions": (3, 400),
                "Health": (3, 1400), "Education": (1, 8000), "Housing": (1, 20000),
                "Entertainment": (2, 1200), "Travel": (0.4, 15000)},
    },
    "frugal_saver": {
        "income": ("SALARY CREDIT", 60000, 1),
        "mix": {"Food & Dining": (5, 250), "Groceries": (8, 800), "Transport": (10, 100),
                "Shopping": (1, 1200), "Utilities": (3, 900), "Subscriptions": (1, 200),
                "Health": (1, 500), "Housing": (1, 9000)},
    },
    "big_spender": {
        "income": ("SALARY CREDIT", 180000, 1),
        "mix": {"Food & Dining": (20, 1100), "Groceries": (4, 1800), "Transport": (16, 600),
                "Shopping": (4, 4000), "Utilities": (3, 2200), "Subscriptions": (6, 600),
                "Entertainment": (7, 2200), "Travel": (0.8, 18000), "Health": (2, 2500), "Housing": (1, 40000)},
    },
    "freelancer": {
        "income": ("CLIENT PAYMENT NEFT", 70000, 3),  # irregular income
        "mix": {"Food & Dining": (15, 380), "Groceries": (5, 800), "Transport": (8, 250),
                "Shopping": (3, 2000), "Utilities": (3, 1300), "Subscriptions": (7, 550),
                "Education": (1, 2500), "Health": (1, 700), "Travel": (0.6, 8000), "Housing": (1, 13000)},
    },
}

# Category-specific seasonal multipliers by month (1..12).
SEASONALITY = {
    "Shopping": {10: 1.5, 11: 1.6, 12: 1.3, 1: 0.9},
    "Utilities": {4: 1.2, 5: 1.5, 6: 1.4, 7: 1.1},
    "Travel": {5: 1.5, 6: 1.6, 12: 1.9, 10: 1.3},
    "Entertainment": {12: 1.3, 6: 1.2},
    "Health": {7: 1.2, 8: 1.2},
}
ANNUAL_INFLATION = 0.06

# Merchant used by a category is sticky per user (people have favourites).
FAVOURITES = 4


def narration(merchant: str, rng: random.Random, style: int | None = None) -> str:
    """Renders a merchant as a noisy Indian bank-statement narration."""
    style = rng.randint(0, 6) if style is None else style
    ref = "".join(rng.choices("0123456789", k=rng.choice([9, 12])))
    city = rng.choice(CITIES)
    if style == 0:
        return f"UPI/{ref}/{merchant}/{rng.choice(['paytm', 'ybl', 'okhdfc', 'oksbi'])}"
    if style == 1:
        return f"POS {rng.randint(1000, 9999)}XXXXXX{rng.randint(1000, 9999)} {merchant} {city}"
    if style == 2:
        return f"{merchant} {city}"
    if style == 3:
        return f"UPI-{merchant}-{ref}@{rng.choice(['ybl', 'axl', 'icici'])}"
    if style == 4:
        return f"ACH D- {merchant}-{rng.randint(100000, 999999)}"
    if style == 5:
        return f"{merchant[: rng.randint(6, max(6, len(merchant)))]} {ref[:6]}"  # truncated
    return f"{merchant} *{ref[:5]}"


def month_start(i: int) -> date:
    m = START.month - 1 + i
    return date(START.year + m // 12, m % 12 + 1, 1)


def days_in_month(d: date) -> int:
    nxt = month_start((d.year - START.year) * 12 + d.month - START.month + 1)
    return (nxt - d).days


def build_user(user_id: int, persona: str, months: int, rng: random.Random) -> list[dict]:
    spec = PERSONAS[persona]
    # Each user has a few favourite merchants per subcategory.
    favs = {
        (cat, sub): rng.sample(ms, min(FAVOURITES, len(ms)))
        for cat, subs in CATALOG.items() for sub, ms in subs.items()
    }
    user_scale = rng.uniform(0.85, 1.2)
    rows: list[dict] = []

    def add(d, merchant, cat, sub, amount, style=None, anomaly="", credit=False):
        rows.append({
            "user_id": user_id, "persona": persona, "date": d,
            "description": narration(merchant, rng, style),
            "amount": round(amount if credit else -amount, 2),
            "category": cat, "subcategory": sub, "merchant_key": merchant,
            "is_anomaly": int(bool(anomaly)), "anomaly_type": anomaly,
        })

    for i in range(months):
        ms = month_start(i)
        n_days = days_in_month(ms)
        infl = (1 + ANNUAL_INFLATION) ** (i / 12)

        # Income
        inc_name, inc_amt, inc_n = spec["income"]
        for k in range(inc_n):
            amt = inc_amt / inc_n * (rng.uniform(0.5, 1.5) if inc_n > 1 else rng.uniform(0.97, 1.03))
            add(ms + timedelta(days=min(n_days - 1, 0 + k * 9 + rng.randint(0, 3))),
                inc_name, "Income", "Salary" if "SALARY" in inc_name else "Other Income", amt * infl ** 0.5,
                style=2, credit=True)

        # Spending
        for cat, (n_mean, median) in spec["mix"].items():
            season = SEASONALITY.get(cat, {}).get(ms.month, 1.0)
            n = max(0, int(round(rng.gauss(n_mean * season ** 0.5, math.sqrt(max(n_mean, 1)) * 0.6))))
            if n_mean < 1 and rng.random() < n_mean:
                n = 1
            subs = list(CATALOG[cat])
            for _ in range(n):
                sub = rng.choice(subs)
                pool = favs[(cat, sub)] if rng.random() < 0.8 else CATALOG[cat][sub]
                merchant = rng.choice(pool)
                amt = rng.lognormvariate(math.log(median * user_scale * infl * season), 0.45)
                d = ms + timedelta(days=rng.randint(0, n_days - 1))
                add(d, merchant, cat, sub, max(amt, 10))

    # ---- Inject anomalies (~2.5% of spend rows) ----
    spend_idx = [k for k, r in enumerate(rows) if r["amount"] < 0]
    n_anom = max(6, int(0.025 * len(spend_idx)))
    for k in rng.sample(spend_idx, n_anom):
        r = rows[k]
        kind = rng.choice(["amount_spike", "duplicate_charge", "burst", "unusual_high_value"])
        if kind == "amount_spike":
            r["amount"] = round(r["amount"] * rng.uniform(5, 15), 2)
            r["is_anomaly"], r["anomaly_type"] = 1, kind
        elif kind == "duplicate_charge":
            r["is_anomaly"], r["anomaly_type"] = 1, kind  # the original is part of the event too
            dup = dict(r)
            dup["date"] = r["date"] + timedelta(days=rng.randint(0, 1))
            rows.append(dup)
        elif kind == "burst":
            r["is_anomaly"], r["anomaly_type"] = 1, kind
            for _ in range(rng.randint(4, 7)):
                b = dict(r)
                b["amount"] = round(r["amount"] * rng.uniform(0.6, 1.4), 2)
                b["description"] = narration(r["merchant_key"], rng)
                b["is_anomaly"], b["anomaly_type"] = 1, kind
                rows.append(b)
        else:  # unusual_high_value: big one-off at a merchant/category the user rarely uses
            cat = rng.choice(["Travel", "Shopping", "Health", "Entertainment"])
            sub = rng.choice(list(CATALOG[cat]))
            merchant = rng.choice(CATALOG[cat][sub])
            d = r["date"]
            add(d, merchant, cat, sub, rng.uniform(25000, 90000), anomaly=kind)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--months", type=int, default=24)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    OUT_DIR.mkdir(exist_ok=True)
    all_rows: list[dict] = []
    personas = list(PERSONAS)
    # 12 users: each persona twice with independent randomness.
    for uid in range(1, 13):
        persona = personas[(uid - 1) % len(personas)]
        user_rows = build_user(uid, persona, args.months, rng)
        user_rows.sort(key=lambda r: (r["date"], r["description"]))
        all_rows.extend(user_rows)
        with open(OUT_DIR / f"user_{uid:02d}_{persona}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "description", "amount"])
            for r in user_rows:
                w.writerow([r["date"].isoformat(), r["description"], f"{r['amount']:.2f}"])

    fields = ["user_id", "persona", "date", "description", "amount", "category",
              "subcategory", "merchant_key", "is_anomaly", "anomaly_type"]
    with open(OUT_DIR / "transactions_labeled.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({**r, "date": r["date"].isoformat()})

    n_anom = sum(r["is_anomaly"] for r in all_rows)
    merchants = {r["merchant_key"] for r in all_rows}
    print(f"users={len({r['user_id'] for r in all_rows})} rows={len(all_rows)} "
          f"merchants={len(merchants)} anomalies={n_anom} ({n_anom / len(all_rows):.1%})")


if __name__ == "__main__":
    main()

