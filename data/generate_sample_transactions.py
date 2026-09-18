"""Generates a synthetic multi-month transaction history for local dev/testing.

Real bank exports are hard to get without an account, and a real ML
forecast needs several months of history to fit a trend on - this
produces plausible recurring transactions (with random noise and a
gentle upward drift on dining/shopping) so the pipeline and forecaster
have realistic data to run against.

Usage: python generate_sample_transactions.py > sample_transactions_multi_month.csv
"""
import random
from datetime import date, timedelta

random.seed(42)

MERCHANTS = [
    # (name, category via CategoryRule pattern, typical amount, day-of-month, drift/month)
    ("SWIGGY BANGALORE", -450, 8),
    ("ZOMATO ORDER", -400, 6),
    ("DMART RETAIL", -2200, 4),
    ("NETFLIX SUBSCRIPTION", -649, 0),
    ("UBER TRIP", -320, 10),
    ("AMAZON PURCHASE", -1500, 3),
    ("ELECTRICITY BOARD", -2000, 1),
]

START_MONTH = date(2026, 1, 1)
NUM_MONTHS = 6
SALARY_AMOUNT = 55000


def month_add(d: date, n: int) -> date:
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    return date(year, month, 1)


def random_day(base: date, spread: int = 5) -> date:
    day_offset = random.randint(0, spread)
    return base + timedelta(days=day_offset)


rows = []

for month_index in range(NUM_MONTHS):
    month_start = month_add(START_MONTH, month_index)

    rows.append((month_start.replace(day=4).isoformat(), "SALARY CREDIT", f"{SALARY_AMOUNT:.2f}"))

    for name, base_amount, occurrences in MERCHANTS:
        drift_factor = 1 + (0.03 * month_index)  # ~3%/month upward drift, mimics inflation/lifestyle creep
        for _ in range(max(occurrences, 1)):
            noise = random.uniform(0.85, 1.15)
            amount = round(base_amount * drift_factor * noise, 2)
            txn_date = random_day(month_start, spread=26)
            rows.append((txn_date.isoformat(), name, f"{amount:.2f}"))

rows.sort(key=lambda r: r[0])

import sys

writer_out = sys.stdout
writer_out.write("date,description,amount\n")
for row in rows:
    writer_out.write(",".join(row) + "\n")
