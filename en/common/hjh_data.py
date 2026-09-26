"""
hjh_data.py — synthetic data generator built just for this curriculum

This module *creates from scratch* every practice dataset used across the
hjh_ai_curriculum lectures. Nothing is downloaded from external sources, so
there are no copyright or licensing concerns whatsoever, and everything works
exactly the same in a classroom with no internet connection.

Every function takes a seed and always reproduces the same result.
That way the numbers printed in the lecture notes never drift away
from what you actually see when you run the code.

Author: Junho Ha (hajunho) · MIT License
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# numpy is used when available and we fall back to pure Python otherwise.
# (lecture01–02 come before numpy is taught, so they must not depend on it.)
try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# 0. Shared utilities
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    """An independent random generator. Never touches the global random state."""
    return random.Random(seed)


def to_csv(rows: list[dict], path: str) -> str:
    """Save a list of dictionaries as a CSV file and return the path."""
    import csv
    if not rows:
        raise ValueError("Cannot save an empty dataset.")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def head(rows: list[dict], n: int = 5) -> None:
    """Pretty-print the first few rows as a table (works without pandas)."""
    if not rows:
        print("(no data)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r[c])) for r in rows[:n])) for c in cols}
    line = " | ".join(str(c).ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows[:n]:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print(f"... {len(rows)} rows total")


# ---------------------------------------------------------------------------
# 1. Sales data — lecture03, 05, 07
# ---------------------------------------------------------------------------

STORES = ["Downtown", "Riverside", "Airport", "University", "Lakeside"]
CATEGORIES = ["Coffee", "Bakery", "Sandwich", "Dessert", "Beverage"]


def sales_table(n_days: int = 365, seed: int = 42) -> list[dict]:
    """
    Daily sales data for a fictional cafe chain.

    The following patterns were planted on purpose (you will uncover them
    one by one during the lectures):
      - weekend sales are higher than weekdays      -> groupby / day-of-week effect
      - beverage sales spike in summer              -> seasonality / time series
      - missing values and negative outliers mixed in -> data cleaning
      - ad spend correlates with revenue            -> regression / correlation vs causation
    """
    rng = _rng(seed)
    rows: list[dict] = []
    for day in range(n_days):
        weekday = day % 7                      # 0=Mon ... 6=Sun
        is_weekend = weekday >= 5
        season = math.sin(2 * math.pi * day / 365.0)   # -1 (winter) ~ +1 (summer)

        for store in STORES:
            store_power = 1.0 + 0.15 * STORES.index(store)
            ad_cost = round(rng.uniform(50_000, 400_000), -3)

            for cat in CATEGORIES:
                base = 300_000 * store_power
                if cat == "Beverage":
                    base *= 1.0 + 0.45 * season          # spikes in summer
                if cat == "Coffee":
                    base *= 1.3
                if is_weekend:
                    base *= 1.25
                noise = rng.gauss(1.0, 0.18)
                revenue = base * noise + ad_cost * 0.35

                # Deliberate contamination for data-cleaning practice:
                # 1% missing, 0.5% negative
                u = rng.random()
                if u < 0.010:
                    revenue_out = None
                elif u < 0.015:
                    revenue_out = -abs(round(revenue))
                else:
                    revenue_out = round(revenue)

                rows.append({
                    "date": f"2025-{day // 31 + 1:02d}-{day % 31 + 1:02d}",
                    "day_index": day,
                    "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday],
                    "store": store,
                    "category": cat,
                    "ad_cost": int(ad_cost),
                    "revenue": revenue_out,
                })
    return rows


# ---------------------------------------------------------------------------
# 2. Customer churn data — lecture06, 07
# ---------------------------------------------------------------------------

def churn_table(n: int = 2000, seed: int = 7) -> list[dict]:
    """
    Subscription-service customer churn data (for binary classification).

    Real signals:  fewer usage days, more support calls, more plan changes -> higher churn odds
    Fake signal:   customer_id means nothing at all (planted for leakage-detection practice)
    The churn rate is about 18%, which makes it a good imbalanced-data exercise.
    """
    rng = _rng(seed)
    rows = []
    for i in range(n):
        tenure = rng.randint(1, 60)                       # months since sign-up
        monthly_fee = rng.choice([9900, 14900, 19900, 29900])
        usage_days = max(0, min(30, int(rng.gauss(18, 7))))
        support_calls = max(0, int(rng.expovariate(1 / 1.3)))
        plan_changes = max(0, int(rng.expovariate(1 / 0.6)))
        is_auto_pay = rng.random() < 0.7

        # The log-odds are designed by hand -> a dataset with a known ground truth
        z = (-1.2
             - 0.05 * usage_days
             + 0.45 * support_calls
             + 0.40 * plan_changes
             - 0.020 * tenure
             + 0.00004 * monthly_fee
             - (0.7 if is_auto_pay else 0.0))
        p = 1 / (1 + math.exp(-z))
        churned = 1 if rng.random() < p else 0

        rows.append({
            "customer_id": f"C{100000 + i}",
            "tenure_months": tenure,
            "monthly_fee": monthly_fee,
            "usage_days_30d": usage_days,
            "support_calls_30d": support_calls,
            "plan_changes": plan_changes,
            "auto_pay": int(is_auto_pay),
            "churned": churned,
        })
    return rows


# ---------------------------------------------------------------------------
# 3. Fraud transaction data — lecture07
# ---------------------------------------------------------------------------

def fraud_table(n: int = 5000, seed: int = 11) -> list[dict]:
    """Card-fraud transaction data. Fraud rate is about 1.5% — severely imbalanced on purpose."""
    rng = _rng(seed)
    rows = []
    for i in range(n):
        is_fraud = rng.random() < 0.015
        if is_fraud:
            amount = rng.lognormvariate(12.5, 1.1)     # abnormally large amounts
            hour = rng.choice([0, 1, 2, 3, 4, 23])     # small hours of the night
            foreign = rng.random() < 0.55
            n_recent = rng.randint(5, 20)              # rapid-fire payments in a short window
        else:
            amount = rng.lognormvariate(10.2, 0.9)
            hour = int(max(0, min(23, rng.gauss(14, 4))))
            foreign = rng.random() < 0.05
            n_recent = rng.randint(0, 4)
        rows.append({
            "tx_id": f"T{i:06d}",
            "amount": round(amount),
            "hour": hour,
            "is_foreign": int(foreign),
            "tx_count_1h": n_recent,
            "is_fraud": int(is_fraud),
        })
    return rows


# ---------------------------------------------------------------------------
# 4. English text corpus — lecture10, 11, 12
# ---------------------------------------------------------------------------

REVIEW_POSITIVE = [
    "Delivery was incredibly fast, loved it", "Excellent quality for the price",
    "I would absolutely buy this again", "Everything was packaged with great care",
    "Much sturdier than I expected, really pleased", "The staff were friendly and helpful",
    "Looks exactly like the photos, very happy", "The store had a lovely, relaxed atmosphere",
    "Performance blew past my expectations", "Setup was quick and painless",
]
REVIEW_NEGATIVE = [
    "Delivery took a whole week to arrive", "Quality is far below what the price suggests",
    "I doubt I will ever order this twice", "The package arrived torn open",
    "Flimsier than expected and broke almost immediately", "I sent an inquiry and never heard back",
    "The color is nothing like the photos", "The store was cramped and noisy",
    "Performance does not match the description at all", "The manual was so confusing I got lost",
]


def review_corpus(n: int = 600, seed: int = 3) -> list[dict]:
    """English product/service reviews for sentiment classification. label 1=positive, 0=negative."""
    rng = _rng(seed)
    fillers = ["", " That said, a couple of small things could be better.",
               " I will definitely use them again.",
               " Hope this helps someone deciding.", " My star rating is completely honest."]
    rows = []
    for i in range(n):
        label = i % 2
        pool = REVIEW_POSITIVE if label == 1 else REVIEW_NEGATIVE
        text = rng.choice(pool) + rng.choice(fillers)
        rows.append({"id": i, "text": text, "label": label})
    rng.shuffle(rows)
    return rows


SAMPLE_DOCS = {
    "policy_vacation.txt": (
        "Article 1. Employees are granted 15 days of annual leave upon completing "
        "one full year of service from their hire date. "
        "After three years of service, one extra day accrues every two years, "
        "up to a maximum of 25 days. "
        "Annual leave must be requested through the approval system at least "
        "3 business days in advance. "
        "Unused leave is paid out as an allowance after the fiscal year closes."
    ),
    "policy_expenses.txt": (
        "Article 2. Business-trip expenses are reimbursed only when the trip "
        "was approved in advance. "
        "The daily allowance is KRW 30,000 for domestic trips and KRW 80,000 "
        "for overseas trips. "
        "Receipts must be submitted within 7 days after the trip ends; "
        "expenses submitted past the deadline are, as a rule, not reimbursed."
    ),
    "policy_remote_work.txt": (
        "Article 3. Remote work is allowed up to twice a week and requires "
        "the team lead's approval. "
        "Even on remote-work days, employees must be reachable during core hours, "
        "from 10 a.m. to 4 p.m. "
        "Taking company assets off-site must be reported to the Information "
        "Security team in advance."
    ),
    "manual_install.txt": (
        "Always disconnect the power supply before installing this product. "
        "Keep at least 10 centimeters of clearance from the wall to ensure ventilation. "
        "Initial setup finishes within 5 minutes: switch the power on and follow "
        "the on-screen instructions. "
        "If you hear unusual noise after installation, stop using the product "
        "immediately and contact customer support."
    ),
    "manual_warranty.txt": (
        "The product warranty lasts 2 years from the date of purchase. "
        "Damage caused by customer mishandling, water immersion, or dropping "
        "is excluded from warranty coverage. "
        "A purchase receipt or order number is required for warranty repairs. "
        "Consumable parts are replaced at cost regardless of the warranty period."
    ),
}


def tiny_corpus(seed: int = 5) -> str:
    """
    English text for pre-training a mini language model (lecture12).
    The grammar patterns repeat on purpose, so even a tiny model
    can pick up a learning signal.
    """
    rng = _rng(seed)
    subjects = ["a student", "an office worker", "a chef", "a developer", "a teacher"]
    objects = ["a report", "a pot of stew", "a program", "a letter", "a proposal"]
    verbs = ["made", "fixed", "checked", "organized", "prepared"]
    times = ["Yesterday", "Today", "This morning", "This evening", "Over the weekend"]
    lines = []
    for _ in range(2000):
        lines.append(f"{rng.choice(times)} {rng.choice(subjects)} "
                     f"{rng.choice(verbs)} {rng.choice(objects)}.")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# 5. Image data — lecture09
# ---------------------------------------------------------------------------

def shape_images(n: int = 800, size: int = 16, seed: int = 13):
    """
    Grayscale images for shape classification. 0=square, 1=circle, 2=triangle.
    We draw them ourselves so you can practice CNNs without downloading MNIST.
    Requires numpy. Returns: (X, y) — X is a (n, size, size) float32 array.
    """
    if _np is None:
        raise ImportError("shape_images() requires numpy. pip install numpy")
    rng = _np.random.default_rng(seed)
    X = _np.zeros((n, size, size), dtype="float32")
    y = _np.zeros(n, dtype="int64")
    for i in range(n):
        label = i % 3
        y[i] = label
        img = _np.zeros((size, size), dtype="float32")
        cy, cx = rng.integers(5, size - 5, size=2)
        r = int(rng.integers(3, 5))
        yy, xx = _np.mgrid[0:size, 0:size]
        if label == 0:                                    # square
            img[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 1.0
        elif label == 1:                                  # circle
            img[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 1.0
        else:                                             # triangle
            mask = (yy >= cy - r) & (yy <= cy + r) & (_np.abs(xx - cx) <= (yy - cy + r) / 2)
            img[mask] = 1.0
        img += rng.normal(0, 0.08, img.shape).astype("float32")   # noise
        X[i] = _np.clip(img, 0.0, 1.0)
    idx = rng.permutation(n)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# 6. Relational DB schema — lecture04
# ---------------------------------------------------------------------------

def build_sqlite(path: str = "hjh_shop.db", seed: int = 21) -> str:
    """
    Build a SQLite database for SQL practice.
    Tables: customers, products, orders, order_items, employees
    Uses only the Python standard library, so no database server to install.
    """
    import os
    import sqlite3

    if os.path.exists(path):
        os.remove(path)
    rng = _rng(seed)
    con = sqlite3.connect(path)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            city TEXT, grade TEXT, joined_at TEXT);
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            category TEXT, price INTEGER, cost INTEGER);
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            dept TEXT, salary INTEGER, manager_id INTEGER);
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY, customer_id INTEGER,
            employee_id INTEGER, ordered_at TEXT, status TEXT);
        CREATE TABLE order_items (
            order_id INTEGER, product_id INTEGER, quantity INTEGER,
            PRIMARY KEY (order_id, product_id));
    """)

    cities = ["New York", "Chicago", "Boston", "Austin", "Seattle", "Denver"]
    grades = ["VIP", "GOLD", "SILVER", "BASIC"]
    surnames = ["Smith", "Johnson", "Williams", "Brown", "Jones",
                "Miller", "Davis", "Wilson", "Taylor", "Anderson"]
    givens = ["James", "Emma", "Liam", "Olivia", "Noah",
              "Ava", "Ethan", "Mia", "Lucas", "Sophia"]

    # Note: both name pools have 10 entries, so drawing them in this order
    # keeps the random stream identical to the original.
    customers = [(i, rng.choice(givens) + " " + rng.choice(surnames), rng.choice(cities),
                  rng.choice(grades), f"202{rng.randint(0, 5)}-{rng.randint(1, 12):02d}-15")
                 for i in range(1, 201)]
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    pnames = [("Laptop", "Electronics"), ("Wireless Mouse", "Electronics"),
              ("Mechanical Keyboard", "Electronics"),
              ("Monitor", "Electronics"), ("Coffee Beans", "Food"), ("Chocolate", "Food"),
              ("Tumbler", "Household"), ("Notebook", "Stationery"),
              ("Fountain Pen", "Stationery"), ("Office Chair", "Furniture")]
    products = []
    for pid, (nm, cat) in enumerate(pnames, start=1):
        price = rng.choice([4900, 12900, 29000, 89000, 350000, 1200000])
        products.append((pid, nm, cat, price, int(price * rng.uniform(0.45, 0.75))))
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    depts = ["Sales", "Marketing", "Engineering", "CS"]
    employees = [(1, "Emily Carter", "Sales", 9000, None)]
    for eid in range(2, 21):
        employees.append((eid, rng.choice(givens) + " " + rng.choice(surnames),
                          rng.choice(depts), rng.randint(3200, 8500),
                          1 if eid <= 5 else rng.randint(2, 5)))
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)

    orders, items = [], []
    for oid in range(1, 1001):
        orders.append((oid, rng.randint(1, 200), rng.randint(1, 20),
                       f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                       rng.choices(["completed", "cancelled", "shipping"], weights=[8, 1, 2])[0]))
        for pid in rng.sample(range(1, 11), rng.randint(1, 3)):
            items.append((oid, pid, rng.randint(1, 5)))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    cur.executemany("INSERT INTO order_items VALUES (?,?,?)", items)

    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    print("=== hjh_data self-check ===\n")
    print("[1] Sales data"); head(sales_table(n_days=10), 3)
    print("\n[2] Churn data"); head(churn_table(200), 3)
    print("\n[3] Fraud data"); head(fraud_table(500), 3)
    print("\n[4] Review corpus"); head(review_corpus(20), 3)
    print(f"\n[5] {len(SAMPLE_DOCS)} documents, mini corpus of {len(tiny_corpus()):,} characters")
    print(f"\n[6] SQLite built -> {build_sqlite('/tmp/hjh_check.db')}")
    print("\nAll generators working correctly.")
