"""Modules and packages — importing my own module (utils.py) plus a standard library tour.

How to import and use utils.py (the shared-function module in this folder),
and a working tour of four 'specialist departments' Python ships with —
datetime / math / random / collections — with business examples.
"""

import datetime
import math
import random
from collections import Counter, defaultdict

import utils                     # utils.py in this folder — one file, one module


def main():
    print("=" * 56)
    print(" Modules and packages — import and the standard library")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] Importing my module: using utils.py's shared functions
    # ---------------------------------------------------------
    print("\n[1] Using my utils module (module.function = department.person)")

    monthly_sales = [1200000, 980000, 1450000, 730000, 2100000]
    total = sum(monthly_sales)
    vat = utils.calc_vat(total)

    print(f"  Total sales of 5 stores : {utils.format_krw(total)}")
    print(f"  VAT (utils.calc_vat)    : {utils.format_krw(vat)}")
    print(f"  Module constant access  : utils.VAT_RATE = {utils.VAT_RATE}")
    print("  -> utils.py's self-demo did NOT run (thanks to the __name__ distinction).")

    # ---------------------------------------------------------
    # [2] The datetime department: date arithmetic
    # ---------------------------------------------------------
    print("\n[2] datetime — date and time arithmetic")

    base_day = datetime.date(2026, 9, 26)          # fixed reference date (reproducible output)
    deadline = datetime.date(2026, 12, 31)
    d_day = (deadline - base_day).days             # subtracting dates yields a duration

    print(f"  Reference date : {base_day} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][base_day.weekday()]})")
    print(f"  To year-end deadline : D-{d_day}")
    print(f"  Formatted      : {base_day.strftime('%B %d, %Y')}")
    ship_day = utils.add_business_days(base_day, 3)
    print(f"  Delivery 3 business days out (weekends skipped, utils function): {ship_day}")

    # ---------------------------------------------------------
    # [3] The math department: when you need to round up
    # ---------------------------------------------------------
    print("\n[3] math — why vehicle planning needs ceil()")

    people = 17
    van_capacity = 5
    exact = people / van_capacity
    vans = math.ceil(exact)                        # there is no such thing as 3.4 vans. You need 4.
    print(f"  17 people / 5-seat vans = {exact}  ->  math.ceil() = {vans} vans")
    print(f"  math.floor(3.4) = {math.floor(3.4)} / math.sqrt(2) = {math.sqrt(2):.4f}"
          f" / math.pi = {math.pi:.4f}")

    # ---------------------------------------------------------
    # [4] The random department: draws and shuffles (seed fixed for reproducibility)
    # ---------------------------------------------------------
    print("\n[4] random — prize draw and duty roster (seed=42 fixed)")

    random.seed(42)                                # fixed seed: same result on every run
    staff = ["Kim", "Lee", "Park", "Choi", "Jung", "Han"]

    winners = random.sample(staff, 2)              # draw 2 without repeats
    print(f"  2 prize winners : {winners}")

    rotation = staff.copy()
    random.shuffle(rotation)                       # copy first to preserve the original
    print(f"  Duty roster     : {rotation}")
    print(f"  One dice roll   : {random.randint(1, 6)}")

    # ---------------------------------------------------------
    # [5] The collections department: Counter and defaultdict
    # ---------------------------------------------------------
    print("\n[5] collections — the tallying specialists")

    orders = ["Americano", "Latte", "Americano", "Tea", "Latte", "Americano", "Tea", "Americano"]
    counter = Counter(orders)                      # frequency tally in one line
    print(f"  Order frequency Counter : {dict(counter)}")
    print(f"  Most ordered            : {counter.most_common(1)[0][0]} ({counter.most_common(1)[0][1]} orders)")

    branch_sales = [("Downtown", 120), ("Riverside", 80), ("Downtown", 200), ("Airport", 150), ("Riverside", 90)]
    by_branch = defaultdict(int)                   # missing keys start automatically at 0
    for branch, amount in branch_sales:
        by_branch[branch] += amount                # accumulate directly, no get(key, 0)
    print(f"  Sales by store, defaultdict: {dict(by_branch)}")

    print("\n[End] When you need a feature, look in the standard library first,")
    print("      and bundle your team's shared procedures into a module like utils.")


if __name__ == "__main__":
    main()
