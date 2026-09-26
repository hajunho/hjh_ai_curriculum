"""Loops (for/while/range/break/continue) — aggregating 100 customers' purchases.

We run the same aggregation "by hand" (writing every line out) and "with a
loop" to see the payoff of automation. Along the way we pick up the
accumulator pattern, loop+conditional combinations, a while simulation,
break/continue, and enumerate — all in one go.
"""

import random


def main():
    print("=" * 56)
    print(" Loops — aggregating 100 customers' purchase data")
    print("=" * 56)

    # Fix the seed for reproducibility: everyone gets the same data
    random.seed(42)

    # This month's purchases for 100 customers (KRW). Some are 0 (dormant).
    purchases = []
    for i in range(100):
        if random.random() < 0.15:          # 15% are dormant customers
            purchases.append(0)
        else:
            purchases.append(random.randint(10, 600) * 1000)  # 10k-600k KRW

    # ---------------------------------------------------------
    # [1] By hand: what if there were no loops?
    # ---------------------------------------------------------
    print("\n[1] Without loops (mimicking just 3 customers)")
    total_by_hand = purchases[0] + purchases[1] + purchases[2]
    print(f"  total = purchases[0] + purchases[1] + purchases[2]  # = {total_by_hand:,} KRW")
    print("  ... for 100 customers you would write 100 terms or 100 lines.")
    print("  The moment customer #101 shows up, the code has to change too.")

    # ---------------------------------------------------------
    # [2] The for loop: 100 customers in 4 lines
    # ---------------------------------------------------------
    print("\n[2] Aggregating with a for loop")

    total = 0                       # initialize the accumulator BEFORE the loop
    best_amount = 0                 # highest purchase
    for amount in purchases:        # take the 100 customers one at a time
        total += amount             # accumulate every lap
        if amount > best_amount:    # update the record
            best_amount = amount

    average = total / len(purchases)
    print(f"  Customers        : {len(purchases)}")
    print(f"  Total purchases  : {total:,} KRW")
    print(f"  Average per head : {average:,.0f} KRW")
    print(f"  Highest purchase : {best_amount:,} KRW")
    print("  -> With 10,000 customers, not a single character of this code changes.")

    # ---------------------------------------------------------
    # [3] Loop + conditional: selecting top customers (using continue)
    # ---------------------------------------------------------
    print("\n[3] Selecting top customers (if/continue combined)")

    VIP_THRESHOLD = 300000          # top-customer bar: 300k KRW and above
    vip_count = 0
    dormant_count = 0
    for amount in purchases:
        if amount == 0:             # dormant customer: skip to the next person
            dormant_count += 1
            continue
        if amount >= VIP_THRESHOLD:
            vip_count += 1

    print(f"  Top customers (>= {VIP_THRESHOLD:,} KRW)    : {vip_count}")
    print(f"  Dormant customers (0 KRW, skipped via continue): {dormant_count}")

    # ---------------------------------------------------------
    # [4] while: budget burn-down simulation (using break)
    # ---------------------------------------------------------
    print("\n[4] while — how many days does a 1,000,000 KRW budget last?")

    budget = 1000000
    day = 0
    while budget > 0:               # repeat while budget remains
        day += 1
        spend = 60000 + random.randint(0, 50) * 1000   # daily spend 60k-110k KRW
        budget -= spend
        if day <= 3 or budget <= 0:                    # print only the first 3 days and the last
            print(f"  Day {day:>2}: spent {spend:>7,} KRW -> remaining {max(budget, 0):>9,} KRW")
        if day >= 60:               # safety net: force-stop past 60 days
            print("  Hit the 60-day cap, exiting with break")
            break

    print(f"  -> The budget ran out on day {day}.")

    # ---------------------------------------------------------
    # [5] range and enumerate: top-5 purchase leaderboard
    # ---------------------------------------------------------
    print("\n[5] Top 5 purchases (ranked with enumerate)")

    top5 = sorted(purchases, reverse=True)[:5]     # sort descending, take first 5
    for rank, amount in enumerate(top5, start=1):  # rank numbers from 1
        print(f"  #{rank}: {amount:,} KRW")

    print("\n  range check: range(5) ->", list(range(5)), "/ range(1, 6) ->", list(range(1, 6)))

    # ---------------------------------------------------------
    # Conclusion
    # ---------------------------------------------------------
    print("\n[End] Line count comparison")
    print("  By hand   : lines grow with the customer count (100 customers = 100+ lines)")
    print("  With a loop: always 4 lines (data grows, code stays the same)")
    print("  'The same procedure for every item in the list' — the key sentence of automation.")


if __name__ == "__main__":
    main()
