"""Comprehensions, generators, lambda, decorators — a taste of Pythonic expressions.

[1] Compare a sales-data transformation as a loop vs a comprehension,
[2] process 1,000,000 transactions as a list (warehouse) vs a generator (tap)
    and measure the actual memory usage of each.
[3] Pass sort criteria with lambda and [4] sample the decorator stamp.
"""

import random
import sys
import time


# =============================================================
# Decorator for [4]: the 'approval stamp' that times execution
# =============================================================
def stopwatch(func):
    """Take a function and return 'the same function with timing layered on'."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)          # the original function runs untouched
        elapsed = time.perf_counter() - start
        print(f"    (stopwatch: {func.__name__} ran in {elapsed * 1000:.1f}ms)")
        return result
    return wrapper


@stopwatch                       # shorthand for: sum_with_list = stopwatch(sum_with_list)
def sum_with_list(n):
    """Build all n transactions as a list, then sum (the warehouse way)."""
    rows = [i % 1000 * 100 for i in range(n)]   # load all n rows into memory
    return sum(rows), sys.getsizeof(rows)


@stopwatch
def sum_with_generator(n):
    """Stream n transactions through a generator while summing (the tap way)."""
    stream = (i % 1000 * 100 for i in range(n))  # nothing created yet
    return sum(stream), sys.getsizeof(stream)


def transaction_stream(n):
    """yield demo: a generator function that emits transactions one at a time."""
    for i in range(n):
        yield {"id": i, "amount": (i * 37) % 900 * 1000}   # hand over one, pause, wait


def main():
    print("=" * 56)
    print(" Comprehensions, generators, lambda, decorators")
    print("=" * 56)

    random.seed(42)
    # Per-store transaction data (a list of dictionaries)
    rows = [
        {"store": store, "amount": random.randint(3, 80) * 10000}
        for store in ["Downtown", "Riverside", "Airport", "University", "Lakeside", "Midtown", "Harbor", "Hillside"]
    ]

    # ---------------------------------------------------------
    # [1] Loop vs comprehension: same transformation, two notations
    # ---------------------------------------------------------
    print("\n[1] Loop vs comprehension — apply VAT to transactions of 100k KRW or more")

    # The loop way (4 lines)
    loop_result = []
    for r in rows:
        if r["amount"] >= 100000:
            loop_result.append(int(r["amount"] * 1.1))

    # The comprehension way (1 line): "the list of amount x1.1 for each r >= 100k"
    comp_result = [int(r["amount"] * 1.1) for r in rows if r["amount"] >= 100000]

    print(f"  Loop way (4 lines)      : {loop_result}")
    print(f"  Comprehension (1 line)  : {comp_result}")
    print(f"  Are the results equal?  : {loop_result == comp_result}")

    # Dictionary comprehension: build a store -> amount table
    by_store = {r["store"]: r["amount"] for r in rows}
    print(f"  Dictionary comprehension: {by_store}")

    # ---------------------------------------------------------
    # [2] Generators: a feel for memory on large streams
    # ---------------------------------------------------------
    print("\n[2] List (warehouse) vs generator (tap) — summing 1,000,000 transactions")

    N = 1_000_000
    total_l, mem_l = sum_with_list(N)
    print(f"  List way      total {total_l:>13,} / memory {mem_l:>10,} bytes (~{mem_l / 1e6:.1f}MB)")
    total_g, mem_g = sum_with_generator(N)
    print(f"  Generator way total {total_g:>13,} / memory {mem_g:>10,} bytes")
    print(f"  -> Same answer, memory differs by ~{mem_l // mem_g:,}x. The bigger the data, the more you want the tap!")

    # Watching yield in action: draw only the first 3 items
    tap = transaction_stream(N)                  # not a single row created yet
    print("  yield demo (draw only what you need):")
    for _ in range(3):
        print(f"    next() -> {next(tap)}")
    print("    ... the rest were never even created, so their cost is zero")

    # A consumed generator cannot be reused
    small = (x for x in range(3))
    print(f"  1st consumption: {list(small)} / 2nd consumption: {list(small)}  <- empty (one use only)")

    # ---------------------------------------------------------
    # [3] Lambda: handing over the 'criterion' for sorting/max on a sticky note
    # ---------------------------------------------------------
    print("\n[3] Lambda — passing a criterion via the key argument")

    top3 = sorted(rows, key=lambda r: r["amount"], reverse=True)[:3]
    print("  Top 3 stores by revenue:")
    for rank, r in enumerate(top3, start=1):
        print(f"    #{rank} {r['store']} {r['amount']:,} KRW")

    best = max(rows, key=lambda r: r["amount"])
    worst = min(rows, key=lambda r: r["amount"])
    print(f"  Best {best['store']} {best['amount']:,} KRW / worst {worst['store']} {worst['amount']:,} KRW")

    # Aggregation via generator expression: sum(expr for ...)
    big_total = sum(r["amount"] for r in rows if r["amount"] >= 300000)
    print(f"  Total of transactions >= 300k KRW (generator expression): {big_total:,} KRW (no intermediate list)")

    # ---------------------------------------------------------
    # [4] Decorators, wrapped up
    # ---------------------------------------------------------
    print("\n[4] Decorators — the stamp was already pressed back in [2]")
    print("  The @stopwatch above sum_with_list / sum_with_generator is the decorator.")
    print("  Not one line of the function bodies changed, yet 'print the run time' was layered on.")
    print("  The @stamp notation = a function wrapping a function. Used for shared chores like logging and permission checks.")

    print("\n[End] Comprehensions 'only when they read as one sentence', generators 'when data is big',")
    print("      lambda 'a sticky note for passing criteria', decorators 'being able to read them is enough'.")


if __name__ == "__main__":
    main()
