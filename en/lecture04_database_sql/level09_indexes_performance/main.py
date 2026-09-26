"""
Lecture 04 · Level 09 — Indexes and Query Performance
Builds a 300,000-row big_orders table and measures the same search
before and after adding an index. Reads the DB's game plan (SCAN vs
SEARCH) with EXPLAIN QUERY PLAN, and puts numbers on the cases where an
index is useless (middle-match LIKE) and on the index's write cost.
"""

import pathlib
import random
import sqlite3
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data

N_ROWS = 300_000      # size of the test bench
N_REPEAT = 200        # how many times to repeat the same search (so it's measurable)


def explain(cur, query, params=()):
    """Print the DB's execution plan via EXPLAIN QUERY PLAN."""
    print(f"  SQL> EXPLAIN QUERY PLAN {query}")
    cur.execute("EXPLAIN QUERY PLAN " + query, params)
    for row in cur.fetchall():
        print(f"       plan: {row[3]}")


def measure(cur, query, params_list):
    """Run a query repeatedly and measure the total elapsed time in seconds."""
    t0 = time.perf_counter()
    for p in params_list:
        cur.execute(query, p)
        cur.fetchall()
    return time.perf_counter() - t0


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rng = random.Random(42)   # fixed seed: same experiment data every run

    # ------------------------------------------------------------------
    print(f"[1] Set up the test bench — generating {N_ROWS:,} rows in big_orders")
    cur.execute("CREATE TABLE big_orders ("
                "order_id INTEGER PRIMARY KEY, customer_id INTEGER, "
                "ordered_at TEXT, status TEXT, amount INTEGER)")
    rows = [(i,
             rng.randint(1, 50_000),
             f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
             rng.choice(["completed", "cancelled", "shipping"]),
             rng.randint(1_000, 500_000))
            for i in range(1, N_ROWS + 1)]
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO big_orders VALUES (?,?,?,?,?)", rows)
    con.commit()
    print(f"  Done ({time.perf_counter() - t0:.2f}s). 300,000 orders from 50,000 customers.\n")

    query = "SELECT * FROM big_orders WHERE customer_id = ?"
    targets = [(rng.randint(1, 50_000),) for _ in range(N_REPEAT)]

    # ------------------------------------------------------------------
    print(f"[2] Search without an index — {N_REPEAT} customer-id lookups, measured")
    explain(cur, query, (7,))
    t_before = measure(cur, query, targets)
    print(f"  Elapsed: {t_before:.3f}s  <- every single search skims all {N_ROWS:,} rows (SCAN)\n")

    # ------------------------------------------------------------------
    print("[3] Creating the index — building the back-of-book index")
    print("  SQL> CREATE INDEX idx_big_customer ON big_orders (customer_id)")
    t0 = time.perf_counter()
    cur.execute("CREATE INDEX idx_big_customer ON big_orders (customer_id)")
    con.commit()
    print(f"  Creation time: {time.perf_counter() - t0:.2f}s (a one-time cost)\n")

    # ------------------------------------------------------------------
    print(f"[4] The same {N_REPEAT} searches after the index — the query text is unchanged!")
    explain(cur, query, (7,))
    t_after = measure(cur, query, targets)
    speedup = t_before / t_after if t_after > 0 else float("inf")
    print(f"  Elapsed: {t_after:.4f}s")
    print(f"  -> {t_before:.3f}s -> {t_after:.4f}s, about {speedup:,.0f}x faster\n")

    # ------------------------------------------------------------------
    print("[5] Searches an index can't help — middle-match LIKE is still a SCAN")
    explain(cur, "SELECT * FROM big_orders WHERE status LIKE '%pp%'")
    print("  -> An index is sorted 'from the first letter', so middle matches can't use it.")
    explain(cur, "SELECT * FROM big_orders WHERE customer_id = ? AND status = 'completed'", (7,))
    print("  -> With an indexed column (customer_id) present, it narrows via SEARCH first, then filters.\n")

    # ------------------------------------------------------------------
    print("[6] The index's invoice — writes (INSERT) actually get slower")
    extra = [(N_ROWS + i, rng.randint(1, 50_000), "2025-12-31", "completed", 1000)
             for i in range(1, 10_001)]
    cur.execute("CREATE TABLE plain_copy AS SELECT * FROM big_orders WHERE 0")  # empty copy, no index
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO plain_copy VALUES (?,?,?,?,?)", extra)
    con.commit()
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO big_orders VALUES (?,?,?,?,?)", extra)  # the indexed table
    con.commit()
    t_indexed = time.perf_counter() - t0
    print(f"  10,000-row INSERT — no index: {t_plain:.3f}s / with index: {t_indexed:.3f}s")
    print("  -> An index buys reads and pays with writes. Only on the columns that need it!\n")

    con.close()
    print("[Recap] Slow-query diagnosis in 2 steps: EXPLAIN to spot the SCAN -> index the WHERE column.")
    print("  Index candidates = columns used often in WHERE/JOIN. Index everything and your writes weep.")


if __name__ == "__main__":
    main()
