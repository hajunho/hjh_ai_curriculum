"""
Lecture 04 · Level 04 — Sorting, Removing Duplicates, and Top N
Practice ORDER BY (lining up), LIMIT/OFFSET (top N and paging), and
DISTINCT (duplicate removal). Build ranking tables like 'top 5 most
expensive products', and see with your own eyes the trap that LIMIT
without ORDER BY is not a top N.
"""

import pathlib
import sqlite3
import sys
import textwrap
import unicodedata

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def disp_width(text):
    """Wide (CJK) characters take 2 cells; compute display width for table alignment."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width):
    return str(text) + " " * (width - disp_width(text))


def run(cur, step, title, sql, note=""):
    """Shared runner: show the SQL statement, then print the result as a table."""
    print(f"[{step}] {title}")
    for line in textwrap.dedent(sql).strip().splitlines():
        print(f"  SQL> {line}")
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    widths = [max(disp_width(c), *(disp_width(r[i]) for r in rows)) if rows else disp_width(c)
              for i, c in enumerate(cols)]
    print("  " + " | ".join(pad(c, w) for c, w in zip(cols, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))
    if note:
        print(f"  -> {note}")
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"Practice DB ready: {db_path.name}")
    print("Execution order: FROM -> WHERE -> SELECT -> ORDER BY -> LIMIT (line up, then cut)\n")

    run(cur, 1, "Top 5 most expensive products — the basic ranking table", """
        SELECT name, category, price
        FROM products
        ORDER BY price DESC
        LIMIT 5
    """, note="DESC = descending (priciest first). Top N is always ORDER BY + LIMIT")

    run(cur, 2, "Top 3 by margin — an expression as the sort key", """
        SELECT name, price, cost, price - cost AS margin
        FROM products
        ORDER BY margin DESC
        LIMIT 3
    """, note="the alias (margin) can be used directly in ORDER BY")

    run(cur, 3, "5 most recent orders — date descending is the 'recent activity' pattern", """
        SELECT order_id, customer_id, ordered_at, status
        FROM orders
        ORDER BY ordered_at DESC
        LIMIT 5
    """)

    run(cur, 4, "Multi-key sort — city alphabetically, then grade within a city", """
        SELECT city, grade, name
        FROM customers
        ORDER BY city ASC, grade ASC
        LIMIT 8
    """, note="comma order is priority order: 1st city, 2nd grade")

    run(cur, 5, "DISTINCT — the 'kinds' of cities customers actually live in", """
        SELECT DISTINCT city
        FROM customers
        ORDER BY city
    """, note="see how 200 customer rows shrink to just the distinct cities")

    run(cur, "5b", "DISTINCT on two columns — kinds of (city, grade) 'combinations'", """
        SELECT DISTINCT city, grade
        FROM customers
        ORDER BY city, grade
        LIMIT 8
    """, note="DISTINCT applies to the whole combination of selected columns")

    run(cur, 6, "Paging — price ranking positions 6-10 (page 2)", """
        SELECT name, price
        FROM products
        ORDER BY price DESC
        LIMIT 5 OFFSET 5
    """, note="OFFSET 5 = skip the first 5 rows, take the next 5")

    run(cur, 7, "Trap — LIMIT 5 with no ORDER BY?", """
        SELECT name, price
        FROM products
        LIMIT 5
    """, note="just 'any 5 rows'. Compare with the top 5 in [1]!")

    con.close()
    print("[Recap] ranking = ORDER BY (DESC) + LIMIT / kinds = DISTINCT")
    print("  The original table's order and contents never change (only the result display).")


if __name__ == "__main__":
    main()
