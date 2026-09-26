"""
Lecture 04 · Level 05 — Aggregate Functions and GROUP BY
Summarize with COUNT/SUM/AVG/MIN/MAX, build per-city and per-category
subtotals with GROUP BY, and apply 'conditions on subtotals' with
HAVING. The heart of this level is the difference between WHERE
(row filter before grouping) and HAVING (group filter after grouping).
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
    print("Execution order: FROM -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY\n")

    run(cur, 1, "Whole-table summary — aggregation without GROUP BY: 'whole table = one basket'", """
        SELECT COUNT(*) AS product_cnt,
               SUM(price) AS price_sum,
               ROUND(AVG(price), 1) AS price_avg,
               MIN(price) AS price_min,
               MAX(price) AS price_max
        FROM products
    """, note="many rows fold into 'one row' of summary values")

    run(cur, 2, "The three faces of COUNT — *, column, DISTINCT", """
        SELECT COUNT(*) AS all_rows,
               COUNT(manager_id) AS has_manager,
               COUNT(DISTINCT dept) AS dept_kinds
        FROM employees
    """, note="COUNT(column) skips NULLs (the CEO's manager_id is NULL)")

    run(cur, 3, "Customers per city — the GROUP BY basic form", """
        SELECT city, COUNT(*) AS customer_cnt
        FROM customers
        GROUP BY city
        ORDER BY customer_cnt DESC
    """, note="one result row = one basket (city). Check the counts sum to 200")

    run(cur, 4, "Customers per city x grade — two grouping keys", """
        SELECT city, grade, COUNT(*) AS cnt
        FROM customers
        GROUP BY city, grade
        ORDER BY city, grade
        LIMIT 8
    """, note="like dropping two fields into a pivot table")

    run(cur, 5, "Revenue per category — line items x products (a taste of next level's JOIN)", """
        SELECT p.category,
               SUM(oi.quantity * p.price) AS revenue
        FROM order_items AS oi
        JOIN products AS p ON p.product_id = oi.product_id
        GROUP BY p.category
        ORDER BY revenue DESC
    """, note="quantity x unit price computed per row, then SUMmed per category basket")

    run(cur, 6, "HAVING — only cities with 35+ customers (a condition on subtotals)", """
        SELECT city, COUNT(*) AS cnt
        FROM customers
        GROUP BY city
        HAVING COUNT(*) >= 35
        ORDER BY cnt DESC
    """, note="WHERE COUNT(*)>=35 is an error — you can't count before bundling")

    run(cur, 7, "WHERE + HAVING together — monthly orders excl. cancelled, months with 80+ only", """
        SELECT SUBSTR(ordered_at, 1, 7) AS month,
               COUNT(*) AS order_cnt
        FROM orders
        WHERE status <> 'cancelled'
        GROUP BY month
        HAVING COUNT(*) >= 80
        ORDER BY month
    """, note="WHERE sieves rows (single orders), HAVING sieves baskets (months)")

    con.close()
    print("[Recap] subtotal = GROUP BY + aggregate function / subtotal condition = HAVING")
    print("  SELECT may hold only 'basket labels' and 'basket summary values'.")


if __name__ == "__main__":
    main()
