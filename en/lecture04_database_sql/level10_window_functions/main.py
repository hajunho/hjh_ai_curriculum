"""
Lecture 04 · Level 10 — Window Functions and Analytical Queries
Practice window functions (OVER), which attach a summary beside each
row without folding the rows away. Runs every reporting staple: the
difference from GROUP BY, purchase sequence per customer with
ROW_NUMBER, the tie handling of the RANK trio, revenue rank within a
department, running totals, and a 3-month moving sum.
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
    print("Watch for it in every result: did the rows fold, or stay?\n")

    run(cur, "1a", "GROUP BY — category average price (rows 'fold' to 5 lines)", """
        SELECT category, ROUND(AVG(price), 0) AS avg_price
        FROM products
        GROUP BY category
    """)

    run(cur, "1b", "Window — the same average attached 'without folding' (10 rows kept)", """
        SELECT name, category, price,
               ROUND(AVG(price) OVER (PARTITION BY category), 0) AS cat_avg,
               price - ROUND(AVG(price) OVER (PARTITION BY category), 0) AS diff
        FROM products
        ORDER BY category, price DESC
    """, note="each row looks out its 'own category window' and posts the average on a post-it")

    run(cur, 2, "ROW_NUMBER — purchase sequence per customer, CTE-wrapped for 'first orders' only", """
        WITH numbered AS (
            SELECT customer_id, order_id, ordered_at,
                   ROW_NUMBER() OVER (PARTITION BY customer_id
                                      ORDER BY ordered_at, order_id) AS nth
            FROM orders
        )
        SELECT c.name, n.order_id, n.ordered_at, n.nth
        FROM numbered AS n
        JOIN customers AS c ON c.customer_id = n.customer_id
        WHERE n.nth = 1
        ORDER BY n.ordered_at
        LIMIT 5
    """, note="'first 1 per group' = ROW_NUMBER + outer filter, the most common working formula")

    run(cur, 3, "The tie-handling trio — product price ranks (ties make the difference visible)", """
        SELECT name, price,
               ROW_NUMBER() OVER (ORDER BY price DESC) AS row_num,
               RANK()       OVER (ORDER BY price DESC) AS rnk,
               DENSE_RANK() OVER (ORDER BY price DESC) AS dense_rnk
        FROM products
        ORDER BY price DESC
    """, note="on equal prices (ties), RANK skips the next position while DENSE_RANK continues")

    run(cur, 4, "Revenue rank within department — per-employee revenue (CTE), then RANK per dept", """
        WITH emp_sales AS (
            SELECT e.employee_id, e.name, e.dept,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN employees AS e   ON e.employee_id = o.employee_id
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p     ON p.product_id = oi.product_id
            WHERE o.status = 'completed'
            GROUP BY e.employee_id, e.name, e.dept
        )
        SELECT dept, name, revenue,
               RANK() OVER (PARTITION BY dept ORDER BY revenue DESC) AS dept_rank
        FROM emp_sales
        ORDER BY dept, dept_rank
        LIMIT 10
    """, note="the 'rank by branch/department' report skeleton: aggregate CTE -> PARTITION BY rank")

    run(cur, 5, "Running total — year-to-date cumulative beside monthly revenue", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = 'completed'
            GROUP BY month
        )
        SELECT month, revenue,
               SUM(revenue) OVER (ORDER BY month) AS cum_revenue
        FROM monthly
        LIMIT 6
    """, note="the ORDER BY makes the window 'start ~ current row', which is what cumulative means")

    run(cur, 6, "3-month moving sum — setting the window size yourself (ROWS BETWEEN)", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = 'completed'
            GROUP BY month
        )
        SELECT month, revenue,
               SUM(revenue) OVER (ORDER BY month
                                  ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS mov3
        FROM monthly
        LIMIT 6
    """, note="previous 2 rows + current row = the last 3 months. The trend-smoothing report trick")

    con.close()
    print("[Recap] Row count shrinks -> GROUP BY; row count kept -> window (OVER).")
    print("  A window's 3 ingredients: PARTITION BY (range) / ORDER BY (order) / ROWS (size).")


if __name__ == "__main__":
    main()
