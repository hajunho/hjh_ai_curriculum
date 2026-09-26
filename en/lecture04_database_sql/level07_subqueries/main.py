"""
Lecture 04 · Level 07 — Subqueries and CTEs
Practice using one query's result as another query's ingredient.
Covers scalar subqueries (one value), IN subqueries (a list),
correlated subqueries (recomputed per row), and WITH (CTE) for tidying
multi-step analyses like 'customers who spend above average' into
readable form.
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
    print("A subquery = a sticky note in parentheses: slot the inner answer into the outer query's blank\n")

    # Note 1: see the threshold value with your own eyes first
    cur.execute("SELECT ROUND(AVG(price), 0) FROM products")
    print(f"(Note 1) Average product price = KRW {cur.fetchone()[0]:,.0f} — this value slots into the parentheses below.\n")

    run(cur, 1, "Scalar subquery — products above the average price", """
        SELECT name, price
        FROM products
        WHERE price > (SELECT AVG(price) FROM products)
        ORDER BY price DESC
    """, note="read it as: the parentheses run first and become 'one value'")

    run(cur, 2, "IN subquery — customers who have ever bought a 'Laptop' (partial)", """
        SELECT customer_id, name, city
        FROM customers
        WHERE customer_id IN (
            SELECT o.customer_id
            FROM orders AS o
            WHERE o.order_id IN (
                SELECT oi.order_id
                FROM order_items AS oi
                JOIN products AS p ON p.product_id = oi.product_id
                WHERE p.name = 'Laptop'))
        ORDER BY customer_id
        LIMIT 6
    """, note="the inner answers (order-ID list -> customer-ID list) fill IN's list slots")

    run(cur, "3a", "Setup — average salary per department (for comparison)", """
        SELECT dept, ROUND(AVG(salary), 0) AS avg_salary
        FROM employees
        GROUP BY dept
    """)

    run(cur, "3b", "Correlated subquery — employees paid above their own department's average", """
        SELECT e.name, e.dept, e.salary
        FROM employees AS e
        WHERE e.salary > (SELECT AVG(e2.salary)
                          FROM employees AS e2
                          WHERE e2.dept = e.dept)
        ORDER BY e.dept, e.salary DESC
    """, note="the inner query references the outer row's e.dept -> the threshold changes per row")

    run(cur, 4, "CTE (WITH) — customers who spend above average (completed orders)", """
        WITH customer_totals AS (
            SELECT o.customer_id,
                   SUM(oi.quantity * p.price) AS total
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = 'completed'
            GROUP BY o.customer_id
        )
        SELECT c.name, c.grade, t.total
        FROM customer_totals AS t
        JOIN customers AS c ON c.customer_id = t.customer_id
        WHERE t.total > (SELECT AVG(total) FROM customer_totals)
        ORDER BY t.total DESC
        LIMIT 6
    """, note="name step 1, reuse it 'twice' in step 2 — the power of CTEs")

    run(cur, 5, "Multi-step CTE — build monthly revenue, then find the best month", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = 'completed'
            GROUP BY month
        )
        SELECT month, revenue
        FROM monthly
        WHERE revenue = (SELECT MAX(revenue) FROM monthly)
    """, note="the classic report query: step-by-step CTEs -> one answer at the end")

    con.close()
    print("[Recap] one value -> scalar / a list -> IN / per-row threshold -> correlated")
    print("  When parentheses go two levels deep, promote to a CTE and write a 'readable query'.")


if __name__ == "__main__":
    main()
