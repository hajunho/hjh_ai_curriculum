"""
Lecture 04 · Level 06 — JOIN: Connecting Multiple Tables
Attaches customer and product info to an order book that holds only
numbers (foreign keys) to analyze 'who bought what'. Demonstrates INNER
JOIN, multi-table joins, JOIN+GROUP BY, finding customers with no orders
via LEFT JOIN, and the fan-out trap where a join inflates aggregates.
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
    print("Map: customers ──< orders >── employees / orders ──< order_items >── products")

    # For the LEFT JOIN exercise, add 3 brand-new customers with no orders yet.
    cur.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?)",
        [(201, "Nora Bennett", "Seattle", "BASIC", "2025-12-01"),
         (202, "Eli Harper", "Boston", "BASIC", "2025-12-02"),
         (203, "Autumn Reed", "Denver", "BASIC", "2025-12-03")])
    con.commit()
    print("(Setup) Added 3 new customers (201-203) with no order history.\n")

    run(cur, 1, "Two-table join — the order book's numbers turn into names", """
        SELECT o.order_id, c.name AS customer, c.city, o.ordered_at, o.status
        FROM orders AS o
        JOIN customers AS c ON c.customer_id = o.customer_id
        ORDER BY o.order_id
        LIMIT 5
    """, note="the ON clause pairs foreign key (o.customer_id) with primary key (c.customer_id)")

    run(cur, 2, "Four-table chain join — who bought what, how many, for how much", """
        SELECT c.name AS customer, p.name AS product,
               oi.quantity, oi.quantity * p.price AS amount
        FROM order_items AS oi
        JOIN orders    AS o ON o.order_id    = oi.order_id
        JOIN customers AS c ON c.customer_id = o.customer_id
        JOIN products  AS p ON p.product_id  = oi.product_id
        ORDER BY o.order_id
        LIMIT 5
    """, note="follow the foreign-key arrows when chaining ON clauses and you won't get lost")

    run(cur, 3, "Join + aggregate — top 5 customers by total spend on completed orders", """
        SELECT c.name, c.grade,
               SUM(oi.quantity * p.price) AS total_amount
        FROM order_items AS oi
        JOIN orders    AS o ON o.order_id    = oi.order_id
        JOIN customers AS c ON c.customer_id = o.customer_id
        JOIN products  AS p ON p.product_id  = oi.product_id
        WHERE o.status = 'completed'
        GROUP BY c.customer_id, c.name, c.grade
        ORDER BY total_amount DESC
        LIMIT 5
    """)

    run(cur, 4, "LEFT JOIN — 'every' customer with their order count (0 if none)", """
        SELECT c.customer_id, c.name,
               COUNT(o.order_id) AS order_cnt
        FROM customers AS c
        LEFT JOIN orders AS o ON o.customer_id = c.customer_id
        GROUP BY c.customer_id, c.name
        ORDER BY order_cnt ASC
        LIMIT 5
    """, note="an INNER JOIN would have dropped the 0-order customers entirely")

    run(cur, 5, "Anti-join — the list of customers who never ordered (dormant customers)", """
        SELECT c.customer_id, c.name, c.city, c.grade
        FROM customers AS c
        LEFT JOIN orders AS o ON o.customer_id = c.customer_id
        WHERE o.order_id IS NULL
    """, note="keeping only the rows left NULL by having no partner = the 'find what's missing' pattern")

    # ------------------------------------------------------------------
    print("[6] The fan-out trap — the same 'order count' question, three answers")
    fanout = []
    cur.execute("SELECT COUNT(*) FROM orders")
    fanout.append(("(a) COUNT(*) on orders alone", cur.fetchone()[0], "correct"))
    cur.execute("""
        SELECT COUNT(*)
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id""")
    fanout.append(("(b) COUNT(*) after joining order_items", cur.fetchone()[0],
                   "inflated! each order replicated once per product"))
    cur.execute("""
        SELECT COUNT(DISTINCT o.order_id)
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id""")
    fanout.append(("(c) COUNT(DISTINCT order_id) after join", cur.fetchone()[0],
                   "DISTINCT removes the copies -> correct again"))
    widths = [max(disp_width(r[i]) for r in fanout) for i in range(3)]
    for r in fanout:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))
    print("  -> When a post-join aggregate feels 'too big', suspect fan-out first.\n")

    con.close()
    print("[Recap] JOIN = carrying a key to another book / LEFT = everyone on the left survives")
    print("  A 1:N join multiplies rows — defend your aggregates with COUNT(DISTINCT key).")


if __name__ == "__main__":
    main()
