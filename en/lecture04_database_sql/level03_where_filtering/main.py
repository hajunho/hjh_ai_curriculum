"""
Lecture 04 · Level 03 — WHERE: Searching with Conditions
Translates six questions you'd actually hear at work into SQL and runs
them. Covers comparisons (=, >=), AND/OR with parentheses, IN, LIKE,
BETWEEN, and IS NULL — plus a demo of the trap where writing '= NULL'
by mistake returns 0 rows.
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


def run(cur, step, question, sql, note=""):
    """Shared runner: business question -> SQL statement -> result table."""
    print(f"[{step}] Business question: {question}")
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
    print(f"  -> {len(rows)} rows" + (f" | {note}" if note else ""))
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"Practice DB ready: {db_path.name}")
    print("WHERE is a filter that judges each row true/false and passes only the true ones.\n")

    run(cur, "Q1", "Give me the list of VIP customers in New York (= and AND)", """
        SELECT name, city, grade
        FROM customers
        WHERE city = 'New York' AND grade = 'VIP'
        LIMIT 6
    """, note="only rows satisfying BOTH conditions pass")

    run(cur, "Q2", "Products at KRW 100,000+ or with a margin above 50%? (OR and parentheses)", """
        SELECT name, price, ROUND(100.0 * (price - cost) / price, 1) AS margin_pct
        FROM products
        WHERE (price >= 100000) OR (100.0 * (price - cost) / price > 50)
    """, note="the moment an OR appears, nail down intent with parentheses!")

    run(cur, "Q3", "New York/Chicago or Boston customers, VIP/GOLD only (combining IN)", """
        SELECT name, city, grade
        FROM customers
        WHERE city IN ('New York', 'Chicago', 'Boston')
          AND grade IN ('VIP', 'GOLD')
        LIMIT 6
    """, note="IN is the tidy shorthand for a chain of ORs")

    run(cur, "Q4", "Find customers whose last name is Smith (LIKE pattern)", """
        SELECT name, city
        FROM customers
        WHERE name LIKE '% Smith'
        LIMIT 6
    """, note="% is the 'any 0 or more characters' wildcard")

    run(cur, "Q5", "Cancelled orders in Q3 (Jul-Sep)? (BETWEEN + AND)", """
        SELECT order_id, customer_id, ordered_at, status
        FROM orders
        WHERE ordered_at BETWEEN '2025-07-01' AND '2025-09-30'
          AND status = 'cancelled'
        LIMIT 6
    """, note="BETWEEN includes both endpoints")

    run(cur, "Q6", "Which employee has no manager (top of the org chart)? (IS NULL)", """
        SELECT employee_id, name, dept
        FROM employees
        WHERE manager_id IS NULL
    """, note="NULL means 'never recorded' — use the dedicated syntax IS NULL")

    # Trap demo: comparing NULL with = is always false -> 0 rows
    run(cur, "Q6-trap", "The same question written with '= NULL' by mistake?", """
        SELECT employee_id, name, dept
        FROM employees
        WHERE manager_id = NULL
    """, note="0 rows! 'unknown = unknown' is also unknown, so it can never be true")

    con.close()
    print("[Recap] The condition wording in the question maps 1:1 onto the WHERE clause.")
    print("  at least/at most -> >= <= | one of -> IN | starts/ends with -> LIKE '..%'")
    print("  period -> BETWEEN | no value -> IS NULL")


if __name__ == "__main__":
    main()
