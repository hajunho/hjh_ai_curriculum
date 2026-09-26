"""
Lecture 04 · Level 02 — SELECT Basics
Practice SELECT/FROM, the starting point of SQL: all columns (*),
picking columns, aliases (AS), calculated columns, and string building.
Each step prints the SQL statement first, then the result right below
it as a table. SQL is the star; Python is just the runner.
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


def run(cur, step, title, sql):
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
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"Practice DB ready: {db_path.name}")
    print("Reading tip: find the FROM (which table) first, then read the SELECT (which columns).\n")

    run(cur, 1, "Skim all columns — say hello to a table with *", """
        SELECT *
        FROM customers
        LIMIT 5
    """)

    run(cur, 2, "Only the columns you need — name the fields on the request", """
        SELECT name, city
        FROM customers
        LIMIT 5
    """)

    run(cur, 3, "Aliases (AS) — report-friendly column headers", """
        SELECT name  AS product_name,
               category AS product_category,
               price AS unit_price
        FROM products
        LIMIT 5
    """)

    run(cur, 4, "Calculated columns — margin and margin % computed per row, on the spot", """
        SELECT name AS product_name,
               price AS unit_price,
               cost  AS unit_cost,
               price - cost AS margin,
               ROUND(100.0 * (price - cost) / price, 1) AS margin_pct
        FROM products
        LIMIT 5
    """)

    run(cur, 5, "String building (||) — a 'name (grade)' display column", """
        SELECT name || ' (' || grade || ')' AS customer_display,
               city AS customer_city
        FROM customers
        LIMIT 5
    """)

    con.close()
    print("[6] Recap")
    print("  - SELECT is read-only: no statement here ever changes the original.")
    print("  - Aliases and calculated columns change only the 'displayed screen'.")
    print("  - Next level: WHERE — keeping only the rows that match a condition.")


if __name__ == "__main__":
    main()
