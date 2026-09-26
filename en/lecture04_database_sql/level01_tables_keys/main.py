"""
Lecture 04 · Level 01 — Tables, Rows, Columns, and Primary Keys
Builds the practice online-store DB (hjh_shop.db) and asks the DB itself
about its structure. You'll learn the routine for reading a schema:
table list -> each table's columns and primary key -> row counts ->
foreign-key relationship diagram. Use this exact order the first time
you meet an unfamiliar company DB, too.
"""

import pathlib
import sqlite3
import sys
import unicodedata

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def disp_width(text):
    """Wide (CJK) characters take 2 cells on screen; compute display width for alignment."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width):
    return str(text) + " " * (width - disp_width(text))


def show_table(cols, rows):
    """Print a query result as a table with aligned column widths."""
    widths = [max(disp_width(c), *(disp_width(r[i]) for r in rows)) if rows else disp_width(c)
              for i, c in enumerate(cols)]
    print("  " + " | ".join(pad(c, w) for c, w in zip(cols, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))


def run(cur, sql):
    """Show the SQL statement, run it, and print the result as a table."""
    print(f"  SQL> {sql}")
    cur.execute(sql)
    show_table([d[0] for d in cur.description], cur.fetchall())
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"Practice DB ready: {db_path.name}\n")

    # ------------------------------------------------------------------
    print("[1] Table list — which books (tables) does this DB contain?")
    run(cur, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")

    # ------------------------------------------------------------------
    print("[2] Each table's structure — column names, types, primary key (pk >= 1 means PK)")
    tables = ["customers", "products", "employees", "orders", "order_items"]
    for t in tables:
        print(f"  SQL> PRAGMA table_info({t})")
        cur.execute(f"PRAGMA table_info({t})")
        rows = [(r[1], r[2], "PK" + str(r[5]) if r[5] else "") for r in cur.fetchall()]
        show_table(["column", "type", "key"], rows)
        print()

    # ------------------------------------------------------------------
    print("[3] Row count per table — gauging the size of the data")
    counts = []
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")   # COUNT(*) counts rows
        counts.append((t, cur.fetchone()[0]))
    show_table(["table", "rows"], counts)
    print()

    # ------------------------------------------------------------------
    print("[4] Foreign-key map — which table's primary key does each _id column point at?")
    relations = [
        ("orders.customer_id",      "-> customers.customer_id", "the customer who owns the order"),
        ("orders.employee_id",      "-> employees.employee_id", "the employee who handled it"),
        ("order_items.order_id",    "-> orders.order_id",       "which order this line belongs to"),
        ("order_items.product_id",  "-> products.product_id",   "which product it is"),
        ("employees.manager_id",    "-> employees.employee_id", "direct manager (self-reference)"),
    ]
    show_table(["foreign key", "references", "meaning"], relations)
    print("""
  customers ──< orders >── employees
                  │            └──(manager_id self-reference)
                  └──< order_items >── products
  (──< is a 1:N relationship — 1 customer has N orders)
""")

    # ------------------------------------------------------------------
    print("[5] Reference check — the order book's customer_id is a row number in the customer register")
    run(cur, "SELECT order_id, customer_id, ordered_at, status "
             "FROM orders WHERE customer_id = 7 LIMIT 3")
    run(cur, "SELECT customer_id, name, city, grade "
             "FROM customers WHERE customer_id = 7")
    print("  -> The order book stores only the reference 'customer register row 7', not a name.")
    print("    That's why a customer change means fixing just one row of customers.")

    con.close()
    print("\n[6] Recap: read an unfamiliar DB in this order — (1) table list (2) columns & PKs")
    print("    (3) row counts (4) foreign-key relationships. Next level: real queries (SELECT)!")


if __name__ == "__main__":
    main()
