"""
Lecture 04 · Level 08 — Modifying Data and Transactions
Demonstrates the INSERT/UPDATE/DELETE basics plus transactions
(BEGIN/COMMIT/ROLLBACK), which bundle several changes into 'all succeed
or all cancelled'. The highlights: reviving a WHERE-less UPDATE accident
with a rollback, and a stock-deduction transaction's success/failure
(insufficient stock -> full rollback) scenarios.
"""

import pathlib
import sqlite3
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def sql(cur, statement, params=()):
    """Show an SQL statement and execute it (this level stars the data-changing statements)."""
    for line in statement.strip().splitlines():
        print(f"  SQL> {line.strip()}")
    cur.execute(statement, params)
    return cur


def show_stock(cur, label):
    """Print a one-line summary of the stock state."""
    cur.execute("SELECT product_id, stock FROM inventory WHERE product_id IN (1, 2)")
    state = ", ".join(f"product {pid} stock={s}" for pid, s in cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM orders")
    print(f"  [{label}] {state}, order count={cur.fetchone()[0]}")


def place_order(con, cur, order_id, product_id, qty):
    """Stock-deduction transaction: create order + deduct stock + verify -> commit or full rollback."""
    print(f"  Order attempt: product {product_id} x {qty} (order id {order_id})")
    cur.execute("BEGIN")
    print("  SQL> BEGIN  -- envelope opened: changes from here are still 'pencil notes'")
    try:
        sql(cur, "INSERT INTO orders (order_id, customer_id, employee_id, ordered_at, status) "
                 "VALUES (?, ?, 1, '2025-12-30', 'completed')", (order_id, 1))
        sql(cur, "UPDATE inventory SET stock = stock - ? WHERE product_id = ?",
            (qty, product_id))
        # Verify: a negative result after deduction breaks the rules -> cancel everything
        cur.execute("SELECT stock FROM inventory WHERE product_id = ?", (product_id,))
        stock_after = cur.fetchone()[0]
        if stock_after < 0:
            raise ValueError(f"insufficient stock (deducting leaves {stock_after})")
        cur.execute("COMMIT")
        print("  SQL> COMMIT  -- verification passed: inked in for good")
    except Exception as e:
        cur.execute("ROLLBACK")
        print(f"  SQL> ROLLBACK  -- problem found ({e}) -> envelope shredded, all undone")


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    con.isolation_level = None      # autocommit mode: we control BEGIN/COMMIT directly in SQL
    cur = con.cursor()

    # Practice stock table: 10 units each for products 1-10
    cur.execute("CREATE TABLE inventory (product_id INTEGER PRIMARY KEY, stock INTEGER)")
    cur.executemany("INSERT INTO inventory VALUES (?, ?)", [(i, 10) for i in range(1, 11)])
    print(f"Practice DB ready: {db_path.name} (+ inventory stock table, 10 per product)\n")

    # ------------------------------------------------------------------
    print("[1] INSERT — adding a new customer")
    cur.execute("SELECT COUNT(*) FROM customers")
    print(f"  Customer count before: {cur.fetchone()[0]}")
    sql(cur, "INSERT INTO customers (customer_id, name, city, grade, joined_at) "
             "VALUES (204, 'Paige Winters', 'New York', 'BASIC', '2025-12-20')")
    cur.execute("SELECT COUNT(*) FROM customers")
    print(f"  Customer count after: {cur.fetchone()[0]} -> one row added\n")

    # ------------------------------------------------------------------
    print("[2] UPDATE — the 'aim (SELECT) -> shoot (UPDATE)' safety rule")
    sql(cur, "SELECT customer_id, name, grade FROM customers WHERE customer_id = 204")
    print(f"  Aim result: {cur.fetchall()} <- confirm it's exactly 1 row!")
    sql(cur, "UPDATE customers SET grade = 'GOLD' WHERE customer_id = 204")
    cur.execute("SELECT grade FROM customers WHERE customer_id = 204")
    print(f"  Grade after shooting: {cur.fetchone()[0]}\n")

    # ------------------------------------------------------------------
    print("[3] The UPDATE-without-WHERE accident — and the ROLLBACK eraser")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    before_vip = cur.fetchone()[0]
    print(f"  VIP count before the accident: {before_vip}")
    cur.execute("BEGIN")
    print("  SQL> BEGIN")
    sql(cur, "UPDATE customers SET grade = 'VIP'   -- forgot the WHERE!")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    print(f"  VIP count right after: {cur.fetchone()[0]} <- every customer is a VIP! (still pencil)")
    cur.execute("ROLLBACK")
    print("  SQL> ROLLBACK")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    print(f"  VIP count after rollback: {cur.fetchone()[0]} -> restored. Before the commit, there's an eraser\n")

    # ------------------------------------------------------------------
    print("[4] Stock-deduction transaction — the success case (order + deduct + verify -> COMMIT)")
    show_stock(cur, "before")
    place_order(con, cur, order_id=1001, product_id=1, qty=3)
    show_stock(cur, "after")
    print("  -> One more order + stock 10->7. Both changes were confirmed together.\n")

    # ------------------------------------------------------------------
    print("[5] Stock-deduction transaction — the failure case (7 in stock, 20 ordered)")
    show_stock(cur, "before")
    place_order(con, cur, order_id=1002, product_id=1, qty=20)
    show_stock(cur, "after")
    print("  -> Order count and stock both identical to before the attempt. No half-state like")
    print("    'the order saved but the stock untouched' — that is atomicity (all or nothing).\n")

    con.close()
    print("[Recap] The life of a change statement is its WHERE; the life of a bundle is its transaction.")
    print("  The working skeleton: BEGIN -> changes -> verify -> COMMIT (ROLLBACK on trouble).")


if __name__ == "__main__":
    main()
