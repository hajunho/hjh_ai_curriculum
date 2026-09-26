"""
Lecture 04 · Level 00 — Why Do We Need Databases?
Replays the three classic accidents of 'passing Excel files around'
(lost updates, consistency pollution, missing permissions) as a Python
simulation, then runs the same scenario the database way (sqlite3) to
contrast what changes. SQL syntax starts in the next level, so for now
just focus on the *difference* in the outputs.
"""

import copy
import pathlib
import sqlite3
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data  # shared data generator (builds the practice online-store DB)


def show_rows(title, rows):
    """Print a list of dictionaries as a simple table."""
    print(f"  {title}")
    for r in rows:
        print("   ", " | ".join(f"{k}={v}" for k, v in r.items()))


def excel_style_disaster():
    """[1] Concurrent edits: two people edit copies of the same file, then save in turn."""
    print("[1] The Excel way — concurrent-edit accident (lost update)")
    shared_file = [  # imagine 'customer_list.xlsx' on the shared drive
        {"customer_id": 1, "name": "James Smith", "grade": "SILVER"},
        {"customer_id": 2, "name": "Emma Johnson", "grade": "BASIC"},
    ]
    # Both people 'download' the file (each makes a copy)
    kim_copy = copy.deepcopy(shared_file)
    park_copy = copy.deepcopy(shared_file)

    kim_copy[0]["grade"] = "VIP"      # Kim: promotes customer 1 to VIP
    park_copy[1]["grade"] = "GOLD"    # Park: promotes customer 2 to GOLD

    shared_file = kim_copy    # Kim saves first (overwrites the whole file)
    shared_file = park_copy   # Park saves later -> Kim's work disappears!

    show_rows("Final saved file:", shared_file)
    print("  -> Customer 1 is still SILVER. Kim's promotion evaporated 'without an error'.\n")


def integrity_disaster():
    """[2] Consistency: a ledger where the same client was entered twice with different spellings."""
    print("[2] The Excel way — consistency pollution (same client, different spelling)")
    dirty_ledger = [
        {"client": "Acme Corp.", "amount": 300},
        {"client": "Acme Corporation", "amount": 200},   # actually the same company
        {"client": "Zenith Trading", "amount": 150},
    ]
    totals = {}
    for row in dirty_ledger:
        totals[row["client"]] = totals.get(row["client"], 0) + row["amount"]
    for name, amount in totals.items():
        print(f"    {name}: {amount} (unit: KRW 10,000)")
    print("  -> Acme is really a 500 client, but it splits into 300/200 and the report is wrong.\n")


def permission_disaster():
    """[3] Permissions: sending the file sends every sensitive column along with it."""
    print("[3] The Excel way — no permissions (sending the file = leaking it all)")
    hr_file = [
        {"name": "Olivia Taylor", "dept": "Sales", "salary": 9000},
        {"name": "Lucas Miller", "dept": "Engineering", "salary": 7200},
    ]
    print("  You attach the file saying 'just for the org chart', but the recipient sees:")
    show_rows("Contents of the file as delivered:", hr_file)
    print("  -> The only way to drop the salary column is 'make a new file'. The original can't be controlled.\n")


def database_way():
    """[4] The DB way: one original + a request counter. Re-run the same scenario."""
    print("[4] The database way — same scenario, different outcome")
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))   # build the practice store DB (nothing to install)
    print(f"  Practice DB created: {db_path.name} (5 tables incl. customers)")

    # Two connections = two users. One original; each just sends 'requests'.
    kim = sqlite3.connect(db_path)
    park = sqlite3.connect(db_path)

    kim.execute("UPDATE customers SET grade='VIP' WHERE customer_id=1")
    kim.commit()      # Kim: request to promote customer 1 -> the counter writes it to the original
    park.execute("UPDATE customers SET grade='GOLD' WHERE customer_id=2")
    park.commit()     # Park: request to promote customer 2 -> also written to the original

    cur = kim.execute(
        "SELECT customer_id, name, grade FROM customers WHERE customer_id IN (1, 2)")
    rows = [{"customer_id": r[0], "name": r[1], "grade": r[2]} for r in cur.fetchall()]
    show_rows("Current state of the original table:", rows)
    print("  -> Both edits survived. One original means no overwrite accidents.")

    # Consistency: data that breaks a rule (constraint) is refused at save time.
    try:
        kim.execute("INSERT INTO customers (customer_id, name) VALUES (1, 'Ghost Customer')")
    except sqlite3.IntegrityError as e:
        print(f"  Attempt to save a duplicate customer_id -> DB refuses: {e}")

    # Permissions: you can expose a read-only 'window' (a view) that omits sensitive columns.
    kim.execute("CREATE VIEW IF NOT EXISTS emp_public AS "
                "SELECT name, dept FROM employees")   # the salary column is left out entirely
    cur = kim.execute("SELECT * FROM emp_public LIMIT 2")
    rows = [{"name": r[0], "dept": r[1]} for r in cur.fetchall()]
    show_rows("What the shared window (view) shows:", rows)
    print("  -> The salary column doesn't exist at this counter, so it cannot leak.\n")

    kim.close()
    park.close()


def main():
    print("=" * 62)
    print(" The three little hells of Excel sharing vs a database")
    print("=" * 62 + "\n")
    excel_style_disaster()
    integrity_disaster()
    permission_disaster()
    database_way()
    print("[5] Recap")
    print("  The Excel way: many copies -> accidents happen without even an error.")
    print("  The DB way   : one original + a request counter -> ordering, rule checks, permissions.")
    print("  From the next level on, we learn SQL — the language for talking to that counter.")


if __name__ == "__main__":
    main()
