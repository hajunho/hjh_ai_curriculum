"""
Lecture 04 · Level 11 — Python Integration and Data Pipelines
The final level connects SQL and Python. It demonstrates, with a safe
local example, how parameter binding (?) blocks SQL injection, then
receives query results as a DataFrame with pandas.read_sql to complete
a mini ETL pipeline: 'extract (SQL) -> transform (pandas) -> load (CSV)
+ summary report'.
"""

import pathlib
import sqlite3
import sys

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def search_city_unsafe(cur, user_input):
    """[Bad example] Splice user input into the SQL as a string — exposed to injection."""
    sql = f"SELECT name, city FROM customers WHERE city = '{user_input}'"
    print(f"  (assembled statement) {sql}")
    cur.execute(sql)
    return cur.fetchall()


def search_city_safe(cur, user_input):
    """[Good example] The statement is a form (?), the value travels separately — input can't become a sentence."""
    sql = "SELECT name, city FROM customers WHERE city = ?"
    print(f"  (statement) {sql}   (value) {user_input!r}")
    cur.execute(sql, (user_input,))     # even a single value goes as a tuple (comma!)
    return cur.fetchall()


def extract(con):
    """[E] Extract: shrink the data in SQL and fetch only what's needed (completed orders)."""
    sql = """
        SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
               p.category,
               c.name AS customer,
               oi.quantity * p.price AS amount
        FROM orders AS o
        JOIN customers AS c   ON c.customer_id = o.customer_id
        JOIN order_items AS oi ON oi.order_id = o.order_id
        JOIN products AS p     ON p.product_id = oi.product_id
        WHERE o.status = ?
    """
    return pd.read_sql(sql, con, params=("completed",))


def transform(df):
    """[T] Transform: build the month x category pivot in pandas and add the monthly total."""
    pivot = pd.pivot_table(df, values="amount", index="month",
                           columns="category", aggfunc="sum", fill_value=0)
    pivot["month_total"] = pivot.sum(axis=1)
    return pivot


def load_and_report(df, pivot, out_dir):
    """[L] Load: save the summary table to CSV and print an executive text report."""
    out_path = out_dir / "monthly_category_revenue.csv"
    pivot.to_csv(out_path, encoding="utf-8-sig")   # an encoding Excel opens cleanly
    print(f"  Saved -> {out_path}")

    total = df["amount"].sum()
    best_month = pivot["month_total"].idxmax()
    best_cat = pivot.drop(columns="month_total").sum().idxmax()
    top3 = df.groupby("customer")["amount"].sum().nlargest(3)
    print("\n  ---- Automated summary report (completed orders) ----")
    print(f"  · Annual total revenue : KRW {total:,}")
    print(f"  · Best month           : {best_month} (KRW {pivot.loc[best_month, 'month_total']:,})")
    print(f"  · Top revenue category : {best_cat}")
    print("  · Top 3 customers by spend:")
    for name, amt in top3.items():
        print(f"      {name}: KRW {amt:,}")


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"Practice DB ready: {db_path.name}\n")

    # ------------------------------------------------------------------
    print("[1] The four steps of SQL from Python — connect -> cursor -> execute (binding) -> fetch")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = ?", ("VIP",))
    print(f"  VIP customers: {cur.fetchone()[0]} (the value 'VIP' travels through the ? blank)\n")

    # ------------------------------------------------------------------
    print("[2] SQL injection demo — same malicious input, two fates (local practice DB only)")
    evil = "New York' OR '1'='1"      # the classic input that tries to make the condition 'always true'
    print(f"  Malicious input: {evil!r}")
    print("  (a) The string-splicing (f-string) way:")
    rows = search_city_unsafe(cur, evil)
    print(f"      -> {len(rows)} rows leaked! The statement was rewritten into an 'always true' condition.")
    print("  (b) The ? binding way:")
    rows = search_city_safe(cur, evil)
    print(f"      -> {len(rows)} rows. The entire input is treated as 'a value that is a city name',")
    print("        no such city exists, so 0 rows — which is the correct answer.")
    print("  The rule: the moment a user value enters SQL, use ? binding!\n")

    # ------------------------------------------------------------------
    print("[3] Extract — the join result as a DataFrame via pandas.read_sql")
    df = extract(con)
    print(f"  Received: {df.shape[0]} rows x {df.shape[1]} cols (line-item amounts of completed orders)")
    print(df.head(3).to_string(index=False))
    print("  -> SQL does the shrinking (WHERE/JOIN); pandas does the shaping.\n")

    # ------------------------------------------------------------------
    print("[4] Transform — month x category revenue pivot (partial view)")
    pivot = transform(df)
    print(pivot.head(4).to_string())
    print()

    # ------------------------------------------------------------------
    print("[5] Load + report — CSV save and automated summary")
    out_dir = BASE / "outputs"
    out_dir.mkdir(exist_ok=True)
    load_and_report(df, pivot, out_dir)

    con.close()
    print("\n[Recap] This script IS a pipeline: hook it into a scheduler (cron etc.) and it")
    print("  becomes 'the automatic Monday-morning report'. Moving to the company DB means")
    print("  swapping only the connect line for that DB's driver — the SQL carries over as is.")


if __name__ == "__main__":
    main()
