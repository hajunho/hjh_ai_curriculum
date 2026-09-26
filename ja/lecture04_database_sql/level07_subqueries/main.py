"""
Lecture 04 · Level 07 — サブクエリと CTE
クエリの結果を別のクエリの材料として使う方法を練習します。
スカラーサブクエリ (値一つ)、IN サブクエリ (リスト)、相関サブクエリ (行ごとに再計算)、
そして WITH (CTE) で「平均以上の購入をした顧客」のような多段階の分析を読みやすく
整理するところまで扱います。
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
    """全角文字は 2 桁分の幅なので、表の整列用に表示幅を計算します。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width):
    return str(text) + " " * (width - disp_width(text))


def run(cur, step, title, sql, note=""):
    """SQL 文を表示し、実行結果を表として出力する共用の実行係。"""
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
        print(f"  → {note}")
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"実習 DB の準備完了: {db_path.name}")
    print("サブクエリ = かっこの中のメモ: 内側の答えを外側のクエリの空欄にはめ込む\n")

    # メモ 1: 基準値を先に目で確認
    cur.execute("SELECT ROUND(AVG(price), 0) FROM products")
    print(f"(メモ 1) 商品の平均価格 = {cur.fetchone()[0]:,.0f}ウォン — この値が下のかっこにはめ込まれます。\n")

    run(cur, 1, "スカラーサブクエリ — 平均より高い商品", """
        SELECT name, price
        FROM products
        WHERE price > (SELECT AVG(price) FROM products)
        ORDER BY price DESC
    """, note="かっこが先に実行されて『値一つ』に変わる、と読めば OK です")

    run(cur, 2, "IN サブクエリ — 『ノートパソコン』を買ったことのある顧客 (一部)", """
        SELECT customer_id, name, city
        FROM customers
        WHERE customer_id IN (
            SELECT o.customer_id
            FROM orders AS o
            WHERE o.order_id IN (
                SELECT oi.order_id
                FROM order_items AS oi
                JOIN products AS p ON p.product_id = oi.product_id
                WHERE p.name = 'ノートパソコン'))
        ORDER BY customer_id
        LIMIT 6
    """, note="内側の答え (注文番号リスト → 顧客番号リスト) が IN のリストの位置に")

    run(cur, "3a", "準備 — 部署別の平均給与 (対照用)", """
        SELECT dept, ROUND(AVG(salary), 0) AS avg_salary
        FROM employees
        GROUP BY dept
    """)

    run(cur, "3b", "相関サブクエリ — 自分の部署の平均より給与が高い従業員", """
        SELECT e.name, e.dept, e.salary
        FROM employees AS e
        WHERE e.salary > (SELECT AVG(e2.salary)
                          FROM employees AS e2
                          WHERE e2.dept = e.dept)
        ORDER BY e.dept, e.salary DESC
    """, note="内側が外側の行の e.dept を参照 → 行ごとに基準が変わります")

    run(cur, 4, "CTE (WITH) — 平均以上の購入をした顧客 (完了注文ベース)", """
        WITH customer_totals AS (
            SELECT o.customer_id,
                   SUM(oi.quantity * p.price) AS total
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '完了'
            GROUP BY o.customer_id
        )
        SELECT c.name, c.grade, t.total
        FROM customer_totals AS t
        JOIN customers AS c ON c.customer_id = t.customer_id
        WHERE t.total > (SELECT AVG(total) FROM customer_totals)
        ORDER BY t.total DESC
        LIMIT 6
    """, note="1 段目に名前を付けて、2 段目で『二回』再利用 — CTE の力")

    run(cur, 5, "CTE 複数ステップ — 月別売上を作り、最高売上の月を探す", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '完了'
            GROUP BY month
        )
        SELECT month, revenue
        FROM monthly
        WHERE revenue = (SELECT MAX(revenue) FROM monthly)
    """, note="レポートクエリの典型: 段階ごとの CTE → 最後に答え一つ")

    con.close()
    print("[まとめ] 値一つ → スカラー / リスト → IN / 行ごとに基準 → 相関")
    print("  かっこが二重になったら CTE に昇格させて『読めるクエリ』を作りましょう。")


if __name__ == "__main__":
    main()
