"""
Lecture 04 · Level 05 — 集計関数と GROUP BY
COUNT/SUM/AVG/MIN/MAX で要約し、GROUP BY で都市別・カテゴリ別の小計を
作り、HAVING で「小計に対する条件」をかけます。WHERE (まとめる前の行フィルター) と
HAVING (まとめた後のグループフィルター) の違いがこのレベルの核心です。
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
    print("実行順序: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY\n")

    run(cur, 1, "全体の要約 — GROUP BY のない集計は『テーブル全体 = かご一つ』", """
        SELECT COUNT(*) AS product_cnt,
               SUM(price) AS price_sum,
               ROUND(AVG(price), 1) AS price_avg,
               MIN(price) AS price_min,
               MAX(price) AS price_max
        FROM products
    """, note="複数の行が要約値の『一行』に折りたたまれます")

    run(cur, 2, "COUNT の三つの顔 — *、列、DISTINCT", """
        SELECT COUNT(*) AS all_rows,
               COUNT(manager_id) AS has_manager,
               COUNT(DISTINCT dept) AS dept_kinds
        FROM employees
    """, note="COUNT(列) は NULL を除いて数えます (社長は manager_id が NULL)")

    run(cur, 3, "都市別の顧客数 — GROUP BY の基本形", """
        SELECT city, COUNT(*) AS customer_cnt
        FROM customers
        GROUP BY city
        ORDER BY customer_cnt DESC
    """, note="結果の一行 = かご (都市) 一つ。合計が 200 になるか検算してみてください")

    run(cur, 4, "都市×グレード別の顧客数 — グループの基準を二つ", """
        SELECT city, grade, COUNT(*) AS cnt
        FROM customers
        GROUP BY city, grade
        ORDER BY city, grade
        LIMIT 8
    """, note="ピボットテーブルにフィールドを二つ入れたのと同じです")

    run(cur, 5, "カテゴリ別の売上 — 注文明細×商品の連結 (次のレベル JOIN の先取り)", """
        SELECT p.category,
               SUM(oi.quantity * p.price) AS revenue
        FROM order_items AS oi
        JOIN products AS p ON p.product_id = oi.product_id
        GROUP BY p.category
        ORDER BY revenue DESC
    """, note="数量×単価を行ごとに計算し、カテゴリのかご別に SUM")

    run(cur, 6, "HAVING — 顧客 35 人以上の都市だけ (小計にかける条件)", """
        SELECT city, COUNT(*) AS cnt
        FROM customers
        GROUP BY city
        HAVING COUNT(*) >= 35
        ORDER BY cnt DESC
    """, note="WHERE COUNT(*)>=35 はエラー — まとめる前には数えられません")

    run(cur, 7, "WHERE + HAVING の合わせ技 — キャンセルを除いた月別注文、80 件以上の月だけ", """
        SELECT SUBSTR(ordered_at, 1, 7) AS month,
               COUNT(*) AS order_cnt
        FROM orders
        WHERE status <> 'キャンセル'
        GROUP BY month
        HAVING COUNT(*) >= 80
        ORDER BY month
    """, note="WHERE は行 (注文 1 件) を、HAVING はかご (月) をふるいにかけます")

    con.close()
    print("[まとめ] 小計 = GROUP BY + 集計関数 / 小計の条件 = HAVING")
    print("  SELECT に置けるのは『かごの名札』と『かごの要約値』だけです。")


if __name__ == "__main__":
    main()
