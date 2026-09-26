"""
Lecture 04 · Level 10 — ウィンドウ関数と分析クエリ
行を折りたたまずに、横へ要約を貼り付けるウィンドウ関数 (OVER) を練習します。
GROUP BY との違い、ROW_NUMBER で顧客別の購入連番、RANK 三兄弟の同点
処理、部署別の売上順位、累計売上、3 か月移動合計まで — 分析レポートの
定番パターンをすべて実行してみます。
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
    print("観戦ポイント: 各結果で『行が折りたたまれたか、保たれたか』を確認してください。\n")

    run(cur, "1a", "GROUP BY — カテゴリ平均価格 (行が 5 行に『折りたたまれる』)", """
        SELECT category, ROUND(AVG(price), 0) AS avg_price
        FROM products
        GROUP BY category
    """)

    run(cur, "1b", "ウィンドウ — 同じ平均価格を『折りたたまずに』横へ貼る (10 行を維持)", """
        SELECT name, category, price,
               ROUND(AVG(price) OVER (PARTITION BY category), 0) AS cat_avg,
               price - ROUND(AVG(price) OVER (PARTITION BY category), 0) AS diff
        FROM products
        ORDER BY category, price DESC
    """, note="行ごとに『自分のカテゴリの窓』の外を眺めて、平均を付箋で貼ります")

    run(cur, 2, "ROW_NUMBER — 顧客別の購入連番、CTE で包んで『初回注文』だけ", """
        WITH numbered AS (
            SELECT customer_id, order_id, ordered_at,
                   ROW_NUMBER() OVER (PARTITION BY customer_id
                                      ORDER BY ordered_at, order_id) AS nth
            FROM orders
        )
        SELECT c.name, n.order_id, n.ordered_at, n.nth
        FROM numbered AS n
        JOIN customers AS c ON c.customer_id = n.customer_id
        WHERE n.nth = 1
        ORDER BY n.ordered_at
        LIMIT 5
    """, note="『グループごとに最初の 1 件』= ROW_NUMBER + 外側のフィルター、実務の最頻出公式")

    run(cur, 3, "同点処理の三兄弟 — 商品価格ランキング (同点の価格があってこそ差が見える)", """
        SELECT name, price,
               ROW_NUMBER() OVER (ORDER BY price DESC) AS row_num,
               RANK()       OVER (ORDER BY price DESC) AS rnk,
               DENSE_RANK() OVER (ORDER BY price DESC) AS dense_rnk
        FROM products
        ORDER BY price DESC
    """, note="同じ価格 (同点) で RANK は次の順位を飛ばし、DENSE_RANK は続けます")

    run(cur, 4, "部署別の売上順位 — 従業員別の担当売上 (CTE) → 部署内で RANK", """
        WITH emp_sales AS (
            SELECT e.employee_id, e.name, e.dept,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN employees AS e   ON e.employee_id = o.employee_id
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p     ON p.product_id = oi.product_id
            WHERE o.status = '完了'
            GROUP BY e.employee_id, e.name, e.dept
        )
        SELECT dept, name, revenue,
               RANK() OVER (PARTITION BY dept ORDER BY revenue DESC) AS dept_rank
        FROM emp_sales
        ORDER BY dept, dept_rank
        LIMIT 10
    """, note="『支店別/部署別ランキング』レポートの骨格: 集計 CTE → PARTITION BY で順位付け")

    run(cur, 5, "累計売上 — 月別売上の横に年初からの累計", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '完了'
            GROUP BY month
        )
        SELECT month, revenue,
               SUM(revenue) OVER (ORDER BY month) AS cum_revenue
        FROM monthly
        LIMIT 6
    """, note="ORDER BY が窓を『先頭〜現在の行』にして、累計になります")

    run(cur, 6, "3 か月移動合計 — 窓のサイズを直接指定 (ROWS BETWEEN)", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '完了'
            GROUP BY month
        )
        SELECT month, revenue,
               SUM(revenue) OVER (ORDER BY month
                                  ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS mov3
        FROM monthly
        LIMIT 6
    """, note="直前 2 行+現在の行 = 直近 3 か月。トレンドを滑らかに見せるレポート技法")

    con.close()
    print("[まとめ] 行数が減るなら GROUP BY、保たれるならウィンドウ (OVER)。")
    print("  窓の定義の 3 要素: PARTITION BY (範囲) / ORDER BY (順序) / ROWS (サイズ)。")


if __name__ == "__main__":
    main()
