"""
Lecture 04 · Level 04 — ソート・重複排除・上位 N 件
ORDER BY (並べ替え)、LIMIT/OFFSET (上位 N・ページ分け)、DISTINCT (重複排除) を
練習します。「最高価格の商品トップ 5」のようなランキング表を作り、ORDER BY なしで
LIMIT だけ書くとトップ N にならないという落とし穴も目で確認します。
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
    print("実行順序: FROM → WHERE → SELECT → ORDER BY → LIMIT (並べてから切る)\n")

    run(cur, 1, "最高価格の商品トップ 5 — ランキング表の基本形", """
        SELECT name, category, price
        FROM products
        ORDER BY price DESC
        LIMIT 5
    """, note="DESC = 降順 (高いものから)。トップ N は ORDER BY + LIMIT のセット")

    run(cur, 2, "粗利の大きい商品トップ 3 — 計算式を並べ替えの基準に", """
        SELECT name, price, cost, price - cost AS margin
        FROM products
        ORDER BY margin DESC
        LIMIT 3
    """, note="別名 (margin) を ORDER BY でそのまま使えます")

    run(cur, 3, "最新の注文 5 件 — 日付の降順は『直近の履歴』パターン", """
        SELECT order_id, customer_id, ordered_at, status
        FROM orders
        ORDER BY ordered_at DESC
        LIMIT 5
    """)

    run(cur, 4, "複数キーのソート — 都市順、同じ都市の中はグレード順", """
        SELECT city, grade, name
        FROM customers
        ORDER BY city ASC, grade ASC
        LIMIT 8
    """, note="カンマの順番が優先順位: 1 次は都市、2 次はグレード")

    run(cur, 5, "DISTINCT — 顧客が実際に住んでいる都市の『種類』", """
        SELECT DISTINCT city
        FROM customers
        ORDER BY city
    """, note="顧客 200 行が都市の種類だけになり、何行に減ったか見てください")

    run(cur, "5b", "DISTINCT 二列 — (都市, グレード) の『組み合わせ』の種類", """
        SELECT DISTINCT city, grade
        FROM customers
        ORDER BY city, grade
        LIMIT 8
    """, note="DISTINCT は選択した列の組み合わせ全体に適用されます")

    run(cur, 6, "ページ分け — 価格ランキング 6〜10 位 (2 ページ目)", """
        SELECT name, price
        FROM products
        ORDER BY price DESC
        LIMIT 5 OFFSET 5
    """, note="OFFSET 5 = 先頭 5 行を飛ばして次の 5 行")

    run(cur, 7, "落とし穴 — ORDER BY なしで LIMIT 5 だけ書くと?", """
        SELECT name, price
        FROM products
        LIMIT 5
    """, note="ただの『適当な 5 行』です。[1] のトップ 5 と見比べてください!")

    con.close()
    print("[まとめ] ランキング = ORDER BY (DESC) + LIMIT / 種類 = DISTINCT")
    print("  原本テーブルの順序や内容はまったく変わりません (結果の画面を整えるだけ)。")


if __name__ == "__main__":
    main()
