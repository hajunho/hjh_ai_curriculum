"""
Lecture 04 · Level 02 — SELECT の基礎
SQL の出発点である SELECT/FROM を練習します。全列 (*)、列の選択、
別名 (AS)、計算列、文字列の連結まで — 各ステップで SQL 文を先に
出力し、そのすぐ下に実行結果を表として表示します。SQL が主役で
Python は実行係です。
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


def run(cur, step, title, sql):
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
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"実習 DB の準備完了: {db_path.name}")
    print("読み方のコツ: FROM (どのテーブルから) を先に探し、SELECT (どの列を) を読みます。\n")

    run(cur, 1, "全列をざっと眺める — テーブルへの最初のあいさつは * で", """
        SELECT *
        FROM customers
        LIMIT 5
    """)

    run(cur, 2, "必要な列だけ選ぶ — 依頼書に項目を明記", """
        SELECT name, city
        FROM customers
        LIMIT 5
    """)

    run(cur, 3, "別名 (AS) — 結果表の列見出しをレポート用に", """
        SELECT name  AS 商品名,
               category AS カテゴリ,
               price AS 販売価格
        FROM products
        LIMIT 5
    """)

    run(cur, 4, "計算列 — 行ごとに粗利と粗利率をその場で計算", """
        SELECT name AS 商品名,
               price AS 販売価格,
               cost  AS 原価,
               price - cost AS 粗利,
               ROUND(100.0 * (price - cost) / price, 1) AS 粗利率
        FROM products
        LIMIT 5
    """)

    run(cur, 5, "文字列の連結 (||) — 『氏名 (グレード)』表示用の列を作る", """
        SELECT name || ' (' || grade || ')' AS 顧客表示名,
               city AS 都市
        FROM customers
        LIMIT 5
    """)

    con.close()
    print("[6] まとめ")
    print("  - SELECT は読み取り専用: どんな文を実行しても原本は変わりません。")
    print("  - 別名と計算列は『見える画面』だけを変えます。")
    print("  - 次のレベル: WHERE で『条件に合う行だけ』を選び出します。")


if __name__ == "__main__":
    main()
