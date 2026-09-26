"""
Lecture 04 · Level 06 — JOIN: 複数テーブルをつなぐ
番号 (外部キー) しか書かれていない注文帳簿に顧客・商品の情報をつなぎ、
「誰が何を買ったのか」を分析します。INNER JOIN、多段ジョイン、JOIN+GROUP BY、
LEFT JOIN で注文のない顧客を探す方法、そしてジョインが集計を膨らませる
ファンアウト (fan-out) の落とし穴まで実演します。
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
    print("関係図: customers ──< orders >── employees / orders ──< order_items >── products")

    # LEFT JOIN の実習用に「まだ注文のない」新規入会の顧客 3 人を入れておきます。
    cur.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?)",
        [(201, "青山はる", "東京", "BASIC", "2025-12-01"),
         (202, "森下たくみ", "大阪", "BASIC", "2025-12-02"),
         (203, "秋田かえで", "札幌", "BASIC", "2025-12-03")])
    con.commit()
    print("(準備) 注文履歴のない新規顧客 3 人 (201〜203) を追加しました。\n")

    run(cur, 1, "二つのテーブルのジョイン — 注文帳簿の番号が名前に変わります", """
        SELECT o.order_id, c.name AS customer, c.city, o.ordered_at, o.status
        FROM orders AS o
        JOIN customers AS c ON c.customer_id = o.customer_id
        ORDER BY o.order_id
        LIMIT 5
    """, note="ON 句 = 外部キー (o.customer_id) と主キー (c.customer_id) のペア合わせ")

    run(cur, 2, "四つのテーブルの数珠つなぎ — 誰が何を何個、いくら分買ったか", """
        SELECT c.name AS customer, p.name AS product,
               oi.quantity, oi.quantity * p.price AS amount
        FROM order_items AS oi
        JOIN orders    AS o ON o.order_id    = oi.order_id
        JOIN customers AS c ON c.customer_id = o.customer_id
        JOIN products  AS p ON p.product_id  = oi.product_id
        ORDER BY o.order_id
        LIMIT 5
    """, note="外部キーの矢印に沿って ON をつなげば道に迷いません")

    run(cur, 3, "ジョイン + 集計 — 完了注文ベースの顧客別総購入額トップ 5", """
        SELECT c.name, c.grade,
               SUM(oi.quantity * p.price) AS total_amount
        FROM order_items AS oi
        JOIN orders    AS o ON o.order_id    = oi.order_id
        JOIN customers AS c ON c.customer_id = o.customer_id
        JOIN products  AS p ON p.product_id  = oi.product_id
        WHERE o.status = '完了'
        GROUP BY c.customer_id, c.name, c.grade
        ORDER BY total_amount DESC
        LIMIT 5
    """)

    run(cur, 4, "LEFT JOIN — 顧客『全員』とそれぞれの注文数 (なければ 0)", """
        SELECT c.customer_id, c.name,
               COUNT(o.order_id) AS order_cnt
        FROM customers AS c
        LEFT JOIN orders AS o ON o.customer_id = c.customer_id
        GROUP BY c.customer_id, c.name
        ORDER BY order_cnt ASC
        LIMIT 5
    """, note="INNER JOIN だったら注文 0 件の顧客はそもそも消えていたはずです")

    run(cur, 5, "アンチジョイン — 一度も注文していない顧客 (休眠顧客) リスト", """
        SELECT c.customer_id, c.name, c.city, c.grade
        FROM customers AS c
        LEFT JOIN orders AS o ON o.customer_id = c.customer_id
        WHERE o.order_id IS NULL
    """, note="相手がおらず NULL のまま残った行だけを選ぶ = 『ないものを探す』王道パターン")

    # ------------------------------------------------------------------
    print("[6] ファンアウトの落とし穴 — 同じ『注文数』の質問、三つの答え")
    fanout = []
    cur.execute("SELECT COUNT(*) FROM orders")
    fanout.append(("(a) orders 単独の COUNT(*)", cur.fetchone()[0], "正解"))
    cur.execute("""
        SELECT COUNT(*)
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id""")
    fanout.append(("(b) order_items をジョインした後の COUNT(*)", cur.fetchone()[0],
                   "膨張! 注文 1 件が商品の数だけ複製"))
    cur.execute("""
        SELECT COUNT(DISTINCT o.order_id)
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id""")
    fanout.append(("(c) ジョイン後の COUNT(DISTINCT order_id)", cur.fetchone()[0],
                   "DISTINCT で複製を除去 → 再び正解"))
    widths = [max(disp_width(r[i]) for r in fanout) for i in range(3)]
    for r in fanout:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))
    print("  → ジョイン後の集計が『体感より大きい』ときは、まずファンアウトを疑ってください。\n")

    con.close()
    print("[まとめ] JOIN = キーを持って別の帳簿を訪ねること / LEFT = 左側は全員生存")
    print("  1:N のジョインは行を膨らませます — 集計は COUNT(DISTINCT キー) で防御。")


if __name__ == "__main__":
    main()
