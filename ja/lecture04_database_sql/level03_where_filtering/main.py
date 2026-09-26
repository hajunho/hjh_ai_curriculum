"""
Lecture 04 · Level 03 — WHERE: 条件検索
業務で出てきそうな質問 6 つを日本語 → SQL に翻訳して実行します。
比較演算 (=, >=)、AND/OR とかっこ、IN、LIKE、BETWEEN、IS NULL をすべて
扱い、'= NULL' と書き間違えるとなぜ 0 件になるのか、落とし穴も実演します。
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


def run(cur, step, question, sql, note=""):
    """業務の質問 → SQL 文 → 実行結果の表、の順で出力する共用の実行係。"""
    print(f"[{step}] 業務の質問: {question}")
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
    print(f"  → {len(rows)}件" + (f" | {note}" if note else ""))
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"実習 DB の準備完了: {db_path.name}")
    print("WHERE は行ごとに真/偽を判定し、『真の行だけ』を通すフィルターです。\n")

    run(cur, "Q1", "東京在住の VIP 顧客リストをください (= と AND)", """
        SELECT name, city, grade
        FROM customers
        WHERE city = '東京' AND grade = 'VIP'
        LIMIT 6
    """, note="二つの条件を『両方』満たす行だけが通過")

    run(cur, "Q2", "10万ウォン以上、または粗利率 50% 超の商品は? (OR とかっこ)", """
        SELECT name, price, ROUND(100.0 * (price - cost) / price, 1) AS margin_pct
        FROM products
        WHERE (price >= 100000) OR (100.0 * (price - cost) / price > 50)
    """, note="OR が混ざったら、かっこで意図を釘付けにする習慣を!")

    run(cur, "Q3", "東京・横浜または大阪の顧客のうち VIP/GOLD だけ (IN の組み合わせ)", """
        SELECT name, city, grade
        FROM customers
        WHERE city IN ('東京', '横浜', '大阪')
          AND grade IN ('VIP', 'GOLD')
        LIMIT 6
    """, note="IN は OR を並べる書き方のすっきりした省略形")

    run(cur, "Q4", "佐藤姓の顧客を探してください (LIKE パターン)", """
        SELECT name, city
        FROM customers
        WHERE name LIKE '佐藤%'
        LIMIT 6
    """, note="% は『任意の文字 0 個以上』のワイルドカード")

    run(cur, "Q5", "第3四半期 (7〜9月) のキャンセル注文の明細は? (BETWEEN + AND)", """
        SELECT order_id, customer_id, ordered_at, status
        FROM orders
        WHERE ordered_at BETWEEN '2025-07-01' AND '2025-09-30'
          AND status = 'キャンセル'
        LIMIT 6
    """, note="BETWEEN は両端の値を含む")

    run(cur, "Q6", "上司のいない従業員 (組織図の頂点) は誰? (IS NULL)", """
        SELECT employee_id, name, dept
        FROM employees
        WHERE manager_id IS NULL
    """, note="NULL は『記録されていない』— 専用の文法 IS NULL を使用")

    # 落とし穴の実演: NULL を = で比較すると常に偽 → 0 件
    run(cur, "Q6-落とし穴", "同じ質問を '= NULL' と書き間違えると?", """
        SELECT employee_id, name, dept
        FROM employees
        WHERE manager_id = NULL
    """, note="0 件! 『不明 = 不明』の答えも不明なので、真にはなれません")

    con.close()
    print("[まとめ] 日本語の質問の条件表現が WHERE 句と 1:1 で対応します。")
    print("  以上/以下 → >= <= | 〜のどれか → IN | 〜で始まる → LIKE '..%'")
    print("  期間 → BETWEEN | 値がない → IS NULL")


if __name__ == "__main__":
    main()
