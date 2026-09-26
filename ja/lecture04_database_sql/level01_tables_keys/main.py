"""
Lecture 04 · Level 01 — テーブル・行・列・主キー
実習用ショップ DB (hjh_shop.db) を作り、DB 自身に構造を尋ねます。
テーブル一覧 → 各テーブルの列と主キー → 行数 → 外部キー関係図の順番で
スキーマ (schema) を読むコツを身につけます。初めて見る会社の DB を把握するときも、
この順番をそのまま使えば大丈夫です。
"""

import pathlib
import sqlite3
import sys
import unicodedata

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def disp_width(text):
    """日本語の全角文字は画面で 2 桁分を占めるため、表の整列用に表示幅を計算します。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width):
    return str(text) + " " * (width - disp_width(text))


def show_table(cols, rows):
    """照会結果を列幅をそろえた表として出力します。"""
    widths = [max(disp_width(c), *(disp_width(r[i]) for r in rows)) if rows else disp_width(c)
              for i, c in enumerate(cols)]
    print("  " + " | ".join(pad(c, w) for c, w in zip(cols, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))


def run(cur, sql):
    """SQL 文を表示してから実行し、結果を表として出力します。"""
    print(f"  SQL> {sql}")
    cur.execute(sql)
    show_table([d[0] for d in cur.description], cur.fetchall())
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"実習 DB の準備完了: {db_path.name}\n")

    # ------------------------------------------------------------------
    print("[1] テーブル一覧 — この DB にはどんな帳簿 (テーブル) があるか?")
    run(cur, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")

    # ------------------------------------------------------------------
    print("[2] 各テーブルの構造 — 列名・データ型・主キー (pk が 1 以上なら主キー)")
    tables = ["customers", "products", "employees", "orders", "order_items"]
    for t in tables:
        print(f"  SQL> PRAGMA table_info({t})")
        cur.execute(f"PRAGMA table_info({t})")
        rows = [(r[1], r[2], "PK" + str(r[5]) if r[5] else "") for r in cur.fetchall()]
        show_table(["column", "type", "key"], rows)
        print()

    # ------------------------------------------------------------------
    print("[3] テーブルごとの行数 — データ規模の確認")
    counts = []
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")   # COUNT(*) は行数を数えます
        counts.append((t, cur.fetchone()[0]))
    show_table(["table", "rows"], counts)
    print()

    # ------------------------------------------------------------------
    print("[4] 外部キー関係図 — _id 列がどのテーブルの主キーを指しているか")
    relations = [
        ("orders.customer_id",      "→ customers.customer_id", "注文の持ち主である顧客"),
        ("orders.employee_id",      "→ employees.employee_id", "注文を処理した従業員"),
        ("order_items.order_id",    "→ orders.order_id",       "どの注文の明細か"),
        ("order_items.product_id",  "→ products.product_id",   "どの商品か"),
        ("employees.manager_id",    "→ employees.employee_id", "直属の上司 (自己参照)"),
    ]
    show_table(["foreign key", "references", "meaning"], relations)
    print("""
  customers ──< orders >── employees
                  │            └──(manager_id 自己参照)
                  └──< order_items >── products
  (──< は 1:N の関係 — 顧客 1 人が注文 N 件を持つ)
""")

    # ------------------------------------------------------------------
    print("[5] 参照の確認 — 注文帳簿の customer_id は顧客名簿の行番号である")
    run(cur, "SELECT order_id, customer_id, ordered_at, status "
             "FROM orders WHERE customer_id = 7 LIMIT 3")
    run(cur, "SELECT customer_id, name, city, grade "
             "FROM customers WHERE customer_id = 7")
    print("  → 注文帳簿には名前の代わりに『顧客名簿の 7 番』という参照だけが書かれています。")
    print("    顧客情報が変わっても customers の 1 行だけ直せば済む理由です。")

    con.close()
    print("\n[6] まとめ: 初めて見る DB は (1)テーブル一覧 (2)列・主キー (3)行数")
    print("    (4)外部キー関係の順に把握します。次のレベルからいよいよ照会 (SELECT)!")


if __name__ == "__main__":
    main()
