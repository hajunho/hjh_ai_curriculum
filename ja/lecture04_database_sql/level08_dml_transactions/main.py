"""
Lecture 04 · Level 08 — データの変更とトランザクション
INSERT/UPDATE/DELETE の基本と一緒に、複数の変更を「全部成功か、全部
取り消しか」でまとめるトランザクション (BEGIN/COMMIT/ROLLBACK) を実演します。
WHERE のない UPDATE 事故をロールバックで救出する場面と、在庫引き落とし
トランザクションの成功/失敗 (在庫不足 → 全体ロールバック) のシナリオが核心です。
"""

import pathlib
import sqlite3
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def sql(cur, statement, params=()):
    """SQL 文を表示してから実行します (このレベルは変更文が主役)。"""
    for line in statement.strip().splitlines():
        print(f"  SQL> {line.strip()}")
    cur.execute(statement, params)
    return cur


def show_stock(cur, label):
    """在庫の状態を一行で要約して出力します。"""
    cur.execute("SELECT product_id, stock FROM inventory WHERE product_id IN (1, 2)")
    state = ", ".join(f"商品{pid} 在庫={s}" for pid, s in cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM orders")
    print(f"  [{label}] {state}, 注文数={cur.fetchone()[0]}")


def place_order(con, cur, order_id, product_id, qty):
    """在庫引き落としトランザクション: 注文作成 + 在庫引き落とし + 検証 → コミットか全体ロールバック。"""
    print(f"  注文の試行: 商品{product_id} × {qty}個 (注文番号 {order_id})")
    cur.execute("BEGIN")
    print("  SQL> BEGIN  -- 封筒を開く: 以降の変更はまだ『鉛筆のメモ』")
    try:
        sql(cur, "INSERT INTO orders (order_id, customer_id, employee_id, ordered_at, status) "
                 "VALUES (?, ?, 1, '2025-12-30', '完了')", (order_id, 1))
        sql(cur, "UPDATE inventory SET stock = stock - ? WHERE product_id = ?",
            (qty, product_id))
        # 検証: 引き落とした結果がマイナスならルール違反 → 全体取り消し
        cur.execute("SELECT stock FROM inventory WHERE product_id = ?", (product_id,))
        stock_after = cur.fetchone()[0]
        if stock_after < 0:
            raise ValueError(f"在庫不足 (引き落とすと {stock_after}個)")
        cur.execute("COMMIT")
        print("  SQL> COMMIT  -- 検証を通過: ボールペンで確定")
    except Exception as e:
        cur.execute("ROLLBACK")
        print(f"  SQL> ROLLBACK  -- 問題発生({e}) → 封筒ごと破棄、全部なかったことに")


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    con.isolation_level = None      # 自動コミットモード: BEGIN/COMMIT を SQL で直接制御
    cur = con.cursor()

    # 実習用の在庫テーブル: 商品 1〜10 番にそれぞれ 10 個ずつ
    cur.execute("CREATE TABLE inventory (product_id INTEGER PRIMARY KEY, stock INTEGER)")
    cur.executemany("INSERT INTO inventory VALUES (?, ?)", [(i, 10) for i in range(1, 11)])
    print(f"実習 DB の準備完了: {db_path.name} (+ inventory 在庫テーブル、商品ごとに 10 個)\n")

    # ------------------------------------------------------------------
    print("[1] INSERT — 新規顧客の追加")
    cur.execute("SELECT COUNT(*) FROM customers")
    print(f"  追加前の顧客数: {cur.fetchone()[0]}")
    sql(cur, "INSERT INTO customers (customer_id, name, city, grade, joined_at) "
             "VALUES (204, '松本ふゆ', '東京', 'BASIC', '2025-12-20')")
    cur.execute("SELECT COUNT(*) FROM customers")
    print(f"  追加後の顧客数: {cur.fetchone()[0]} → 一行増えました\n")

    # ------------------------------------------------------------------
    print("[2] UPDATE — 『照準 (SELECT) → 発射 (UPDATE)』の安全ルール")
    sql(cur, "SELECT customer_id, name, grade FROM customers WHERE customer_id = 204")
    print(f"  照準の結果: {cur.fetchall()} ← ちょうど 1 行かどうか確認!")
    sql(cur, "UPDATE customers SET grade = 'GOLD' WHERE customer_id = 204")
    cur.execute("SELECT grade FROM customers WHERE customer_id = 204")
    print(f"  発射後のグレード: {cur.fetchone()[0]}\n")

    # ------------------------------------------------------------------
    print("[3] WHERE を忘れた UPDATE 事故 — そして ROLLBACK という消しゴム")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    before_vip = cur.fetchone()[0]
    print(f"  事故前の VIP 数: {before_vip}")
    cur.execute("BEGIN")
    print("  SQL> BEGIN")
    sql(cur, "UPDATE customers SET grade = 'VIP'   -- WHERE を忘れた!")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    print(f"  事故直後の VIP 数: {cur.fetchone()[0]} ← 全顧客が VIP! (まだ鉛筆のメモ)")
    cur.execute("ROLLBACK")
    print("  SQL> ROLLBACK")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    print(f"  ロールバック後の VIP 数: {cur.fetchone()[0]} → 原状回復。コミット前なら消しゴムがあります\n")

    # ------------------------------------------------------------------
    print("[4] 在庫引き落としトランザクション — 成功例 (注文 + 引き落とし + 検証 → COMMIT)")
    show_stock(cur, "試行前")
    place_order(con, cur, order_id=1001, product_id=1, qty=3)
    show_stock(cur, "試行後")
    print("  → 注文が 1 件増加 + 在庫 10→7。二つの変更が一緒に確定されました。\n")

    # ------------------------------------------------------------------
    print("[5] 在庫引き落としトランザクション — 失敗例 (在庫 7 個なのに 20 個の注文)")
    show_stock(cur, "試行前")
    place_order(con, cur, order_id=1002, product_id=1, qty=20)
    show_stock(cur, "試行後")
    print("  → 注文数も在庫も試行前とまったく同じ。『注文だけ保存されて在庫はそのまま』")
    print("    のような中途半端な状態がありません — これが原子性 (全か無か)。\n")

    con.close()
    print("[まとめ] 変更文の命は WHERE、まとまった変更の命はトランザクション。")
    print("  実務の骨格: BEGIN → 変更たち → 検証 → COMMIT (問題があれば ROLLBACK)。")


if __name__ == "__main__":
    main()
