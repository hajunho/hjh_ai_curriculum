"""
Lecture 04 · Level 11 — Python 連携とデータパイプライン
SQL と Python をつなぐ最後のレベルです。パラメータバインディング (?) が SQL
インジェクションを防ぐ原理を安全なローカル例で実演し、pandas.read_sql で照会結果を
DataFrame として受け取り、「抽出 (SQL) → 加工 (pandas) → 保存 (CSV)+要約レポート」の
ミニ ETL パイプラインを完成させます。
"""

import pathlib
import sqlite3
import sys

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def search_city_unsafe(cur, user_input):
    """[悪い例] ユーザー入力を文字列連結で SQL に組み立てる — インジェクションにさらされる。"""
    sql = f"SELECT name, city FROM customers WHERE city = '{user_input}'"
    print(f"  (組み立てられた文) {sql}")
    cur.execute(sql)
    return cur.fetchall()


def search_city_safe(cur, user_input):
    """[良い例] 文は書式 (?)、値は別で渡す — 入力が文になることはできない。"""
    sql = "SELECT name, city FROM customers WHERE city = ?"
    print(f"  (文) {sql}   (値) {user_input!r}")
    cur.execute(sql, (user_input,))     # 値が一つでもタプル (カンマ!)
    return cur.fetchall()


def extract(con):
    """[E] 抽出: 必要なデータだけを SQL で絞って持ってきます (完了注文のみ)。"""
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
    return pd.read_sql(sql, con, params=("完了",))


def transform(df):
    """[T] 加工: pandas で月×カテゴリのピボットを作り、月合計を付けます。"""
    pivot = pd.pivot_table(df, values="amount", index="month",
                           columns="category", aggfunc="sum", fill_value=0)
    pivot["月合計"] = pivot.sum(axis=1)
    return pivot


def load_and_report(df, pivot, out_dir):
    """[L] 保存: 要約表を CSV で保存し、経営陣向けのテキストレポートを出力します。"""
    out_path = out_dir / "monthly_category_revenue.csv"
    pivot.to_csv(out_path, encoding="utf-8-sig")   # Excel で日本語が文字化けしないエンコーディング
    print(f"  保存完了 → {out_path}")

    total = df["amount"].sum()
    best_month = pivot["月合計"].idxmax()
    best_cat = pivot.drop(columns="月合計").sum().idxmax()
    top3 = df.groupby("customer")["amount"].sum().nlargest(3)
    print("\n  ---- 自動要約レポート (完了注文ベース) ----")
    print(f"  · 年間総売上          : {total:,}ウォン")
    print(f"  · 最高売上の月        : {best_month} ({pivot.loc[best_month, '月合計']:,}ウォン)")
    print(f"  · 売上 1 位のカテゴリ : {best_cat}")
    print("  · 購入額トップ 3 顧客 :")
    for name, amt in top3.items():
        print(f"      {name}: {amt:,}ウォン")


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"実習 DB の準備完了: {db_path.name}\n")

    # ------------------------------------------------------------------
    print("[1] Python から SQL を実行する 4 ステップ — 接続→カーソル→実行 (バインディング)→結果")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = ?", ("VIP",))
    print(f"  VIP 顧客数: {cur.fetchone()[0]}人 (値 'VIP' は ? の空欄として渡す)\n")

    # ------------------------------------------------------------------
    print("[2] SQL インジェクションの実演 — 同じ悪意ある入力、二つの運命 (ローカルの実習 DB 内だけ)")
    evil = "東京' OR '1'='1"      # 条件を「常に真」に変えようとする古典的な入力
    print(f"  悪意ある入力: {evil!r}")
    print("  (a) 文字列連結 (f-string) 方式:")
    rows = search_city_unsafe(cur, evil)
    print(f"      → {len(rows)}件が流出! 条件が『常に真』の文に改ざんされました。")
    print("  (b) ? バインディング方式:")
    rows = search_city_safe(cur, evil)
    print(f"      → {len(rows)}件。入力全体が『都市名という値』として扱われ、")
    print("        そんな都市はないので 0 件 — これが正解です。")
    print("  ルール: SQL にユーザーの値が入るなら、無条件で ? バインディング!\n")

    # ------------------------------------------------------------------
    print("[3] Extract — pandas.read_sql でジョイン結果を DataFrame に")
    df = extract(con)
    print(f"  受け取ったデータ: {df.shape[0]}行 × {df.shape[1]}列 (完了注文の品目別金額)")
    print(df.head(3).to_string(index=False))
    print("  → 絞る仕事 (WHERE/JOIN) は SQL が、整える仕事は pandas がやります。\n")

    # ------------------------------------------------------------------
    print("[4] Transform — 月×カテゴリ売上のピボット (一部のみ表示)")
    pivot = transform(df)
    print(pivot.head(4).to_string())
    print()

    # ------------------------------------------------------------------
    print("[5] Load + レポート — CSV 保存と自動要約")
    out_dir = BASE / "outputs"
    out_dir.mkdir(exist_ok=True)
    load_and_report(df, pivot, out_dir)

    con.close()
    print("\n[まとめ] このスクリプトがそのままパイプラインです: 予約実行 (cron など) に載せれば")
    print("  『毎週月曜の朝の自動レポート』になります。会社の DB に切り替えるときは")
    print("  connect の部分をその DB のドライバーに差し替えるだけで、SQL はそのまま通用します。")


if __name__ == "__main__":
    main()
