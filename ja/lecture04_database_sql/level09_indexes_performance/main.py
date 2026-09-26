"""
Lecture 04 · Level 09 — インデックスとクエリ性能
30 万行の big_orders テーブルを作り、同じ検索をインデックスの前/後で
実測します。EXPLAIN QUERY PLAN で DB の作戦 (SCAN vs SEARCH) を読み、
中間一致の LIKE のようにインデックスが役に立たない場合と、インデックスの
書き込みコストまで数字で確認します。
"""

import pathlib
import random
import sqlite3
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data

N_ROWS = 300_000      # 実験台のサイズ
N_REPEAT = 200        # 同じ検索の繰り返し回数 (時間を測れるように)


def explain(cur, query, params=()):
    """EXPLAIN QUERY PLAN で DB の実行作戦を出力します。"""
    print(f"  SQL> EXPLAIN QUERY PLAN {query}")
    cur.execute("EXPLAIN QUERY PLAN " + query, params)
    for row in cur.fetchall():
        print(f"       作戦: {row[3]}")


def measure(cur, query, params_list):
    """クエリを繰り返し実行して、合計所要時間 (秒) を実測します。"""
    t0 = time.perf_counter()
    for p in params_list:
        cur.execute(query, p)
        cur.fetchall()
    return time.perf_counter() - t0


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rng = random.Random(42)   # seed 固定: 毎回同じ実験データ

    # ------------------------------------------------------------------
    print(f"[1] 実験台の準備 — big_orders テーブルに {N_ROWS:,}行を生成")
    cur.execute("CREATE TABLE big_orders ("
                "order_id INTEGER PRIMARY KEY, customer_id INTEGER, "
                "ordered_at TEXT, status TEXT, amount INTEGER)")
    rows = [(i,
             rng.randint(1, 50_000),
             f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
             rng.choice(["完了", "キャンセル", "配送中"]),
             rng.randint(1_000, 500_000))
            for i in range(1, N_ROWS + 1)]
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO big_orders VALUES (?,?,?,?,?)", rows)
    con.commit()
    print(f"  生成完了 ({time.perf_counter() - t0:.2f}秒)。顧客 5 万人の注文 30 万件。\n")

    query = "SELECT * FROM big_orders WHERE customer_id = ?"
    targets = [(rng.randint(1, 50_000),) for _ in range(N_REPEAT)]

    # ------------------------------------------------------------------
    print(f"[2] インデックスなしで検索 — 顧客番号の検索 {N_REPEAT}回を実測")
    explain(cur, query, (7,))
    t_before = measure(cur, query, targets)
    print(f"  所要時間: {t_before:.3f}秒  ← 検索 1 回ごとに {N_ROWS:,}行を全部なめる (SCAN)\n")

    # ------------------------------------------------------------------
    print("[3] インデックスの作成 — 索引を一冊作る")
    print("  SQL> CREATE INDEX idx_big_customer ON big_orders (customer_id)")
    t0 = time.perf_counter()
    cur.execute("CREATE INDEX idx_big_customer ON big_orders (customer_id)")
    con.commit()
    print(f"  作成時間: {time.perf_counter() - t0:.2f}秒 (一度だけ払うコスト)\n")

    # ------------------------------------------------------------------
    print(f"[4] インデックス後に同じ検索を {N_REPEAT}回 — クエリの文はそのまま!")
    explain(cur, query, (7,))
    t_after = measure(cur, query, targets)
    speedup = t_before / t_after if t_after > 0 else float("inf")
    print(f"  所要時間: {t_after:.4f}秒")
    print(f"  → {t_before:.3f}秒 → {t_after:.4f}秒、約 {speedup:,.0f}倍の高速化\n")

    # ------------------------------------------------------------------
    print("[5] インデックスが役に立たない検索 — 中間一致の LIKE は相変わらず SCAN")
    explain(cur, "SELECT * FROM big_orders WHERE status LIKE '%送%'")
    print("  → 索引は『先頭の文字から』の並びなので、中間一致は探せません。")
    explain(cur, "SELECT * FROM big_orders WHERE customer_id = ? AND status = '完了'", (7,))
    print("  → 同じクエリでも、インデックスのある列 (customer_id) があれば SEARCH で絞ってからふるいます。\n")

    # ------------------------------------------------------------------
    print("[6] インデックスの請求書 — 書き込み (INSERT) はむしろ遅くなります")
    extra = [(N_ROWS + i, rng.randint(1, 50_000), "2025-12-31", "完了", 1000)
             for i in range(1, 10_001)]
    cur.execute("CREATE TABLE plain_copy AS SELECT * FROM big_orders WHERE 0")  # インデックスのない空のコピー
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO plain_copy VALUES (?,?,?,?,?)", extra)
    con.commit()
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO big_orders VALUES (?,?,?,?,?)", extra)  # インデックスのあるテーブル
    con.commit()
    t_indexed = time.perf_counter() - t0
    print(f"  1 万行の INSERT — インデックスなし: {t_plain:.3f}秒 / インデックスあり: {t_indexed:.3f}秒")
    print("  → インデックスは検索を買って書き込みで支払う取引です。必要な列にだけ!\n")

    con.close()
    print("[まとめ] 遅いクエリの診断 2 ステップ: EXPLAIN で SCAN を確認 → WHERE の列にインデックス。")
    print("  インデックスの候補 = WHERE/JOIN でよく使う列。全部に張ると書き込みが泣きます。")


if __name__ == "__main__":
    main()
