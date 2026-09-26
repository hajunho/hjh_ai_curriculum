"""
Lecture 04 · Level 09 — 索引与查询性能
造一张 30 万行的 big_orders 表，把同一个查询在建索引前后各实测一遍。
用 EXPLAIN QUERY PLAN 读出数据库的作战方案 (SCAN vs SEARCH)，
再用数字确认两件事: 像中间匹配 LIKE 这样索引帮不上忙的情况，
以及索引带来的写入成本。
"""

import pathlib
import random
import sqlite3
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data

N_ROWS = 300_000      # 实验台规模
N_REPEAT = 200        # 同一个查询重复多少次 (次数够多才量得出时间)


def explain(cur, query, params=()):
    """用 EXPLAIN QUERY PLAN 打印数据库的执行作战方案。"""
    print(f"  SQL> EXPLAIN QUERY PLAN {query}")
    cur.execute("EXPLAIN QUERY PLAN " + query, params)
    for row in cur.fetchall():
        print(f"       作战方案: {row[3]}")


def measure(cur, query, params_list):
    """反复执行查询，实测总耗时 (秒)。"""
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
    rng = random.Random(42)   # 固定 seed: 每次都是同一份实验数据

    # ------------------------------------------------------------------
    print(f"[1] 搭实验台 — 在 big_orders 表里生成 {N_ROWS:,} 行")
    cur.execute("CREATE TABLE big_orders ("
                "order_id INTEGER PRIMARY KEY, customer_id INTEGER, "
                "ordered_at TEXT, status TEXT, amount INTEGER)")
    rows = [(i,
             rng.randint(1, 50_000),
             f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
             rng.choice(["已完成", "已取消", "配送中"]),
             rng.randint(1_000, 500_000))
            for i in range(1, N_ROWS + 1)]
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO big_orders VALUES (?,?,?,?,?)", rows)
    con.commit()
    print(f"  生成完成 ({time.perf_counter() - t0:.2f}秒)。5 万名客户的 30 万笔订单。\n")

    query = "SELECT * FROM big_orders WHERE customer_id = ?"
    targets = [(rng.randint(1, 50_000),) for _ in range(N_REPEAT)]

    # ------------------------------------------------------------------
    print(f"[2] 没有索引时的检索 — 按客户号检索 {N_REPEAT} 次实测")
    explain(cur, query, (7,))
    t_before = measure(cur, query, targets)
    print(f"  耗时: {t_before:.3f}秒  ← 每检索 1 次就把 {N_ROWS:,} 行全翻一遍 (SCAN)\n")

    # ------------------------------------------------------------------
    print("[3] 建索引 — 相当于给书加一份目录")
    print("  SQL> CREATE INDEX idx_big_customer ON big_orders (customer_id)")
    t0 = time.perf_counter()
    cur.execute("CREATE INDEX idx_big_customer ON big_orders (customer_id)")
    con.commit()
    print(f"  创建耗时: {time.perf_counter() - t0:.2f}秒 (只付一次的成本)\n")

    # ------------------------------------------------------------------
    print(f"[4] 建索引后同样检索 {N_REPEAT} 次 — 查询语句一个字都没改!")
    explain(cur, query, (7,))
    t_after = measure(cur, query, targets)
    speedup = t_before / t_after if t_after > 0 else float("inf")
    print(f"  耗时: {t_after:.4f}秒")
    print(f"  → {t_before:.3f}秒 → {t_after:.4f}秒，快了大约 {speedup:,.0f} 倍\n")

    # ------------------------------------------------------------------
    print("[5] 索引帮不上忙的检索 — 中间匹配 LIKE 依然是 SCAN")
    explain(cur, "SELECT * FROM big_orders WHERE status LIKE '%送%'")
    print("  → 目录是按 '从第一个字起' 排序的，所以中间匹配找不到。")
    explain(cur, "SELECT * FROM big_orders WHERE customer_id = ? AND status = '已完成'", (7,))
    print("  → 同一个查询，只要带上了有索引的列 (customer_id)，就先 SEARCH 缩小再过滤。\n")

    # ------------------------------------------------------------------
    print("[6] 索引的账单 — 写入 (INSERT) 反而变慢了")
    extra = [(N_ROWS + i, rng.randint(1, 50_000), "2025-12-31", "已完成", 1000)
             for i in range(1, 10_001)]
    cur.execute("CREATE TABLE plain_copy AS SELECT * FROM big_orders WHERE 0")  # 没有索引的空副本
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO plain_copy VALUES (?,?,?,?,?)", extra)
    con.commit()
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO big_orders VALUES (?,?,?,?,?)", extra)  # 有索引的表
    con.commit()
    t_indexed = time.perf_counter() - t0
    print(f"  1 万行 INSERT — 无索引: {t_plain:.3f}秒 / 有索引: {t_indexed:.3f}秒")
    print("  → 索引是一笔交易: 买来检索速度，付出写入速度。只加在真正需要的列上!\n")

    con.close()
    print("[小结] 慢查询诊断两步走: 先用 EXPLAIN 确认是不是 SCAN → 再给 WHERE 的列加索引。")
    print("  索引候选 = WHERE/JOIN 里常用的列。全都加上，写入就要哭了。")


if __name__ == "__main__":
    main()
