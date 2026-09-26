"""
Lecture 04 · Level 06 — JOIN: 连接多张表
给只写着编号 (外键) 的订单账本接上客户、商品信息，分析 '谁买了什么'。
演示 INNER JOIN、多表连接、JOIN+GROUP BY、用 LEFT JOIN 找没下过单的客户，
以及连接会把汇总吹大的扇出 (fan-out) 陷阱。
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
    """中文字符占 2 格宽，这里计算表格对齐用的显示宽度。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width):
    return str(text) + " " * (width - disp_width(text))


def run(cur, step, title, sql, note=""):
    """展示 SQL 语句并把执行结果打印成表格的公用执行器。"""
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
    print(f"练习数据库准备完成: {db_path.name}")
    print("关系图: customers ──< orders >── employees / orders ──< order_items >── products")

    # 为 LEFT JOIN 练习准备 3 位 '还没有订单' 的新注册客户。
    cur.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?)",
        [(201, "陈新雨", "北京", "BASIC", "2025-12-01"),
         (202, "周一诺", "上海", "BASIC", "2025-12-02"),
         (203, "吴思颖", "成都", "BASIC", "2025-12-03")])
    con.commit()
    print("(准备) 添加了 3 位没有订单记录的新客户 (201~203)。\n")

    run(cur, 1, "两张表连接 — 订单账本上的编号变成了姓名", """
        SELECT o.order_id, c.name AS customer, c.city, o.ordered_at, o.status
        FROM orders AS o
        JOIN customers AS c ON c.customer_id = o.customer_id
        ORDER BY o.order_id
        LIMIT 5
    """, note="ON 子句 = 外键(o.customer_id)和主键(c.customer_id)的配对")

    run(cur, 2, "四张表链式连接 — 谁买了什么、几件、多少钱", """
        SELECT c.name AS customer, p.name AS product,
               oi.quantity, oi.quantity * p.price AS amount
        FROM order_items AS oi
        JOIN orders    AS o ON o.order_id    = oi.order_id
        JOIN customers AS c ON c.customer_id = o.customer_id
        JOIN products  AS p ON p.product_id  = oi.product_id
        ORDER BY o.order_id
        LIMIT 5
    """, note="顺着外键箭头把 ON 一节节接上就不会迷路")

    run(cur, 3, "连接 + 聚合 — 按已完成订单计算客户总购买额 Top5", """
        SELECT c.name, c.grade,
               SUM(oi.quantity * p.price) AS total_amount
        FROM order_items AS oi
        JOIN orders    AS o ON o.order_id    = oi.order_id
        JOIN customers AS c ON c.customer_id = o.customer_id
        JOIN products  AS p ON p.product_id  = oi.product_id
        WHERE o.status = '已完成'
        GROUP BY c.customer_id, c.name, c.grade
        ORDER BY total_amount DESC
        LIMIT 5
    """)

    run(cur, 4, "LEFT JOIN — 客户 '全员' 与各自的订单数 (没有就是 0)", """
        SELECT c.customer_id, c.name,
               COUNT(o.order_id) AS order_cnt
        FROM customers AS c
        LEFT JOIN orders AS o ON o.customer_id = c.customer_id
        GROUP BY c.customer_id, c.name
        ORDER BY order_cnt ASC
        LIMIT 5
    """, note="换成 INNER JOIN 的话，0 单客户会直接消失")

    run(cur, 5, "反连接 — 一次都没下过单的客户 (沉睡客户) 名单", """
        SELECT c.customer_id, c.name, c.city, c.grade
        FROM customers AS c
        LEFT JOIN orders AS o ON o.customer_id = c.customer_id
        WHERE o.order_id IS NULL
    """, note="只挑没配上对、留成 NULL 的行 = '找不存在之物' 的正统套路")

    # ------------------------------------------------------------------
    print("[6] 扇出陷阱 — 同一个 '订单数' 问题，三种答案")
    fanout = []
    cur.execute("SELECT COUNT(*) FROM orders")
    fanout.append(("(a) orders 单表 COUNT(*)", cur.fetchone()[0], "正确答案"))
    cur.execute("""
        SELECT COUNT(*)
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id""")
    fanout.append(("(b) 接上 order_items 后 COUNT(*)", cur.fetchone()[0],
                   "被吹大了! 一笔订单被复制成商品数那么多行"))
    cur.execute("""
        SELECT COUNT(DISTINCT o.order_id)
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id""")
    fanout.append(("(c) 连接后 COUNT(DISTINCT order_id)", cur.fetchone()[0],
                   "用 DISTINCT 去掉复制 → 又是正确答案"))
    widths = [max(disp_width(r[i]) for r in fanout) for i in range(3)]
    for r in fanout:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))
    print("  → 连接后的汇总 '比感觉的大' 时，先怀疑扇出。\n")

    con.close()
    print("[小结] JOIN = 拿着键去别的账本查 / LEFT = 左边全员生还")
    print("  1:N 连接会让行变多 — 汇总时用 COUNT(DISTINCT 键) 防御。")


if __name__ == "__main__":
    main()
