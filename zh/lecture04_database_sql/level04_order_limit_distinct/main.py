"""
Lecture 04 · Level 04 — 排序、去重与前 N 名
练习 ORDER BY (排队)、LIMIT/OFFSET (前 N·分页)、DISTINCT (去重)。
做出 '最贵商品 Top5' 这样的排行榜，并亲眼确认一个陷阱:
不加 ORDER BY 只用 LIMIT，得到的并不是 Top N。
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
    print("执行顺序: FROM → WHERE → SELECT → ORDER BY → LIMIT (先排队再截取)\n")

    run(cur, 1, "最贵商品 Top5 — 排行榜的基本形", """
        SELECT name, category, price
        FROM products
        ORDER BY price DESC
        LIMIT 5
    """, note="DESC = 降序(从贵到便宜)。Top N 是 ORDER BY + LIMIT 成套")

    run(cur, 2, "毛利最大的商品 Top3 — 用计算式当排序标准", """
        SELECT name, price, cost, price - cost AS margin
        FROM products
        ORDER BY margin DESC
        LIMIT 3
    """, note="别名(margin)可以直接用在 ORDER BY 里")

    run(cur, 3, "最新 5 笔订单 — 日期降序就是 '最近记录' 套路", """
        SELECT order_id, customer_id, ordered_at, status
        FROM orders
        ORDER BY ordered_at DESC
        LIMIT 5
    """)

    run(cur, 4, "多重排序 — 先按城市，同城再按等级", """
        SELECT city, grade, name
        FROM customers
        ORDER BY city ASC, grade ASC
        LIMIT 8
    """, note="逗号顺序就是优先级: 第一城市，第二等级")

    run(cur, 5, "DISTINCT — 客户实际居住的城市 '种类'", """
        SELECT DISTINCT city
        FROM customers
        ORDER BY city
    """, note="看看客户 200 行只剩城市种类后缩成了几行")

    run(cur, "5b", "DISTINCT 两列 — (城市, 等级) '组合' 的种类", """
        SELECT DISTINCT city, grade
        FROM customers
        ORDER BY city, grade
        LIMIT 8
    """, note="DISTINCT 作用于选中的整个列组合")

    run(cur, 6, "分页 — 价格排行第 6~10 名 (第 2 页)", """
        SELECT name, price
        FROM products
        ORDER BY price DESC
        LIMIT 5 OFFSET 5
    """, note="OFFSET 5 = 跳过前 5 行，取接下来的 5 行")

    run(cur, 7, "陷阱 — 不加 ORDER BY 只用 LIMIT 5 会怎样?", """
        SELECT name, price
        FROM products
        LIMIT 5
    """, note="只是 '随便 5 行'。和 [1] 的 Top5 对比一下!")

    con.close()
    print("[小结] 排行榜 = ORDER BY (DESC) + LIMIT / 种类 = DISTINCT")
    print("  原表的顺序和内容完全不变 (修整的只是结果画面)。")


if __name__ == "__main__":
    main()
