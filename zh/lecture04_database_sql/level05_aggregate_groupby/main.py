"""
Lecture 04 · Level 05 — 聚合函数与 GROUP BY
用 COUNT/SUM/AVG/MIN/MAX 做汇总，用 GROUP BY 做按城市、按品类的小计，
再用 HAVING 给 '小计本身' 设条件。WHERE (分组前的行过滤) 和
HAVING (分组后的组过滤) 的区别是本关的核心。
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
    print("执行顺序: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY\n")

    run(cur, 1, "整体汇总 — 不带 GROUP BY 的聚合就是 '整张表 = 一个篮子'", """
        SELECT COUNT(*) AS product_cnt,
               SUM(price) AS price_sum,
               ROUND(AVG(price), 1) AS price_avg,
               MIN(price) AS price_min,
               MAX(price) AS price_max
        FROM products
    """, note="多行被折叠成汇总值 '一行'")

    run(cur, 2, "COUNT 的三副面孔 — *、列、DISTINCT", """
        SELECT COUNT(*) AS all_rows,
               COUNT(manager_id) AS has_manager,
               COUNT(DISTINCT dept) AS dept_kinds
        FROM employees
    """, note="COUNT(列) 会跳过 NULL (总经理的 manager_id 是 NULL)")

    run(cur, 3, "各城市客户数 — GROUP BY 基本形", """
        SELECT city, COUNT(*) AS customer_cnt
        FROM customers
        GROUP BY city
        ORDER BY customer_cnt DESC
    """, note="结果一行 = 一个篮子(城市)。检算一下合计是不是 200")

    run(cur, 4, "城市×等级客户数 — 两个分组标准", """
        SELECT city, grade, COUNT(*) AS cnt
        FROM customers
        GROUP BY city, grade
        ORDER BY city, grade
        LIMIT 8
    """, note="等于往数据透视表里拖了两个字段")

    run(cur, 5, "各品类销售额 — 订单明细×商品连接 (下一关 JOIN 的预告片)", """
        SELECT p.category,
               SUM(oi.quantity * p.price) AS revenue
        FROM order_items AS oi
        JOIN products AS p ON p.product_id = oi.product_id
        GROUP BY p.category
        ORDER BY revenue DESC
    """, note="逐行算出 数量×单价，再按品类篮子 SUM")

    run(cur, 6, "HAVING — 只要客户 35 人以上的城市 (给小计设条件)", """
        SELECT city, COUNT(*) AS cnt
        FROM customers
        GROUP BY city
        HAVING COUNT(*) >= 35
        ORDER BY cnt DESC
    """, note="WHERE COUNT(*)>=35 会报错 — 分组之前没法数数")

    run(cur, 7, "WHERE + HAVING 联手 — 剔除已取消后按月数订单，只要 80 笔以上的月份", """
        SELECT SUBSTR(ordered_at, 1, 7) AS month,
               COUNT(*) AS order_cnt
        FROM orders
        WHERE status <> '已取消'
        GROUP BY month
        HAVING COUNT(*) >= 80
        ORDER BY month
    """, note="WHERE 筛的是行(单笔订单)，HAVING 筛的是篮子(月份)")

    con.close()
    print("[小结] 小计 = GROUP BY + 聚合函数 / 小计条件 = HAVING")
    print("  SELECT 里只能放 '篮子标签' 和 '篮子汇总值'。")


if __name__ == "__main__":
    main()
