"""
Lecture 04 · Level 10 — 窗口函数与分析查询
练习不把行折起来、而是在旁边贴上汇总的窗口函数 (OVER)。
和 GROUP BY 的区别、用 ROW_NUMBER 排每位客户的购买序号、RANK 三兄弟的
并列处理、部门内销售额排名、累计销售额、3 个月移动求和 —
把分析报表里的常客套路全跑一遍。
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
    print("观察重点: 每个结果里都确认一下 '行被折起来了，还是保留着'。\n")

    run(cur, "1a", "GROUP BY — 各品类均价 (行被 '折' 成了 5 行)", """
        SELECT category, ROUND(AVG(price), 0) AS avg_price
        FROM products
        GROUP BY category
    """)

    run(cur, "1b", "窗口 — 同样的均价 '不折行' 贴在旁边 (10 行保留)", """
        SELECT name, category, price,
               ROUND(AVG(price) OVER (PARTITION BY category), 0) AS cat_avg,
               price - ROUND(AVG(price) OVER (PARTITION BY category), 0) AS diff
        FROM products
        ORDER BY category, price DESC
    """, note="每一行都朝 '自己品类的那扇窗' 望一眼，把平均值当便利贴贴上去")

    run(cur, 2, "ROW_NUMBER — 每位客户的购买序号，用 CTE 裹住只留 '首单'", """
        WITH numbered AS (
            SELECT customer_id, order_id, ordered_at,
                   ROW_NUMBER() OVER (PARTITION BY customer_id
                                      ORDER BY ordered_at, order_id) AS nth
            FROM orders
        )
        SELECT c.name, n.order_id, n.ordered_at, n.nth
        FROM numbered AS n
        JOIN customers AS c ON c.customer_id = n.customer_id
        WHERE n.nth = 1
        ORDER BY n.ordered_at
        LIMIT 5
    """, note="'每组取最早 1 条' = ROW_NUMBER + 外层过滤，实战中最常用的公式")

    run(cur, 3, "并列处理三兄弟 — 商品价格排名 (有并列价格才看得出区别)", """
        SELECT name, price,
               ROW_NUMBER() OVER (ORDER BY price DESC) AS row_num,
               RANK()       OVER (ORDER BY price DESC) AS rnk,
               DENSE_RANK() OVER (ORDER BY price DESC) AS dense_rnk
        FROM products
        ORDER BY price DESC
    """, note="价格相同 (并列) 时，RANK 会跳过下一个名次，DENSE_RANK 则接着往下排")

    run(cur, 4, "部门内销售额排名 — 先算员工业绩 (CTE)，再在部门内 RANK", """
        WITH emp_sales AS (
            SELECT e.employee_id, e.name, e.dept,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN employees AS e   ON e.employee_id = o.employee_id
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p     ON p.product_id = oi.product_id
            WHERE o.status = '已完成'
            GROUP BY e.employee_id, e.name, e.dept
        )
        SELECT dept, name, revenue,
               RANK() OVER (PARTITION BY dept ORDER BY revenue DESC) AS dept_rank
        FROM emp_sales
        ORDER BY dept, dept_rank
        LIMIT 10
    """, note="'各门店/各部门排名' 这类报表的骨架: 聚合 CTE → PARTITION BY 排名")

    run(cur, 5, "累计销售额 — 在月度销售额旁边贴上年初至今的累计", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '已完成'
            GROUP BY month
        )
        SELECT month, revenue,
               SUM(revenue) OVER (ORDER BY month) AS cum_revenue
        FROM monthly
        LIMIT 6
    """, note="ORDER BY 把窗口变成 '从开头到当前行'，于是就成了累计")

    run(cur, 6, "3 个月移动求和 — 自己指定窗口大小 (ROWS BETWEEN)", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '已完成'
            GROUP BY month
        )
        SELECT month, revenue,
               SUM(revenue) OVER (ORDER BY month
                                  ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS mov3
        FROM monthly
        LIMIT 6
    """, note="前 2 行 + 当前行 = 最近 3 个月。让趋势看起来更平滑的报表手法")

    con.close()
    print("[小结] 行数变少就用 GROUP BY，行数保留就用窗口 (OVER)。")
    print("  窗口定义三要素: PARTITION BY (范围) / ORDER BY (顺序) / ROWS (大小)。")


if __name__ == "__main__":
    main()
