"""
Lecture 04 · Level 07 — 子查询
练习把一个查询的结果当另一个查询的原料。
覆盖标量子查询 (一个值)、IN 子查询 (名单)、相关子查询 (每行重新计算)，
以及用 WITH (CTE) 把 '购买额高于平均的客户' 这类多步分析
整理得清晰好读。
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
    print("子查询 = 括号里的便条: 把内层的答案填进外层查询的空格\n")

    # 便条 1: 先亲眼确认标准值
    cur.execute("SELECT ROUND(AVG(price), 0) FROM products")
    print(f"(便条 1) 商品平均价 = {cur.fetchone()[0]:,.0f} 韩元 — 这个值会被填进下面的括号。\n")

    run(cur, 1, "标量子查询 — 比平均贵的商品", """
        SELECT name, price
        FROM products
        WHERE price > (SELECT AVG(price) FROM products)
        ORDER BY price DESC
    """, note="读法: 括号先执行、变成 '一个值'")

    run(cur, 2, "IN 子查询 — 买过 '笔记本电脑' 的客户 (部分)", """
        SELECT customer_id, name, city
        FROM customers
        WHERE customer_id IN (
            SELECT o.customer_id
            FROM orders AS o
            WHERE o.order_id IN (
                SELECT oi.order_id
                FROM order_items AS oi
                JOIN products AS p ON p.product_id = oi.product_id
                WHERE p.name = '笔记本电脑'))
        ORDER BY customer_id
        LIMIT 6
    """, note="内层的答案 (订单号名单 → 客户号名单) 填进 IN 的名单位置")

    run(cur, "3a", "准备 — 各部门平均薪资 (对照用)", """
        SELECT dept, ROUND(AVG(salary), 0) AS avg_salary
        FROM employees
        GROUP BY dept
    """)

    run(cur, "3b", "相关子查询 — 薪资高于本部门平均的员工", """
        SELECT e.name, e.dept, e.salary
        FROM employees AS e
        WHERE e.salary > (SELECT AVG(e2.salary)
                          FROM employees AS e2
                          WHERE e2.dept = e.dept)
        ORDER BY e.dept, e.salary DESC
    """, note="内层引用了外层行的 e.dept → 每一行的标准都不同")

    run(cur, 4, "CTE(WITH) — 购买额高于平均的客户 (按已完成订单)", """
        WITH customer_totals AS (
            SELECT o.customer_id,
                   SUM(oi.quantity * p.price) AS total
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '已完成'
            GROUP BY o.customer_id
        )
        SELECT c.name, c.grade, t.total
        FROM customer_totals AS t
        JOIN customers AS c ON c.customer_id = t.customer_id
        WHERE t.total > (SELECT AVG(total) FROM customer_totals)
        ORDER BY t.total DESC
        LIMIT 6
    """, note="给第 1 步起名，第 2 步 '两次' 复用它 — CTE 的威力")

    run(cur, 5, "多步 CTE — 先做月度销售额，再找销售额最高的月份", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '已完成'
            GROUP BY month
        )
        SELECT month, revenue
        FROM monthly
        WHERE revenue = (SELECT MAX(revenue) FROM monthly)
    """, note="报表查询的典型: 分步 CTE → 最后取一个答案")

    con.close()
    print("[小结] 一个值 → 标量 / 名单 → IN / 每行不同标准 → 相关")
    print("  括号叠到两层就升格成 CTE，写出 '读得下去的查询'。")


if __name__ == "__main__":
    main()
