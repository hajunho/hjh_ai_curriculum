"""
Lecture 04 · Level 03 — WHERE: 条件查询
把 6 个工作中可能出现的问题 中文 → SQL 地翻译并执行。
覆盖比较运算 (=, >=)、AND/OR 与括号、IN、LIKE、BETWEEN、IS NULL，
并演示错写成 '= NULL' 时为什么会查出 0 条的陷阱。
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


def run(cur, step, question, sql, note=""):
    """按 业务问题 → SQL 语句 → 结果表 的顺序输出的公用执行器。"""
    print(f"[{step}] 业务问题: {question}")
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
    print(f"  → {len(rows)} 条" + (f" | {note}" if note else ""))
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"练习数据库准备完成: {db_path.name}")
    print("WHERE 是逐行判定真/假、只放行 '为真的行' 的过滤器。\n")

    run(cur, "Q1", "把住在北京的 VIP 客户名单给我 (= 和 AND)", """
        SELECT name, city, grade
        FROM customers
        WHERE city = '北京' AND grade = 'VIP'
        LIMIT 6
    """, note="只有 '同时' 满足两个条件的行才通过")

    run(cur, "Q2", "10万韩元以上或者毛利率超过 50% 的商品? (OR 与括号)", """
        SELECT name, price, ROUND(100.0 * (price - cost) / price, 1) AS margin_pct
        FROM products
        WHERE (price >= 100000) OR (100.0 * (price - cost) / price > 50)
    """, note="混进 OR 就用括号把意图钉死的好习惯!")

    run(cur, "Q3", "北京·上海或广州的客户里只要 VIP/GOLD (IN 组合)", """
        SELECT name, city, grade
        FROM customers
        WHERE city IN ('北京', '上海', '广州')
          AND grade IN ('VIP', 'GOLD')
        LIMIT 6
    """, note="IN 是一串 OR 的清爽缩写")

    run(cur, "Q4", "帮我找出姓王的客户 (LIKE 模式)", """
        SELECT name, city
        FROM customers
        WHERE name LIKE '王%'
        LIMIT 6
    """, note="% 是 '任意 0 个以上字符' 的通配符")

    run(cur, "Q5", "三季度(7~9月)的已取消订单明细? (BETWEEN + AND)", """
        SELECT order_id, customer_id, ordered_at, status
        FROM orders
        WHERE ordered_at BETWEEN '2025-07-01' AND '2025-09-30'
          AND status = '已取消'
        LIMIT 6
    """, note="BETWEEN 包含两端的值")

    run(cur, "Q6", "没有上级的员工(组织架构顶端)是谁? (IS NULL)", """
        SELECT employee_id, name, dept
        FROM employees
        WHERE manager_id IS NULL
    """, note="NULL 是 '没有记录' — 使用专用语法 IS NULL")

    # 陷阱演示: 用 = 比较 NULL 永远为假 → 0 条
    run(cur, "Q6-陷阱", "同样的问题错写成 '= NULL' 会怎样?", """
        SELECT employee_id, name, dept
        FROM employees
        WHERE manager_id = NULL
    """, note="0 条! '不知道 = 不知道' 的答案还是不知道，永远成不了真")

    con.close()
    print("[小结] 中文问题里的条件表达和 WHERE 子句一一对应。")
    print("  以上/以下 → >= <= | 其中之一 → IN | 以~开头 → LIKE '..%'")
    print("  期间 → BETWEEN | 没有值 → IS NULL")


if __name__ == "__main__":
    main()
