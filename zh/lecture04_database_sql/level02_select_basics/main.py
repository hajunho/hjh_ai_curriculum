"""
Lecture 04 · Level 02 — SELECT 基础
练习 SQL 的出发点 SELECT/FROM。从所有列 (*)、挑选列、别名 (AS)、
计算列到字符串拼接 — 每一步都先打印 SQL 语句，
紧接着把运行结果输出成表格。SQL 是主角，Python 只是执行器。
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


def run(cur, step, title, sql):
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
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"练习数据库准备完成: {db_path.name}")
    print("阅读要领: 先找 FROM (从哪张表)，再读 SELECT (要哪些列)。\n")

    run(cur, 1, "扫一眼所有列 — 和表打照面就用 *", """
        SELECT *
        FROM customers
        LIMIT 5
    """)

    run(cur, 2, "只挑需要的列 — 在申请单上写明项目", """
        SELECT name, city
        FROM customers
        LIMIT 5
    """)

    run(cur, 3, "别名(AS) — 把结果表的列标题改成报表风格", """
        SELECT name  AS 商品名,
               category AS 品类,
               price AS 售价
        FROM products
        LIMIT 5
    """)

    run(cur, 4, "计算列 — 逐行现场算出毛利和毛利率", """
        SELECT name AS 商品名,
               price AS 售价,
               cost  AS 成本,
               price - cost AS 毛利,
               ROUND(100.0 * (price - cost) / price, 1) AS 毛利率
        FROM products
        LIMIT 5
    """)

    run(cur, 5, "字符串拼接(||) — 造一个 '姓名 (等级)' 展示列", """
        SELECT name || ' (' || grade || ')' AS 客户显示名,
               city AS 城市
        FROM customers
        LIMIT 5
    """)

    con.close()
    print("[6] 小结")
    print("  - SELECT 是只读的: 随便执行什么语句，原件都不会变。")
    print("  - 别名和计算列改变的只是 '显示出来的画面'。")
    print("  - 下一关: 用 WHERE 只挑出 '符合条件的行'。")


if __name__ == "__main__":
    main()
