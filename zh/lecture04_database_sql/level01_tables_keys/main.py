"""
Lecture 04 · Level 01 — 表、行、列与主键
创建练习用网店数据库 (hjh_shop.db)，然后让数据库自己介绍自己的结构。
按 表清单 → 各表的列和主键 → 行数 → 外键关系图 的顺序，
练习阅读结构 (schema) 的要领。以后面对陌生的公司数据库，
照这个顺序摸清它就行。
"""

import pathlib
import sqlite3
import sys
import unicodedata

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def disp_width(text):
    """中文字符在屏幕上占 2 格，这里计算表格对齐用的显示宽度。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width):
    return str(text) + " " * (width - disp_width(text))


def show_table(cols, rows):
    """把查询结果打印成列宽对齐的表格。"""
    widths = [max(disp_width(c), *(disp_width(r[i]) for r in rows)) if rows else disp_width(c)
              for i, c in enumerate(cols)]
    print("  " + " | ".join(pad(c, w) for c, w in zip(cols, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))


def run(cur, sql):
    """先展示 SQL 语句，执行后把结果打印成表格。"""
    print(f"  SQL> {sql}")
    cur.execute(sql)
    show_table([d[0] for d in cur.description], cur.fetchall())
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"练习数据库准备完成: {db_path.name}\n")

    # ------------------------------------------------------------------
    print("[1] 表清单 — 这个数据库里有哪些账本 (表)?")
    run(cur, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")

    # ------------------------------------------------------------------
    print("[2] 各表的结构 — 列名·数据类型·主键 (pk 为 1 以上就是主键)")
    tables = ["customers", "products", "employees", "orders", "order_items"]
    for t in tables:
        print(f"  SQL> PRAGMA table_info({t})")
        cur.execute(f"PRAGMA table_info({t})")
        rows = [(r[1], r[2], "PK" + str(r[5]) if r[5] else "") for r in cur.fetchall()]
        show_table(["column", "type", "key"], rows)
        print()

    # ------------------------------------------------------------------
    print("[3] 各表的行数 — 确认数据规模")
    counts = []
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")   # COUNT(*) 用来数行数
        counts.append((t, cur.fetchone()[0]))
    show_table(["table", "rows"], counts)
    print()

    # ------------------------------------------------------------------
    print("[4] 外键关系图 — _id 列指向哪张表的主键")
    relations = [
        ("orders.customer_id",      "→ customers.customer_id", "订单的主人客户"),
        ("orders.employee_id",      "→ employees.employee_id", "处理订单的员工"),
        ("order_items.order_id",    "→ orders.order_id",       "属于哪笔订单的明细"),
        ("order_items.product_id",  "→ products.product_id",   "是哪件商品"),
        ("employees.manager_id",    "→ employees.employee_id", "直属上级(自引用)"),
    ]
    show_table(["foreign key", "references", "meaning"], relations)
    print("""
  customers ──< orders >── employees
                  │            └──(manager_id 自引用)
                  └──< order_items >── products
  (──< 表示 1:N 关系 — 1 位客户拥有 N 笔订单)
""")

    # ------------------------------------------------------------------
    print("[5] 引用验证 — 订单账本上的 customer_id 就是客户名册的行号")
    run(cur, "SELECT order_id, customer_id, ordered_at, status "
             "FROM orders WHERE customer_id = 7 LIMIT 3")
    run(cur, "SELECT customer_id, name, city, grade "
             "FROM customers WHERE customer_id = 7")
    print("  → 订单账本上不写姓名，只写 '客户名册 7 号' 这个引用。")
    print("    这就是客户信息变了也只需改 customers 一行的原因。")

    con.close()
    print("\n[6] 小结: 面对陌生数据库，按 (1)表清单 (2)列和主键 (3)行数")
    print("    (4)外键关系 的顺序摸清结构。下一关正式开始查询 (SELECT)!")


if __name__ == "__main__":
    main()
