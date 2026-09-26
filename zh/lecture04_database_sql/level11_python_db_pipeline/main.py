"""
Lecture 04 · Level 11 — 连接 Python 与数据管道
这是把 SQL 和 Python 接起来的最后一关。先用安全的本地示例演示参数绑定 (?)
是怎么挡住 SQL 注入的，再用 pandas.read_sql 把查询结果收成 DataFrame，
完成一条 '抽取(SQL) → 加工(pandas) → 装载(CSV)+汇总报表' 的迷你 ETL 管道。
"""

import pathlib
import sqlite3
import sys

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def search_city_unsafe(cur, user_input):
    """[坏例子] 把用户输入当字符串拼进 SQL — 直接暴露在注入风险下。"""
    sql = f"SELECT name, city FROM customers WHERE city = '{user_input}'"
    print(f"  (拼出来的语句) {sql}")
    cur.execute(sql)
    return cur.fetchall()


def search_city_safe(cur, user_input):
    """[好例子] 语句是印好的表格 (?)，值单独传 — 输入永远变不成语句。"""
    sql = "SELECT name, city FROM customers WHERE city = ?"
    print(f"  (语句) {sql}   (值) {user_input!r}")
    cur.execute(sql, (user_input,))     # 就算只有一个值也要写成元组 (别忘了逗号!)
    return cur.fetchall()


def extract(con):
    """[E] 抽取: 用 SQL 先把数据缩小，只取需要的 (只要已完成订单)。"""
    sql = """
        SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
               p.category,
               c.name AS customer,
               oi.quantity * p.price AS amount
        FROM orders AS o
        JOIN customers AS c   ON c.customer_id = o.customer_id
        JOIN order_items AS oi ON oi.order_id = o.order_id
        JOIN products AS p     ON p.product_id = oi.product_id
        WHERE o.status = ?
    """
    return pd.read_sql(sql, con, params=("已完成",))


def transform(df):
    """[T] 加工: 用 pandas 做出月×品类的透视表，并加上月合计列。"""
    pivot = pd.pivot_table(df, values="amount", index="month",
                           columns="category", aggfunc="sum", fill_value=0)
    pivot["月合计"] = pivot.sum(axis=1)
    return pivot


def load_and_report(df, pivot, out_dir):
    """[L] 装载: 把汇总表存成 CSV，并打印一份给管理层看的文字报表。"""
    out_path = out_dir / "monthly_category_revenue.csv"
    pivot.to_csv(out_path, encoding="utf-8-sig")   # 在 Excel 里中文不会乱码的编码
    print(f"  保存完成 → {out_path}")

    total = df["amount"].sum()
    best_month = pivot["月合计"].idxmax()
    best_cat = pivot.drop(columns="月合计").sum().idxmax()
    top3 = df.groupby("customer")["amount"].sum().nlargest(3)
    print("\n  ---- 自动汇总报表 (按已完成订单) ----")
    print(f"  · 年度总销售额        : {total:,} 韩元")
    print(f"  · 销售额最高的月份    : {best_month} ({pivot.loc[best_month, '月合计']:,} 韩元)")
    print(f"  · 销售额第一的品类    : {best_cat}")
    print("  · 消费额前三的客户    :")
    for name, amt in top3.items():
        print(f"      {name}: {amt:,} 韩元")


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"练习数据库准备完成: {db_path.name}\n")

    # ------------------------------------------------------------------
    print("[1] 在 Python 里执行 SQL 的四步 — 连接→游标→执行(绑定)→取结果")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = ?", ("VIP",))
    print(f"  VIP 客户数: {cur.fetchone()[0]} 人 (值 'VIP' 是通过 ? 这个空格传进去的)\n")

    # ------------------------------------------------------------------
    print("[2] SQL 注入演示 — 同一个恶意输入，两种命运 (全程只在本地练习库里)")
    evil = "北京' OR '1'='1"      # 想把条件改成'永远为真'的经典输入
    print(f"  恶意输入: {evil!r}")
    print("  (a) 字符串拼接 (f-string) 的方式:")
    rows = search_city_unsafe(cur, evil)
    print(f"      → 泄露了 {len(rows)} 条! 语句被篡改成了条件'永远为真'。")
    print("  (b) ? 绑定的方式:")
    rows = search_city_safe(cur, evil)
    print(f"      → {len(rows)} 条。整段输入都被当作'一个城市名的值'来对待，")
    print("        而根本没有这样的城市，所以 0 条 — 这才是正确答案。")
    print("  铁律: 只要 SQL 里掺进了用户的值，就无条件用 ? 绑定!\n")

    # ------------------------------------------------------------------
    print("[3] Extract — 用 pandas.read_sql 把连接结果收成 DataFrame")
    df = extract(con)
    print(f"  取到的数据: {df.shape[0]}行 × {df.shape[1]}列 (已完成订单的逐条商品金额)")
    print(df.head(3).to_string(index=False))
    print("  → 缩小数据的活 (WHERE/JOIN) 交给 SQL，整理数据的活交给 pandas。\n")

    # ------------------------------------------------------------------
    print("[4] Transform — 月×品类销售额透视表 (只显示一部分)")
    pivot = transform(df)
    print(pivot.head(4).to_string())
    print()

    # ------------------------------------------------------------------
    print("[5] Load + 报表 — 存成 CSV 并自动汇总")
    out_dir = BASE / "outputs"
    out_dir.mkdir(exist_ok=True)
    load_and_report(df, pivot, out_dir)

    con.close()
    print("\n[小结] 这个脚本本身就是一条管道: 挂到定时执行 (cron 之类) 上，")
    print("  它就变成了'每周一早上自动出报表'。要换成公司的数据库时，")
    print("  只需把 connect 那一段换成对应数据库的驱动，SQL 原样就能用。")


if __name__ == "__main__":
    main()
