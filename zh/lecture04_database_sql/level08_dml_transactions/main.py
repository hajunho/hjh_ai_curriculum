"""
Lecture 04 · Level 08 — 数据修改与事务
在练习 INSERT/UPDATE/DELETE 基本功的同时，演示把多个修改捆成
'要么全部成功、要么全部取消' 的事务 (BEGIN/COMMIT/ROLLBACK)。
重头戏是: 用回滚救回没写 WHERE 的 UPDATE 事故，
以及扣减库存事务的成功/失败 (库存不足 → 整体回滚) 两个场景。
"""

import pathlib
import sqlite3
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def sql(cur, statement, params=()):
    """展示 SQL 语句并执行 (本关的主角是修改语句)。"""
    for line in statement.strip().splitlines():
        print(f"  SQL> {line.strip()}")
    cur.execute(statement, params)
    return cur


def show_stock(cur, label):
    """把库存状态汇总成一行输出。"""
    cur.execute("SELECT product_id, stock FROM inventory WHERE product_id IN (1, 2)")
    state = ", ".join(f"商品{pid} 库存={s}" for pid, s in cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM orders")
    print(f"  [{label}] {state}, 订单数={cur.fetchone()[0]}")


def place_order(con, cur, order_id, product_id, qty):
    """扣减库存事务: 创建订单 + 扣库存 + 验证 → 提交或整体回滚。"""
    print(f"  尝试下单: 商品{product_id} × {qty}件 (订单号 {order_id})")
    cur.execute("BEGIN")
    print("  SQL> BEGIN  -- 打开信封: 之后的修改还只是 '铅笔草稿'")
    try:
        sql(cur, "INSERT INTO orders (order_id, customer_id, employee_id, ordered_at, status) "
                 "VALUES (?, ?, 1, '2025-12-30', '已完成')", (order_id, 1))
        sql(cur, "UPDATE inventory SET stock = stock - ? WHERE product_id = ?",
            (qty, product_id))
        # 验证: 扣减后是负数就违反规则 → 整体取消
        cur.execute("SELECT stock FROM inventory WHERE product_id = ?", (product_id,))
        stock_after = cur.fetchone()[0]
        if stock_after < 0:
            raise ValueError(f"库存不足 (扣完会剩 {stock_after} 件)")
        cur.execute("COMMIT")
        print("  SQL> COMMIT  -- 验证通过: 钢笔定稿")
    except Exception as e:
        cur.execute("ROLLBACK")
        print(f"  SQL> ROLLBACK  -- 出了问题({e}) → 信封作废，全当没发生")


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    con.isolation_level = None      # 自动提交模式: 由我们自己用 SQL 控制 BEGIN/COMMIT
    cur = con.cursor()

    # 练习用库存表: 商品 1~10 号各 10 件
    cur.execute("CREATE TABLE inventory (product_id INTEGER PRIMARY KEY, stock INTEGER)")
    cur.executemany("INSERT INTO inventory VALUES (?, ?)", [(i, 10) for i in range(1, 11)])
    print(f"练习数据库准备完成: {db_path.name} (+ inventory 库存表，每件商品 10 件)\n")

    # ------------------------------------------------------------------
    print("[1] INSERT — 添加新客户")
    cur.execute("SELECT COUNT(*) FROM customers")
    print(f"  添加前客户数: {cur.fetchone()[0]}")
    sql(cur, "INSERT INTO customers (customer_id, name, city, grade, joined_at) "
             "VALUES (204, '张雪', '北京', 'BASIC', '2025-12-20')")
    cur.execute("SELECT COUNT(*) FROM customers")
    print(f"  添加后客户数: {cur.fetchone()[0]} → 多了一行\n")

    # ------------------------------------------------------------------
    print("[2] UPDATE — '瞄准(SELECT) → 开枪(UPDATE)' 安全守则")
    sql(cur, "SELECT customer_id, name, grade FROM customers WHERE customer_id = 204")
    print(f"  瞄准结果: {cur.fetchall()} ← 确认正好命中 1 行!")
    sql(cur, "UPDATE customers SET grade = 'GOLD' WHERE customer_id = 204")
    cur.execute("SELECT grade FROM customers WHERE customer_id = 204")
    print(f"  开枪后的等级: {cur.fetchone()[0]}\n")

    # ------------------------------------------------------------------
    print("[3] 忘写 WHERE 的 UPDATE 事故 — 以及 ROLLBACK 橡皮")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    before_vip = cur.fetchone()[0]
    print(f"  事故前 VIP 人数: {before_vip}")
    cur.execute("BEGIN")
    print("  SQL> BEGIN")
    sql(cur, "UPDATE customers SET grade = 'VIP'   -- 忘了写 WHERE!")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    print(f"  事故后 VIP 人数: {cur.fetchone()[0]} ← 全体客户都成了 VIP! (还只是铅笔草稿)")
    cur.execute("ROLLBACK")
    print("  SQL> ROLLBACK")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    print(f"  回滚后 VIP 人数: {cur.fetchone()[0]} → 原样恢复。提交之前橡皮一直都在\n")

    # ------------------------------------------------------------------
    print("[4] 扣减库存事务 — 成功案例 (下单 + 扣减 + 验证 → COMMIT)")
    show_stock(cur, "尝试前")
    place_order(con, cur, order_id=1001, product_id=1, qty=3)
    show_stock(cur, "尝试后")
    print("  → 订单多 1 笔 + 库存 10→7。两个修改一起定稿了。\n")

    # ------------------------------------------------------------------
    print("[5] 扣减库存事务 — 失败案例 (库存只剩 7 件却要买 20 件)")
    show_stock(cur, "尝试前")
    place_order(con, cur, order_id=1002, product_id=1, qty=20)
    show_stock(cur, "尝试后")
    print("  → 订单数和库存都和尝试前一模一样。不存在 '订单存了、库存没扣'")
    print("    这种半截子状态 — 这就是原子性 (要么全部、要么全无)。\n")

    con.close()
    print("[小结] 修改语句的命根子是 WHERE，成组修改的命根子是事务。")
    print("  实战骨架: BEGIN → 一串修改 → 验证 → COMMIT (出问题就 ROLLBACK)。")


if __name__ == "__main__":
    main()
