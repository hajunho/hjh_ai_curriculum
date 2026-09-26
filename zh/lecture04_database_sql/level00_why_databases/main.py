"""
Lecture 04 · Level 00 — 为什么需要数据库
用 Python 模拟重现 "Excel 文件传阅" 的三大事故 (并发修改丢失、一致性污染、
权限缺失)，再用数据库 (sqlite3) 方式重演同样的剧本，对比结果有什么不同。
SQL 语法从下一关才开始学，这里只需要关注输出结果的 "差异" 就够了。
"""

import copy
import pathlib
import sqlite3
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data  # 公用数据生成器 (帮我们创建练习用网店数据库)


def show_rows(title, rows):
    """把字典列表以简单表格形式打印出来。"""
    print(f"  {title}")
    for r in rows:
        print("   ", " | ".join(f"{k}={v}" for k, v in r.items()))


def excel_style_disaster():
    """[1] 并发修改: 两个人各自修改同一个文件的副本再先后保存，会怎样？"""
    print("[1] Excel 方式 — 并发修改事故 (更新丢失)")
    shared_file = [  # 请把它想象成共享文件夹里的 '客户名单.xlsx'
        {"客户编号": 1, "姓名": "王志明", "等级": "SILVER"},
        {"客户编号": 2, "姓名": "李秀英", "等级": "BASIC"},
    ]
    # 两个人各自 '下载' 文件 (生成副本)
    kim_copy = copy.deepcopy(shared_file)
    park_copy = copy.deepcopy(shared_file)

    kim_copy[0]["等级"] = "VIP"      # 小王: 把 1 号客户升级为 VIP
    park_copy[1]["等级"] = "GOLD"    # 李经理: 把 2 号客户升级为 GOLD

    shared_file = kim_copy    # 小王先保存 (整个文件覆盖)
    shared_file = park_copy   # 李经理后保存 → 小王的工作消失了!

    show_rows("最终保存下来的文件:", shared_file)
    print("  → 1 号客户还是 SILVER。小王的升级操作 '没有任何报错' 就蒸发了。\n")


def integrity_disaster():
    """[2] 一致性: 同一家往来单位仅仅因为写法不同被录入了两次的账本。"""
    print("[2] Excel 方式 — 一致性污染 (同一单位，不同写法)")
    dirty_ledger = [
        {"往来单位": "华彩有限公司", "金额": 300},
        {"往来单位": "华彩公司", "金额": 200},   # 其实和上面是同一家公司
        {"往来单位": "未来商社", "金额": 150},
    ]
    totals = {}
    for row in dirty_ledger:
        totals[row["往来单位"]] = totals.get(row["往来单位"], 0) + row["金额"]
    for name, amount in totals.items():
        print(f"    {name}: {amount}万韩元")
    print("  → 华彩其实是家 500 万韩元的往来单位，却被拆成 300/200，报表就错了。\n")


def permission_disaster():
    """[3] 权限: 文件发出去的那一刻，敏感的列也被整个带出去了。"""
    print("[3] Excel 方式 — 权限缺失 (发文件 = 全部泄露)")
    hr_file = [
        {"姓名": "张欣怡", "部门": "销售", "年薪": 9000},
        {"姓名": "刘子豪", "部门": "研发", "年薪": 7200},
    ]
    print("  你写着 '仅供参考部门情况' 把文件发出去，对方屏幕上看到的是:")
    show_rows("被发送的文件内容:", hr_file)
    print("  → 想去掉年薪列，唯一的办法是 '另做一个文件'。原件根本无法管控。\n")


def database_way():
    """[4] 数据库方式: 一处原件 + 请求柜台。把同样的剧本再演一遍。"""
    print("[4] 数据库方式 — 同样的剧本，不同的结局")
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))   # 创建练习用网店数据库 (无需安装)
    print(f"  练习数据库已生成: {db_path.name} (customers 等 5 张表)")

    # 两个连接 = 两位用户。原件只有一份，各自只发送 '请求'。
    kim = sqlite3.connect(db_path)
    park = sqlite3.connect(db_path)

    kim.execute("UPDATE customers SET grade='VIP' WHERE customer_id=1")
    kim.commit()      # 小王: 请求升级 1 号客户 → 柜台记入原件
    park.execute("UPDATE customers SET grade='GOLD' WHERE customer_id=2")
    park.commit()     # 李经理: 请求升级 2 号客户 → 同样记入原件

    cur = kim.execute(
        "SELECT customer_id, name, grade FROM customers WHERE customer_id IN (1, 2)")
    rows = [{"客户编号": r[0], "姓名": r[1], "等级": r[2]} for r in cur.fetchall()]
    show_rows("原件表格的当前状态:", rows)
    print("  → 两个人的修改都保留下来了。原件只有一份，所以不存在覆盖事故。")

    # 一致性: 违反规则 (约束条件) 的数据，连保存这一步都会被拒绝。
    try:
        kim.execute("INSERT INTO customers (customer_id, name) VALUES (1, '幽灵客户')")
    except sqlite3.IntegrityError as e:
        print(f"  尝试保存重复的客户编号 → 数据库拒绝: {e}")

    # 权限: 可以只开放一扇去掉敏感列的 '只读窗口 (视图)'。
    kim.execute("CREATE VIEW IF NOT EXISTS emp_public AS "
                "SELECT name, dept FROM employees")   # 干脆不包含 salary 列
    cur = kim.execute("SELECT * FROM emp_public LIMIT 2")
    rows = [{"姓名": r[0], "部门": r[1]} for r in cur.fetchall()]
    show_rows("通过共享窗口 (视图) 能看到的内容:", rows)
    print("  → 薪资列在窗口里根本不存在，从源头上杜绝了泄露。\n")

    kim.close()
    park.close()


def main():
    print("=" * 62)
    print(" Excel 传阅的三重灾难 vs 数据库")
    print("=" * 62 + "\n")
    excel_style_disaster()
    integrity_disaster()
    permission_disaster()
    database_way()
    print("[5] 小结")
    print("  Excel 方式: 副本满天飞 → 出了事故连 '报错' 都没有。")
    print("  数据库方式: 一份原件 + 请求柜台 → 顺序保证、规则检查、权限管理。")
    print("  从下一关开始，我们学习和这个柜台对话的语言 — SQL。")


if __name__ == "__main__":
    main()
