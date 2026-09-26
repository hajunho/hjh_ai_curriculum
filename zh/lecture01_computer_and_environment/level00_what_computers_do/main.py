"""
Lecture 01 / Level 00 — 计算机到底是一台做什么的机器

以"输入 -> 计算 -> 输出"的流程体验计算机的干活方式。
把 CPU (业务骨干)、内存 (办公桌)、存储设备 (档案库) 的分工，
比作咖啡连锁总部的销售汇总业务来观察。
只使用标准库，不需要联网。
"""

import json
import tempfile
import time
from pathlib import Path

# 重复计算次数 ("动手试试"第 1 题请修改这个值)
LOOP_COUNT = 1_000_000


def receive_orders():
    """[输入] 接待窗口: 各门店的订单数据进来了。"""
    # (门店名, 订单数, 总销售额[韩元]) — 代码里现造的虚构数据。
    orders = [
        ("朝阳店", 182, 1_512_000),
        ("海淀店", 141, 1_098_000),
        ("浦东店", 210, 1_745_000),
        ("天河店", 95, 702_000),
    ]
    return orders


def save_to_storage(orders, path):
    """[存储设备] 归档进档案库: 存成文件后，断电也不会丢。"""
    rows = [{"store": s, "count": c, "revenue": r} for s, c, r in orders]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.stat().st_size  # 档案库里这份文件的大小 (字节)


def load_from_storage(path):
    """[内存] 把档案库的文件摊到桌上: 读取文件、载入内存。"""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(row["store"], row["count"], row["revenue"]) for row in rows]


def process_orders(orders):
    """[计算] CPU 职员: 接收输入、做计算、返回结果。(IPO 模型)"""
    total_revenue = sum(revenue for _, _, revenue in orders)
    total_count = sum(count for _, count, _ in orders)
    avg_per_order = total_revenue / total_count
    best_store = max(orders, key=lambda row: row[2])
    return {
        "total_revenue": total_revenue,
        "total_count": total_count,
        "avg_per_order": avg_per_order,
        "best_store": best_store[0],
        "best_revenue": best_store[2],
    }


def measure_cpu_speed():
    """实测 CPU 重复简单计算到底有多快。"""
    started = time.perf_counter()
    acc = 0
    for i in range(LOOP_COUNT):
        acc += i  # 不停地重复一个极小的加法
    elapsed = time.perf_counter() - started
    return elapsed, acc


def show_binary(text):
    """展示字符在计算机内部是如何用 0 和 1 表示的。"""
    for ch in text:
        code = ord(ch)  # 每个字符约定好的编号 (Unicode)
        print(f"    字符 '{ch}' -> 编号 {code} -> 二进制 {code:08b}")


def main():
    print("=" * 60)
    print("计算机公司(股份) 业务流程体验 — 输入 -> 计算 -> 输出")
    print("=" * 60)

    # [1] 输入: 订单数据从接待窗口进来。
    orders = receive_orders()
    print(f"\n[1] 输入(Input): 接待窗口收到 {len(orders)} 家门店的订单")
    for store, count, revenue in orders:
        print(f"    - {store}: {count}单, {revenue:,}韩元")

    # [2] 存储设备 <-> 内存: 先归档进档案库，再取回桌面。
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "orders_archive.json"
        size = save_to_storage(orders, archive)
        print(f"\n[2] 存储设备(档案库): 已存为 '{archive.name}' 文件 ({size}字节)")
        print("    - 存成文件的内容断电后依然保留 (非易失性)")
        orders_on_desk = load_from_storage(archive)
        print(f"    - 从档案库取回、摊到办公桌(内存)上: 恢复 {len(orders_on_desk)} 条")
        print("    - 内存里的数据在程序结束后就消失 (易失性)")

    # [3] 计算: CPU 职员算出合计与平均。
    report = process_orders(orders_on_desk)
    elapsed, _ = measure_cpu_speed()
    print(f"\n[3] 计算(Process): 体验 CPU 职员的审批速度")
    print(f"    - 重复 {LOOP_COUNT:,} 次简单加法耗时: {elapsed:.3f}秒")
    print(f"    - 折合每秒约 {LOOP_COUNT / elapsed:,.0f} 次 — 人类连模仿都做不到")

    # [4] 输出: 以人读起来舒服的报表寄出。
    print(f"\n[4] 输出(Output): 销售汇总报表")
    print(f"    - 总销售额          : {report['total_revenue']:,}韩元")
    print(f"    - 总订单数          : {report['total_count']:,}单")
    print(f"    - 单均销售额        : {report['avg_per_order']:,.0f}韩元")
    print(f"    - 销售额最高门店    : {report['best_store']} ({report['best_revenue']:,}韩元)")

    # [5] 二进制初体验: 一切信息最终都是 0 和 1。
    print(f"\n[5] 二进制初体验: 'AI' 这两个字符的真实存储样貌")
    show_binary("AI")

    print("\n小结: 无论什么程序，都拆成'输入 -> 计算 -> 输出'三块来看。")
    print("      CPU=职员，内存=办公桌(快·易失)，存储设备=档案库(慢·永久)。")


if __name__ == "__main__":
    main()
