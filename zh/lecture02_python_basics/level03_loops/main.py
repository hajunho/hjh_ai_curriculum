"""循环语句(for/while/range/break/continue) — 汇总 100 位客户的消费数据。

同一份汇总，我们分别用'手算方式'(一行行全写出来)和'循环方式'来做对比，
确认自动化到底赚了多少。累加变量模式、循环+条件的组合、
while 模拟、break/continue，一直到 enumerate，一次全部拿下。
"""

import random


def main():
    print("=" * 56)
    print(" 循环语句 — 汇总 100 位客户的消费数据")
    print("=" * 56)

    # 为了可复现而固定 seed: 谁来运行都会得到同一份数据
    random.seed(42)

    # 100 位客户本月的消费额(韩元)。其中一部分是 0 元(休眠客户)。
    purchases = []
    for i in range(100):
        if random.random() < 0.15:          # 15% 是休眠客户
            purchases.append(0)
        else:
            purchases.append(random.randint(10, 600) * 1000)  # 1 万~60 万韩元

    # ---------------------------------------------------------
    # [1] 手算方式: 如果没有循环语句呢?
    # ---------------------------------------------------------
    print("\n[1] 如果没有循环语句 (只演示 3 个人的份)")
    total_by_hand = purchases[0] + purchases[1] + purchases[2]
    print(f"  total = purchases[0] + purchases[1] + purchases[2]  # = {total_by_hand:,}韩元")
    print("  ... 换成 100 个人，这样的加法就得写 100 项，或者排 100 行。")
    print("  而客户一变成 101 位，代码也得跟着改。")

    # ---------------------------------------------------------
    # [2] for 循环: 4 行汇总 100 个人
    # ---------------------------------------------------------
    print("\n[2] 用 for 循环来汇总")

    total = 0                       # 累加变量要在循环'之前'初始化
    best_amount = 0                 # 最高消费额
    for amount in purchases:        # 把 100 个人的份一笔笔取出来
        total += amount             # 每转一圈就累加
        if amount > best_amount:    # 刷新最高记录
            best_amount = amount

    average = total / len(purchases)
    print(f"  客户数     : {len(purchases)}位")
    print(f"  消费总额   : {total:,}韩元")
    print(f"  人均       : {average:,.0f}韩元")
    print(f"  最高消费额 : {best_amount:,}韩元")
    print("  -> 就算客户变成 1 万位，上面这段代码一个字都不用改。")

    # ---------------------------------------------------------
    # [3] 循环 + 条件: 筛出优质客户 (用上 continue)
    # ---------------------------------------------------------
    print("\n[3] 筛出优质客户 (if/continue 组合)")

    VIP_THRESHOLD = 300000          # 优质客户标准: 30 万韩元以上
    vip_count = 0
    dormant_count = 0
    for amount in purchases:
        if amount == 0:             # 休眠客户跳过，换下一个人
            dormant_count += 1
            continue
        if amount >= VIP_THRESHOLD:
            vip_count += 1

    print(f"  优质客户(>= {VIP_THRESHOLD:,}韩元) : {vip_count}位")
    print(f"  休眠客户(0 元，用 continue 跳过)   : {dormant_count}位")

    # ---------------------------------------------------------
    # [4] while: 市场预算耗尽模拟 (用上 break)
    # ---------------------------------------------------------
    print("\n[4] while — 100 万韩元预算能撑几天")

    budget = 1000000
    day = 0
    while budget > 0:               # 只要预算还有剩就继续
        day += 1
        spend = 60000 + random.randint(0, 50) * 1000   # 每天支出 6 万~11 万韩元
        budget -= spend
        if day <= 3 or budget <= 0:                    # 只打印前 3 天和最后一天
            print(f"  第{day:>2}天 支出 {spend:>7,}韩元 -> 余额 {max(budget, 0):>9,}韩元")
        if day >= 60:               # 安全装置: 超过 60 天就强制结束
            print("  达到 60 天上限，用 break 结束")
            break

    print(f"  -> 预算在第 {day} 天耗尽。")

    # ---------------------------------------------------------
    # [5] range 与 enumerate: 消费额前 5 名排行榜
    # ---------------------------------------------------------
    print("\n[5] 消费额前 5 名 (用 enumerate 加上排名)")

    top5 = sorted(purchases, reverse=True)[:5]     # 降序排序后取前 5 个
    for rank, amount in enumerate(top5, start=1):  # 排名从 1 开始
        print(f"  第{rank}名: {amount:,}韩元")

    print("\n  确认 range: range(5) ->", list(range(5)), "/ range(1, 6) ->", list(range(1, 6)))

    # ---------------------------------------------------------
    # 结论
    # ---------------------------------------------------------
    print("\n[完] 代码行数对比")
    print("  手算方式 : 行数随客户数一起涨 (100 位 = 100 行+)")
    print("  循环方式 : 永远 4 行 (数据再多，代码照旧)")
    print("  '对列表里的每一项做同样的流程' — 这就是自动化的核心句子。")


if __name__ == "__main__":
    main()
