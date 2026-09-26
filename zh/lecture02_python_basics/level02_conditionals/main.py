"""条件语句(if/elif/else) — 出差费用自动审批判定器。

我们把公司的费用报销规定搬进 if/elif/else 阶梯，让它自动做判定。
确认比较运算符(==, <=, ...)和逻辑运算符(and/or/not)给出的 True/False，
并用实验证明: 条件的'顺序'就是规定的优先级。
"""


def judge_expense(amount, has_receipt):
    """按公司规定判定 1 笔费用。

    规定(从上往下，只适用第一条命中的条款):
      1) 5 万韩元以下              -> 自动批准
      2) 15 万韩元以下 + 有发票    -> 组长审批
      3) 50 万韩元以下             -> 部门负责人审批
      4) 其他                      -> 驳回(要求说明)
    """
    if amount <= 50000:                       # 条款 1
        return "自动批准"
    elif amount <= 150000 and has_receipt:    # 条款 2 (and: 两个都为真才行)
        return "组长审批"
    elif amount <= 500000:                    # 条款 3
        return "部门负责人审批"
    else:                                     # 条款 4: 以上都不命中
        return "驳回(要求说明)"


def judge_wrong_order(amount, has_receipt):
    """故意把'宽松的条件放在前面'的错误规定。用来对比顺序的重要性。"""
    if amount <= 500000:                      # 宽松的条件放到最上面的话……
        return "部门负责人审批"
    elif amount <= 150000 and has_receipt:    # 这一条永远执行不到!
        return "组长审批"
    elif amount <= 50000:                     # 这一条也一样
        return "自动批准"
    else:
        return "驳回(要求说明)"


def main():
    print("=" * 56)
    print(" 条件语句 — 出差费用自动审批判定器")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 单笔判定: 把 1 笔费用送进规定里跑一遍
    # ---------------------------------------------------------
    print("\n[1] 单笔判定")
    amount = 120000          # 申请金额(韩元)
    has_receipt = True       # 是否提交了发票

    print(f"  申请金额 {amount:,}韩元 / 发票{'有' if has_receipt else '无'}")
    print(f"  -> 判定: {judge_expense(amount, has_receipt)}")
    print("  (超过 5 万所以条款1 落选 -> 15 万以下+有发票，在条款2 定案，后面的条款不再看)")

    # ---------------------------------------------------------
    # [2] 比较·逻辑运算符实验室
    # ---------------------------------------------------------
    print("\n[2] 比较·逻辑运算符的 True/False")
    print(f"  amount <= 150000        -> {amount <= 150000}")
    print(f"  amount == 120000        -> {amount == 120000}")
    print(f"  amount != 120000        -> {amount != 120000}")
    print(f"  amount <= 150000 and has_receipt -> {amount <= 150000 and has_receipt}")
    print(f"  amount <= 50000 or has_receipt   -> {amount <= 50000 or has_receipt}")
    print(f"  not has_receipt         -> {not has_receipt}")
    print(f"  50000 < amount <= 150000 (区间比较) -> {50000 < amount <= 150000}")
    grade = "VIP"
    print(f"  grade in ('VIP','VVIP') -> {grade in ('VIP', 'VVIP')}")

    # ---------------------------------------------------------
    # [3] 条件'顺序'的重要性: 同样的条款，不同的顺序
    # ---------------------------------------------------------
    print("\n[3] 顺序一换，规定就废了")
    test_amount = 30000      # 本来应该是'自动批准'的金额
    ok = judge_expense(test_amount, True)
    bad = judge_wrong_order(test_amount, True)
    print(f"  3 万韩元的费用，正确顺序(严格条件在前) -> {ok}")
    print(f"  3 万韩元的费用，错误顺序(宽松条件在前) -> {bad}")
    print("  -> 宽松的条件在上面，下面那些严格的条款就永远执行不到。")

    # ---------------------------------------------------------
    # [4] 批量判定: 用同一套规定处理 6 笔申请 (循环语句初体验)
    # ---------------------------------------------------------
    print("\n[4] 本周 6 笔费用申请批量判定")

    # (申请人, 金额, 是否有发票)
    requests = [
        ("王主管", 32000, True),
        ("李经理", 120000, True),
        ("张助理", 120000, False),   # 金额相同、没有发票 -> 结果就不一样
        ("陈总监", 480000, True),
        ("赵专员", 750000, True),
        ("刘助理", 50000, False),
    ]

    approved = 0     # 自动批准的笔数
    escalated = 0    # 需要审批的笔数
    rejected = 0     # 驳回的笔数

    for name, amt, receipt in requests:
        decision = judge_expense(amt, receipt)
        print(f"  {name} | {amt:>8,}韩元 | 发票 {'O' if receipt else 'X'} -> {decision}")
        if decision == "自动批准":
            approved += 1
        elif decision == "驳回(要求说明)":
            rejected += 1
        else:
            escalated += 1

    print(f"\n  汇总: 自动批准 {approved}笔 / 需要审批 {escalated}笔 / 驳回 {rejected}笔")
    print("\n[完] 规定文件里的条款 = if/elif/else 的分支。条款顺序 = 条件顺序。")


if __name__ == "__main__":
    main()
