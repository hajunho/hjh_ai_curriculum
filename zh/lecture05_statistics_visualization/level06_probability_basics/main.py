"""
level06 — 概率的基础

用三个模拟让直觉与计算正面对决。
1) 大数定律: 抛硬币比例的收敛
2) 疾病检测悖论: 阳性判定者中真病人占百分之几 (贝叶斯尝鲜)
3) 蒙提霍尔问题: 换门 vs 不换策略的胜率
"""

import numpy as np


def law_of_large_numbers() -> None:
    """抛硬币次数越多，正面比例越逼近 0.5。"""
    rng = np.random.default_rng(606)  # 固定 seed
    flips = rng.integers(0, 2, size=10_000)  # 1=正面
    print("[1] 大数定律 — 硬币正面比例的收敛")
    for n in [10, 100, 1_000, 10_000]:
        ratio = flips[:n].mean()
        print(f"    抛 {n:>6,} 次: 正面比例 {ratio:.4f} (与 0.5 的距离 {abs(ratio - 0.5):.4f})")
    print("    -> 概率 0.5 不是对'下一次'的预言，而是对'长期比例'的承诺。")


def disease_test_paradox() -> None:
    """对 10 万人实施患病率 1%、灵敏度 99%、特异度 95% 的检测。"""
    rng = np.random.default_rng(607)
    n = 100_000
    prevalence = 0.01     # 患病率: 患病的先验概率
    sensitivity = 0.99    # P(阳性|病) — 抓出病人的概率
    specificity = 0.95    # P(阴性|健康) — 放行健康人的概率

    sick = rng.random(n) < prevalence
    # 病人按灵敏度概率呈阳性，健康人按 (1-特异度) 概率被冤枉成阳性
    positive = np.where(sick,
                        rng.random(n) < sensitivity,
                        rng.random(n) < (1 - specificity))

    n_pos = positive.sum()
    n_true = (sick & positive).sum()          # 真病人的阳性
    n_false = (~sick & positive).sum()        # 被冤枉的阳性(误报)
    p_sick_given_pos = n_true / n_pos

    # 与贝叶斯定理算出的理论值比较
    theory = (prevalence * sensitivity) / (
        prevalence * sensitivity + (1 - prevalence) * (1 - specificity))

    print()
    print("[2] 疾病检测悖论 — 100,000 人虚拟人口模拟")
    print(f"    病人 {sick.sum():,}名 / 健康 {(~sick).sum():,}名 (患病率 {prevalence:.0%})")
    print(f"    阳性判定 {n_pos:,}名 = 真病人 {n_true:,}名 + 被冤枉的阳性 {n_false:,}名")
    print(f"    P(病 | 阳性) 模拟值 = {p_sick_given_pos:.3f}")
    print(f"    P(病 | 阳性) 贝叶斯理论值 = {theory:.3f}")
    print("    -> 就算在'99% 准确的检测'里呈阳性，真患病的概率也只有约 17%!")
    print("       健康人占绝对多数，他们那 5% 的误报就压倒了病人总数。")
    print("       (P(阳性|病)=0.99 和 P(病|阳性)=0.17 是完全不同的概率)")


def monty_hall(n_games: int = 10_000) -> None:
    """蒙提霍尔: 三扇门，主持人知道并打开空门。"""
    rng = np.random.default_rng(608)
    prize = rng.integers(0, 3, size=n_games)       # 藏着奖品的门
    first_pick = rng.integers(0, 3, size=n_games)  # 参赛者的第一次选择

    # '不换'策略: 第一次选中即获胜
    stay_wins = (first_pick == prize).sum()
    # '换门'策略: 第一次选择'不是'正确答案就获胜
    #   (主持人替你打开了剩下的空门，换过去必然是正确的那扇)
    switch_wins = (first_pick != prize).sum()

    print()
    print(f"[3] 蒙提霍尔问题 — {n_games:,} 局模拟")
    print(f"    不换策略 胜率: {stay_wins / n_games:.3f} (理论值 1/3 = 0.333)")
    print(f"    换门策略 胜率: {switch_wins / n_games:.3f} (理论值 2/3 = 0.667)")
    print("    -> 主持人可以'挑着空门'打开，这个动作本身就是信息。")
    print("       直觉再怎么坚持五五开，一万局的账本都说是 2/3。")


def main() -> None:
    law_of_large_numbers()
    disease_test_paradox()
    monty_hall()
    print()
    print("[4] 今日教训: 概率吵起来的时候，先别搬公式，先跑模拟。")
    print("    '造出来数一数'是永远好使的概率计算器。")


if __name__ == "__main__":
    main()
