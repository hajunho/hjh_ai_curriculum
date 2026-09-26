"""
level06 — 確率の基礎

3 つのシミュレーションで、直感と計算を対決させます。
1) 大数の法則: コイン投げの比率の収束
2) 病気検査のパラドックス: 陽性判定者のうち本当の病人は何 % か (ベイズの味見)
3) モンティ・ホール問題: 替える vs 維持する戦略の勝率
"""

import numpy as np


def law_of_large_numbers() -> None:
    """コイン投げの回数が増えるほど、表の比率が 0.5 に近づきます。"""
    rng = np.random.default_rng(606)  # seed 固定
    flips = rng.integers(0, 2, size=10_000)  # 1=表
    print("[1] 大数の法則 — コインの表の比率の収束")
    for n in [10, 100, 1_000, 10_000]:
        ratio = flips[:n].mean()
        print(f"    {n:>6,}回投げ: 表の比率 {ratio:.4f} (0.5 との距離 {abs(ratio - 0.5):.4f})")
    print("    -> 確率 0.5 は「次の 1 回」の予言ではなく、「長期的な比率」の約束です。")


def disease_test_paradox() -> None:
    """有病率 1%、感度 99%、特異度 95% の検査を 10 万人に実施します。"""
    rng = np.random.default_rng(607)
    n = 100_000
    prevalence = 0.01     # 有病率: 病気である事前確率
    sensitivity = 0.99    # P(陽性|病気) — 病人を捕まえる確率
    specificity = 0.95    # P(陰性|健康) — 健康な人を通過させる確率

    sick = rng.random(n) < prevalence
    # 病人は感度の確率で陽性、健康な人は (1-特異度) の確率で濡れ衣の陽性
    positive = np.where(sick,
                        rng.random(n) < sensitivity,
                        rng.random(n) < (1 - specificity))

    n_pos = positive.sum()
    n_true = (sick & positive).sum()          # 本当に病気の陽性
    n_false = (~sick & positive).sum()        # 濡れ衣の陽性 (誤検知)
    p_sick_given_pos = n_true / n_pos

    # ベイズの定理で求めた理論値と比較
    theory = (prevalence * sensitivity) / (
        prevalence * sensitivity + (1 - prevalence) * (1 - specificity))

    print()
    print("[2] 病気検査のパラドックス — 仮想人口 100,000 人のシミュレーション")
    print(f"    病人 {sick.sum():,}人 / 健康 {(~sick).sum():,}人 (有病率 {prevalence:.0%})")
    print(f"    陽性判定 {n_pos:,}人 = 本当の病人 {n_true:,}人 + 濡れ衣の陽性 {n_false:,}人")
    print(f"    P(病気 | 陽性) シミュレーション = {p_sick_given_pos:.3f}")
    print(f"    P(病気 | 陽性) ベイズの理論値 = {theory:.3f}")
    print("    -> 「99% 正確な検査」で陽性でも、実際に病気である確率は約 17%!")
    print("       健康な人が圧倒的に多いため、その 5% の誤検知が病人の数を圧倒します。")
    print("       (P(陽性|病気)=0.99 と P(病気|陽性)=0.17 はまったく別の確率です)")


def monty_hall(n_games: int = 10_000) -> None:
    """モンティ・ホール: ドア 3 つ、司会者はハズレのドアを知っていて開けてくれます。"""
    rng = np.random.default_rng(608)
    prize = rng.integers(0, 3, size=n_games)       # 賞品の入ったドア
    first_pick = rng.integers(0, 3, size=n_games)  # 参加者の最初の選択

    # 「維持」戦略: 最初の選択が正解なら勝ち
    stay_wins = (first_pick == prize).sum()
    # 「替える」戦略: 最初の選択が正解で「なければ」勝ち
    #   (司会者が残りのハズレを開けてくれるので、替えれば必ず残ったドア = 正解)
    switch_wins = (first_pick != prize).sum()

    print()
    print(f"[3] モンティ・ホール問題 — {n_games:,}回のシミュレーション")
    print(f"    維持する戦略 勝率: {stay_wins / n_games:.3f} (理論値 1/3 = 0.333)")
    print(f"    替える戦略   勝率: {switch_wins / n_games:.3f} (理論値 2/3 = 0.667)")
    print("    -> 司会者は「ハズレを選んで」開けられるので、その行動自体が情報です。")
    print("       直感が 50:50 だと言い張っても、1 万回の帳簿は 2/3 だと語ります。")


def main() -> None:
    law_of_large_numbers()
    disease_test_paradox()
    monty_hall()
    print()
    print("[4] 今日の教訓: 確率の議論が紛糾したら、公式より先にシミュレーションを回してみましょう。")
    print("    「作って数えてみる」は、いつでも通用する確率計算機です。")


if __name__ == "__main__":
    main()
