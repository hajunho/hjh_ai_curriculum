"""
level10 — A/B テストの設計と解釈

コンバージョン率 A/B テストのシミュレーター。
1) 効果がある場合 (3.0% vs 3.6%) とない場合 (A/A) の z 検定
2) 効果のないテストを 2,000 回繰り返し -> 第一種の過誤率が設計どおり 5% かを検証
3) peeking (途中で何度も確認し、有意なら早期終了) -> 偽陽性率の急騰実験

※ 図中のテキストは、フォントの文字化けを避けるため英語にしています。
"""

import os

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def setup_plot_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False


def z_test_two_proportions(conv_a: int, n_a: int, conv_b: int, n_b: int) -> tuple[float, float]:
    """2 つのコンバージョン率の差の z 検定 (直接実装)。(z, p) を返します。"""
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)          # H0: 2 つの比率は同じ
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    p_value = 2 * stats.norm.sf(abs(z))               # 両側検定
    return z, p_value


def run_single_test(rate_a: float, rate_b: float, n: int, seed: int, label: str) -> None:
    """グループあたり n 人のテストを 1 回回して判定します。"""
    rng = np.random.default_rng(seed)
    conv_a = rng.binomial(n, rate_a)
    conv_b = rng.binomial(n, rate_b)
    z, p = z_test_two_proportions(conv_a, n, conv_b, n)
    verdict = "B の勝利を宣言 (有意)" if p < 0.05 else "判断保留 (有意でない)"
    print(f"    {label}: A {conv_a / n:.3%} vs B {conv_b / n:.3%} "
          f"(差 {(conv_b - conv_a) / n:+.3%})")
    print(f"      z = {z:+.2f}, p = {p:.4f} -> {verdict}")


def type1_error_and_peeking() -> None:
    """効果のない (A/A) テスト 2,000 回 — 誠実な検定 vs peeking。"""
    rng = np.random.default_rng(1010)
    n_sims = 2_000
    rate = 0.03
    total_n = 20_000          # グループあたりの最終標本
    step = 1_000              # peeking の確認周期
    checkpoints = np.arange(step, total_n + 1, step)

    honest_fp = 0            # 誠実: 最後に一度だけ検定
    peeking_fp = 0           # のぞき見: 途中で有意なら即終了
    sample_trajectories = []  # 図用の p 値の軌跡をいくつか

    for sim in range(n_sims):
        # step 人ずつ入ってくるたびの累積コンバージョン数 (両グループとも効果なし)
        inc_a = rng.binomial(step, rate, size=len(checkpoints))
        inc_b = rng.binomial(step, rate, size=len(checkpoints))
        cum_a, cum_b = np.cumsum(inc_a), np.cumsum(inc_b)

        p_traj = np.array([
            z_test_two_proportions(ca, n, cb, n)[1]
            for ca, cb, n in zip(cum_a, cum_b, checkpoints)
        ])
        if p_traj[-1] < 0.05:
            honest_fp += 1
        if (p_traj < 0.05).any():
            peeking_fp += 1
        if sim < 12:
            sample_trajectories.append(p_traj)

    print()
    print(f"[3] 第一種の過誤率の検証 — 効果のないテストを {n_sims:,}回繰り返し")
    print(f"    誠実な検定 (最後に 1 回):     偽陽性 {honest_fp:>4}回 "
          f"= {honest_fp / n_sims:.1%}  (設計値 5% の近く)")
    print()
    print(f"[4] peeking — {step:,}人ごとに確認、有意になった瞬間「勝利宣言して終了」")
    print(f"    のぞき見の方針:               偽陽性 {peeking_fp:>4}回 "
          f"= {peeking_fp / n_sims:.1%}  (約 {peeking_fp / max(honest_fp, 1):.1f}倍に急騰!)")
    print("    -> 同じデータ、同じ有意水準なのに、「いつ見るか」だけで誤検知が数倍に")
    print("       なります。終了時点は、結果を見る前に決めなければなりません。")

    # 図: p 値の軌跡 — 0.05 の線を一瞬突き破って戻る「釣られる瞬間」たち
    fig, ax = plt.subplots(figsize=(9, 5))
    for traj in sample_trajectories:
        dipped = (traj < 0.05).any()
        ax.plot(checkpoints, traj, lw=1.6 if dipped else 1.0,
                color="#d1495b" if dipped else "#9aa7b5", alpha=0.85)
    ax.axhline(0.05, color="black", ls="--", lw=1, label="Significance level 0.05")
    ax.set_ylim(0, 1)
    ax.set_title("p-value paths of 12 tests with NO effect — red paths are peeking's prey")
    ax.set_xlabel("Cumulative sample size per group")
    ax.set_ylabel("p-value at that point")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "peeking.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[5] p 値の軌跡の図を保存: {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()

    print("[1] 効果がある場合 — 本当のコンバージョン率 A 3.0% vs B 3.6%、グループあたり 20,000人")
    run_single_test(0.030, 0.036, 20_000, seed=101, label="本テスト")
    print()
    print("[2] 効果がない場合 (A/A) — 両方とも 3.0%、グループあたり 20,000人")
    run_single_test(0.030, 0.030, 20_000, seed=102, label="A/A テスト")

    type1_error_and_peeking()
    print()
    print("[6] 実務のまとめ: 指標・最小効果・標本サイズ・終了ルールを実験の「前」に文書で決め、")
    print("    標本が満ちるまで p 値を見ないことが、最も安上がりな誤検知の防止策です。")


if __name__ == "__main__":
    main()
