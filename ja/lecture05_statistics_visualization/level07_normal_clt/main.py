"""
level07 — 正規分布と中心極限定理 (CLT)

1) 68-95-99.7 ルールを乱数 100 万個で検証します。
2) 一様分布・指数分布から標本サイズ n=1/5/30 の標本平均を 5,000 回ずつ抜き取り、
   元の形と無関係に平均の分布が鐘の形へ収束していく過程を
   outputs/clt_grid.png 1 枚 (2x3 グリッド) に保存します。

※ 図中のテキストは、フォントの文字化けを避けるため英語にしています。
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SAMPLE_SIZES = [1, 5, 30]
N_REPEAT = 5_000  # 標本平均を何回抜き取るか


def setup_plot_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False


def rule_68_95_997() -> None:
    """正規分布の乱数 100 万個で 68-95-99.7 ルールを数えます。"""
    rng = np.random.default_rng(707)
    z = rng.normal(0, 1, size=1_000_000)
    print("[1] 68-95-99.7 ルールの検証 (標準正規乱数 1,000,000個)")
    for k, expect in [(1, 68.3), (2, 95.4), (3, 99.7)]:
        ratio = (np.abs(z) <= k).mean() * 100
        print(f"    ±{k}σ 内の比率: {ratio:5.2f}%  (理論値 約 {expect}%)")


def normal_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """正規分布の確率密度関数 (scipy なしで直接)。"""
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def clt_experiment() -> None:
    """一様・指数分布の標本平均の分布をグリッドで描きます。"""
    rng = np.random.default_rng(708)

    # (名前, 図中用の英語名, 標本生成関数, 母平均, 母標準偏差)
    uniform_spec = ("一様分布 U(0,1)", "Uniform U(0,1)",
                    lambda size: rng.uniform(0, 1, size), 0.5, 1 / np.sqrt(12))
    expo_spec = ("指数分布 (平均 1)", "Exponential (mean 1)",
                 lambda size: rng.exponential(1.0, size), 1.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    print()
    print("[2][3] 標本平均の実験 — 各マスは「標本 n 個の平均」を 5,000 回記録した分布")
    print(f"    {'元の分布':<14} {'n':>4} {'平均の平均':>12} {'平均の標準偏差':>14} {'理論値 σ/√n':>12}")

    for row, (name, name_en, sampler, mu, sigma) in enumerate([uniform_spec, expo_spec]):
        for col, n in enumerate(SAMPLE_SIZES):
            # 核心: (N_REPEAT, n) の表を作り、行ごとに平均 -> 標本平均 N_REPEAT 個
            means = sampler((N_REPEAT, n)).mean(axis=1)
            se_theory = sigma / np.sqrt(n)
            print(f"    {name:<14} {n:>4} {means.mean():>12.4f} "
                  f"{means.std():>14.4f} {se_theory:>12.4f}")

            ax = axes[row][col]
            ax.hist(means, bins=50, density=True, color="#9fbce8", edgecolor="white")
            xs = np.linspace(means.min(), means.max(), 200)
            ax.plot(xs, normal_pdf(xs, mu, se_theory), color="#d1495b", lw=2,
                    label="CLT theoretical curve")
            ax.set_title(f"{name_en}, n={n}")
            if col == 0:
                ax.set_ylabel("Density")
            if row == 0 and col == 2:
                ax.legend(fontsize=9)

    fig.suptitle("Central Limit Theorem — whatever the source, the means gather into a bell",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "clt_grid.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print()
    print(f"[4] CLT グリッドを保存: {path}")
    print("    左の列 (n=1) は元の形のまま、右の列 (n=30) は鐘の形に密着します。")
    print("[5] 上の表で「平均の標準偏差」が理論値 σ/√n と一致することを確かめてください。")
    print("    -> 標本を 4 倍集めてはじめて誤差が半分になります (√n の経済学)。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    rule_68_95_997()
    clt_experiment()


if __name__ == "__main__":
    main()
