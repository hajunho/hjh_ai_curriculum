"""
level08 — 標本と信頼区間

カフェの売上全体を「母集団」とみなし、
1) n=50 の標本で 95% 信頼区間を 100 回作り、本当の平均を何回含むかを数え
2) 標本サイズに応じて区間の幅が sqrt(n) に反比例して狭くなることを確かめ
3) 偏った標本 (週末だけ調査) は標本を大きくしても間違い続けることを示します。

※ 図中のテキストは、フォントの文字化けを避けるため英語にしています。
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def setup_plot_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False


def confidence_interval(sample: np.ndarray, level: float = 0.95):
    """t 分布に基づく信頼区間 (標本が小さくても安全)。"""
    n = len(sample)
    mean = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(n)          # 標準誤差
    t_crit = stats.t.ppf((1 + level) / 2, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def ring_toss(pop: np.ndarray, mu: float) -> None:
    """[2][3] 輪投げ: 信頼区間 100 個のうち、いくつが mu を含むか。"""
    rng = np.random.default_rng(808)
    n_trials, n_sample = 100, 50
    intervals, hits = [], 0
    for _ in range(n_trials):
        sample = rng.choice(pop, size=n_sample, replace=False)
        lo, hi = confidence_interval(sample)
        contains = lo <= mu <= hi
        hits += contains
        intervals.append((lo, hi, contains))

    print()
    print(f"[2] 輪投げ — n={n_sample} の標本で 95% 信頼区間を {n_trials} 回製作")
    print(f"    本当の平均 μ を含んだ区間: {hits}/{n_trials} = {hits / n_trials:.0%}")
    print("    -> 「95%」は区間 1 つの確率ではなく、「この製作方法」の長期的な的中率です。")

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (lo, hi, ok) in enumerate(intervals):
        color = "#9aa7b5" if ok else "#d1495b"
        ax.plot([lo / 1e4, hi / 1e4], [i, i], color=color, lw=1.6)
    ax.axvline(mu / 1e4, color="#2e6f40", lw=2, label=f"True mean = {mu / 1e4:.1f} (10k KRW)")
    ax.set_title(f"{n_trials - hits} of {n_trials} intervals missed the true mean (red)")
    ax.set_xlabel("Revenue per sale (10k KRW)")
    ax.set_ylabel("Trial #")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ci_rings.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 輪投げの図を保存: {path}")


def width_vs_n(pop: np.ndarray) -> None:
    """[4] 標本サイズごとの区間の幅 — sqrt(n) 反比例の確認。"""
    rng = np.random.default_rng(809)
    sizes = [20, 50, 200, 800]
    widths = []
    print()
    print("[4] 標本サイズと区間の幅 (幅を半分に縮めるには標本 4 倍)")
    for n in sizes:
        # 同じサイズで 200 回繰り返して平均の幅を測定 (1 回きりの偶然を排除)
        w = [np.subtract(*confidence_interval(rng.choice(pop, n, replace=False))[::-1])
             for _ in range(200)]
        widths.append(np.mean(w))
        print(f"    n = {n:>4}: 平均区間幅 {np.mean(w) / 1e4:8.2f} 万ウォン")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, [w / 1e4 for w in widths], marker="o", color="#4878cf", label="Measured width")
    ref = widths[0] * np.sqrt(sizes[0]) / np.sqrt(np.array(sizes))
    ax.plot(sizes, ref / 1e4, ls="--", color="#d1495b", label="Theory: 1/sqrt(n) curve")
    ax.set_title("CI width shrinks in proportion to 1/sqrt(n)")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("95% CI width (10k KRW)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ci_width.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    幅の図を保存: {path}")


def biased_sampling(df: pd.DataFrame, mu: float) -> None:
    """[5] 偏った標本: 週末だけ調査すると、標本を大きくしても間違い続けます。"""
    rng = np.random.default_rng(810)
    weekend = df[df["weekday"].isin(["土", "日"])]["revenue"].to_numpy()
    print()
    print("[5] バイアスの実験 — 「週末にだけ」調査した標本の信頼区間")
    for n in [50, 400, 2000]:
        # 週末の客だけを調査し続ける状況 (復元抽出で調査規模の拡大を模倣)
        sample = rng.choice(weekend, size=n, replace=True)
        lo, hi = confidence_interval(sample)
        verdict = "含む" if lo <= mu <= hi else "外す!"
        print(f"    n = {n:>4}: [{lo / 1e4:7.1f}, {hi / 1e4:7.1f}] 万ウォン -> μ を{verdict}")
    print(f"    (本当の μ = {mu / 1e4:.1f} 万ウォン)")
    print("    -> 区間はどんどん狭く自信満々になりますが、一貫して間違っています。")
    print("       バイアスは標本サイズでは解決しません。「どう抽出したか」が先です。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()

    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    pop = df["revenue"].to_numpy(dtype=float)
    mu = pop.mean()
    print(f"[1] 母集団の準備: 売上 {len(pop):,}件、本当の平均 μ = {mu / 1e4:.1f} 万ウォン")
    print("    (現実では μ は未知数ですが、今日は実験のために知った上で始めます)")

    ring_toss(pop, mu)
    width_vs_n(pop)
    biased_sampling(df, mu)


if __name__ == "__main__":
    main()
