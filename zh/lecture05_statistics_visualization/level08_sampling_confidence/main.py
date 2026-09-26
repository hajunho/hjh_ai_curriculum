"""
level08 — 样本与置信区间

把咖啡店全部销售记录当作'总体'，
1) 用 n=50 的样本做 100 个 95% 置信区间，数几个包含真实平均
2) 确认区间宽度随样本量按 sqrt(n) 成反比变窄
3) 展示有偏样本(只调查周末)就算加大样本也一直错。

※ 图内文字统一用英文，避免因环境缺字体而乱码。
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
    """基于 t 分布的置信区间 (样本小也安全)。"""
    n = len(sample)
    mean = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(n)          # 标准误
    t_crit = stats.t.ppf((1 + level) / 2, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def ring_toss(pop: np.ndarray, mu: float) -> None:
    """[2][3] 套圈: 100 个置信区间里有几个包含 mu。"""
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
    print(f"[2] 套圈 — 用 n={n_sample} 的样本制作 95% 置信区间 {n_trials} 次")
    print(f"    包含真实平均 μ 的区间: {hits}/{n_trials} = {hits / n_trials:.0%}")
    print("    -> '95%' 不是单个区间的概率，而是'这种制作方法'的长期命中率。")

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
    print(f"[3] 套圈图已保存: {path}")


def width_vs_n(pop: np.ndarray) -> None:
    """[4] 各样本量下的区间宽度 — 验证与 sqrt(n) 成反比。"""
    rng = np.random.default_rng(809)
    sizes = [20, 50, 200, 800]
    widths = []
    print()
    print("[4] 样本量与区间宽度 (宽度想减半，样本得要 4 倍)")
    for n in sizes:
        # 同一大小重复 200 次、量平均宽度 (去掉单次的偶然)
        w = [np.subtract(*confidence_interval(rng.choice(pop, n, replace=False))[::-1])
             for _ in range(200)]
        widths.append(np.mean(w))
        print(f"    n = {n:>4}: 平均区间宽度 {np.mean(w) / 1e4:8.2f} 万韩元")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, [w / 1e4 for w in widths], marker="o", color="#4878cf", label="Measured width")
    ref = widths[0] * np.sqrt(sizes[0]) / np.sqrt(np.array(sizes))
    ax.plot(sizes, ref / 1e4, ls="--", color="#d1495b", label="Theory: 1/sqrt(n) curve")
    ax.set_title("CI width shrinks like 1/sqrt(n)")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("95% CI width (10k KRW)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ci_width.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    宽度图已保存: {path}")


def biased_sampling(df: pd.DataFrame, mu: float) -> None:
    """[5] 有偏样本: 只调查周末，样本再大也一直错。"""
    rng = np.random.default_rng(810)
    weekend = df[df["weekday"].isin(["周六", "周日"])]["revenue"].to_numpy()
    print()
    print("[5] 偏倚实验 — '只在周末'调查的样本的置信区间")
    for n in [50, 400, 2000]:
        # 一直只调查周末客人的情形 (用有放回抽样模拟扩大调查规模)
        sample = rng.choice(weekend, size=n, replace=True)
        lo, hi = confidence_interval(sample)
        verdict = "包含" if lo <= mu <= hi else "错过!"
        print(f"    n = {n:>4}: [{lo / 1e4:7.1f}, {hi / 1e4:7.1f}] 万韩元 -> μ {verdict}")
    print(f"    (真实 μ = {mu / 1e4:.1f} 万韩元)")
    print("    -> 区间越来越窄、越来越自信，却一贯地错。")
    print("       偏倚不是样本量能解决的。'怎么抽的'排在前面。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()

    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    pop = df["revenue"].to_numpy(dtype=float)
    mu = pop.mean()
    print(f"[1] 总体准备完毕: 销售记录 {len(pop):,}条, 真实平均 μ = {mu / 1e4:.1f} 万韩元")
    print("    (现实中 μ 是未知数，但今天为了实验先开挂看到答案)")

    ring_toss(pop, mu)
    width_vs_n(pop)
    biased_sampling(df, mu)


if __name__ == "__main__":
    main()
