"""
level02 — 分布与直方图

把连锁咖啡店销售数据的分布保存为 3 张直方图 PNG。
1) bin 的个数改变故事 (形状) 的实验
2) 右尾 (偏度) 与平均数、中位数的错位
3) 工作日/周末混在一起形成的双峰分布

※ 图内文字统一用英文，以避免中文字体缺失导致乱码。
   (终端输出为中文。)
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 不弹窗口，只保存到文件
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def setup_plot_style() -> None:
    """图表的通用设置。图内标签用英文，因此无需额外字体。"""
    plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示成方块


def load_sales() -> pd.DataFrame:
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    df["revenue_man"] = df["revenue"] / 10_000  # 换成万韩元单位更好读
    return df


def describe(df: pd.DataFrame) -> None:
    rev = df["revenue_man"]
    mean, med = rev.mean(), rev.median()
    p95 = np.percentile(rev, 95)
    skew = rev.skew()
    print("[1] 销售额(每笔，万韩元)分布概要")
    print(f"    平均 {mean:.1f} | 中位数 {med:.1f} | p95 {p95:.1f} | 偏度 {skew:.2f}")
    print(f"    -> 平均 > 中位数、偏度 > 0 : 这是右尾很长的分布。")


def plot_bins_experiment(df: pd.DataFrame) -> None:
    """把同一份数据分别用 bin 5 / 30 / 200 画出来比较。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, bins in zip(axes, [5, 30, 200]):
        ax.hist(df["revenue_man"], bins=bins, color="#4878cf", edgecolor="white")
        ax.set_title(f"bins = {bins}")
        ax.set_xlabel("Revenue per sale (10k KRW)")
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Same data, different bins — the scale changes the story")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_bins.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] bin 实验已保存: {path}")
    print("    bins=5 把峰糊掉了，bins=200 连随机数的锯齿都画了出来。")


def plot_skew(df: pd.DataFrame) -> None:
    """在右尾分布上叠加平均数/中位数的竖线。"""
    rev = df["revenue_man"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(rev, bins=40, color="#9fbce8", edgecolor="white")
    ax.axvline(rev.mean(), color="#d1495b", lw=2, label=f"Mean {rev.mean():.0f}")
    ax.axvline(rev.median(), color="#2e6f40", lw=2, ls="--",
               label=f"Median {rev.median():.0f}")
    ax.set_title("The right tail drags the mean")
    ax.set_xlabel("Revenue per sale (10k KRW)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_skew.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 偏度图已保存: {path}")


def plot_bimodal(df: pd.DataFrame) -> None:
    """按工作日/周末拆开叠加着画，就能看见隐藏的两个群体。"""
    weekend = df[df["weekday"].isin(["周六", "周日"])]["revenue_man"]
    weekday = df[~df["weekday"].isin(["周六", "周日"])]["revenue_man"]
    bins = np.linspace(df["revenue_man"].min(), df["revenue_man"].max(), 40)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # density=True: 样本量不同也能按比例公平比较
    ax.hist(weekday, bins=bins, density=True, alpha=0.6, label="Weekday", color="#4878cf")
    ax.hist(weekend, bins=bins, density=True, alpha=0.6, label="Weekend", color="#e1a03c")
    ax.set_title("One lump was actually the sum of two groups")
    ax.set_xlabel("Revenue per sale (10k KRW)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_bimodal.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] 工作日/周末分离图已保存: {path}")
    print(f"    工作日中位数 {weekday.median():.0f} vs 周末中位数 {weekend.median():.0f} (万韩元)")
    print("    -> 把群体混杂的分布整个拿来解读，就等于把两个故事压成了一个。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    df = load_sales()
    describe(df)
    plot_bins_experiment(df)
    plot_skew(df)
    plot_bimodal(df)


if __name__ == "__main__":
    main()
