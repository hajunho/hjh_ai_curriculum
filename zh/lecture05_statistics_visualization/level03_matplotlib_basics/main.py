"""
level03 — Matplotlib 基础图表

沿着 figure/axes 结构，用连锁咖啡店销售额绘制
折线图 (月度走势) · 柱状图 (门店比较) · 散点图 (广告费-销售额)，
并保存为 outputs/ 下的 PNG。还包含中文字体的自动设置函数。

※ 图内文字统一用英文，避免因环境缺字体而乱码。
   (终端输出为中文。)
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 必须在 import pyplot '之前' — 不弹窗口，直接存文件
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 图中使用的门店英文名 (没有中文字体的环境也不会乱码)
STORE_EN = {"朝阳店": "Chaoyang", "海淀店": "Haidian", "浦东店": "Pudong",
            "天河店": "Tianhe", "南山店": "Nanshan"}


def set_chinese_font() -> None:
    """按顺序查找 macOS/Windows/Linux 上有代表性的中文字体并注册。

    本次实战的图用英文标签，所以这一步并非必需；
    但想在图中写中文的项目可以把这个函数原样拷走。
    """
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "SimHei"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            print(f"[0] 中文字体已设置: {cand}")
            break
    else:
        print("[0] 未找到中文字体，使用默认字体 (图是英文标签，所以没有问题)")
    plt.rcParams["axes.unicode_minus"] = False


def load_sales() -> pd.DataFrame:
    """清洗销售数据 + 添加月份列。"""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    df["month"] = df["date"].str[:7]            # '2025-03' 的形式
    print(f"[1] 数据准备完成: {len(df):,}行, 时间范围 {df['date'].min()} ~ {df['date'].max()}")
    return df


def plot_line_monthly(df: pd.DataFrame) -> None:
    """折线图: 随时间的变化用线来画。"""
    monthly = df.groupby("month")["revenue"].sum() / 1e8  # 亿韩元单位

    fig, ax = plt.subplots(figsize=(8, 4.5))              # 纸张 + 段落
    ax.plot(monthly.index, monthly.values,                # 句子(画线)
            marker="o", color="#4878cf", lw=2)
    for x, y in monthly.items():                          # 数值标签
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title("Monthly Total Revenue Trend")           # 小标题
    ax.set_xlabel("Month")
    ax.set_ylabel("Total revenue (100M KRW)")
    ax.set_ylim(0, monthly.max() * 1.2)                   # 从 0 开始 (level04 的预告)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "line_monthly.png")
    fig.savefig(path, dpi=120)                            # 提交(保存)
    plt.close(fig)                                        # 收起纸张
    print(f"[2] 折线图已保存: {path}")


def plot_bar_stores(df: pd.DataFrame) -> None:
    """柱状图: 类别比较用柱子。只给第一名上色强调。"""
    stores = (df.groupby("store")["revenue"].sum() / 1e8).sort_values(ascending=False)
    colors = ["#d1495b" if i == 0 else "#9aa7b5" for i in range(len(stores))]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [STORE_EN.get(s, s) for s in stores.index]
    ax.bar(labels, stores.values, color=colors)
    ax.set_title("Total Revenue by Store (highlight = top store)")
    ax.set_xlabel("Store")
    ax.set_ylabel("Total revenue (100M KRW)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "bar_stores.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 柱状图已保存: {path} (第一名: {stores.index[0]})")


def plot_scatter_ad(df: pd.DataFrame) -> None:
    """散点图: 两个数值的关系用点来画。(隔一关的预告片)"""
    daily = df.groupby("day_index").agg(
        ad=("ad_cost", "sum"), rev=("revenue", "sum"))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(daily["ad"] / 1e6, daily["rev"] / 1e8,
               s=18, alpha=0.5, color="#2e6f40")
    ax.set_title("Daily Ad Spend vs Daily Revenue")
    ax.set_xlabel("Ad spend (million KRW)")
    ax.set_ylabel("Revenue (100M KRW)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "scatter_ad.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] 散点图已保存: {path}")
    print("    点云看起来朝右上方倾斜吗? 这个'关系'会在 level05 深挖。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_chinese_font()
    df = load_sales()
    plot_line_monthly(df)
    plot_bar_stores(df)
    plot_scatter_ad(df)
    print("[5] 所有图表都是按'纸张->段落->句子->小标题->保存'五个步骤画出来的。")


if __name__ == "__main__":
    main()
