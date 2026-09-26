"""
level04 — 好图表 vs 坏图表

用完全相同的数据并排画出'歪曲图表'和'诚实图表'。
1) 截断坐标轴，把 3% 的差距画成压倒性优势
2) 3D 风格的饼图 vs 排好序的水平柱状图
3) 操纵 y 轴范围，把微小起伏画成过山车

※ 图内文字统一用英文，避免因环境缺字体而乱码。
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 图中使用的门店英文名
STORE_EN = {"朝阳店": "Chaoyang", "海淀店": "Haidian", "浦东店": "Pudong",
            "天河店": "Tianhe", "南山店": "Nanshan"}


def setup_plot_style() -> None:
    """图内标签用英文，因此只需要调整负号设置。"""
    plt.rcParams["axes.unicode_minus"] = False


def load_data():
    rows = hjh_data.sales_table(n_days=90, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    store_rev = (df.groupby("store")["revenue"].sum() / 1e8)  # 亿韩元
    return df, store_rev


def crime1_truncated_axis(store_rev: pd.Series) -> None:
    """罪状 1: 截断柱状图的 y 轴。"""
    two = store_rev.sort_values(ascending=False).iloc[[1, 2]]  # 差距很小的两家门店
    a, b = two.index
    gap_pct = (two[a] - two[b]) / two[b] * 100

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#d1495b", "#9aa7b5"]
    labels = [STORE_EN.get(s, s) for s in two.index]

    bad.bar(labels, two.values, color=colors)
    bad.set_ylim(two.min() * 0.985, two.max() * 1.005)   # 截断坐标轴!
    bad.set_title(f"[Distorted] Truncated axis — {gap_pct:.1f}% gap looks like a landslide")
    bad.set_ylabel("Total revenue (100M KRW)")

    good.bar(labels, two.values, color=colors)
    good.set_ylim(0, two.max() * 1.15)                   # 从 0 开始
    good.set_title("[Honest] Bars start at zero — the real gap")
    good.set_ylabel("Total revenue (100M KRW)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "truncated_axis.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] 截断坐标轴的对比已保存: {path}")
    print(f"    实际差距只有 {gap_pct:.1f}%。请和左图给你的第一印象比一比。")


def crime2_pie_vs_bar(store_rev: pd.Series) -> None:
    """罪状 2: 数值彼此接近时的饼图 vs 排序后的水平柱。"""
    share = store_rev / store_rev.sum() * 100

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.8))
    # 阴影 + 弹出的一块 = 重现会议室里的 3D 饼图
    bad.pie(share.values, labels=[STORE_EN.get(s, s) for s in share.index],
            shadow=True,
            explode=[0.08 if i == 0 else 0 for i in range(len(share))],
            startangle=90)
    bad.set_title("[Distorted] Pie — can you tell which slice is 2nd?")

    ordered = share.sort_values()
    good.barh([STORE_EN.get(s, s) for s in ordered.index], ordered.values,
              color="#4878cf")
    for y, v in enumerate(ordered.values):
        good.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    good.set_title("[Honest] Sorted bars — rank and gap at a glance")
    good.set_xlabel("Revenue share (%)")
    good.set_xlim(0, ordered.max() * 1.25)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pie_vs_bar.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 饼图 vs 柱状图已保存: {path}")


def crime3_inflated_line(df: pd.DataFrame) -> None:
    """罪状 3: 收窄 y 轴范围，把微小起伏画成过山车。"""
    daily = df.groupby("day_index")["revenue"].sum() / 1e8
    weekly = daily.rolling(7).mean().dropna()            # 7 日移动平均使曲线平缓

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.2))
    bad.plot(weekly.index, weekly.values, color="#d1495b", lw=2)
    bad.set_ylim(weekly.min() * 0.998, weekly.max() * 1.002)  # 操纵范围!
    bad.set_title("[Distorted] Zoomed axis — crash-and-rebound drama?")
    bad.set_xlabel("Day")
    bad.set_ylabel("7-day avg daily revenue (100M KRW)")

    good.plot(weekly.index, weekly.values, color="#4878cf", lw=2)
    good.set_ylim(0, weekly.max() * 1.2)
    good.set_title("[Honest] Contextual range — essentially stable")
    good.set_xlabel("Day")
    good.set_ylabel("7-day avg daily revenue (100M KRW)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "inflated_line.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    swing = (weekly.max() - weekly.min()) / weekly.mean() * 100
    print(f"[4] 折线图范围操纵已保存: {path}")
    print(f"    实际的起伏幅度只有平均值的 ±{swing / 2:.1f}% 左右。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    df, store_rev = load_data()

    print("[1] 作为原料的真实数字 (各门店总销售额, 亿韩元)")
    for store, v in store_rev.sort_values(ascending=False).items():
        print(f"    {store:<6} {v:8.2f}")
    print("    -> 下面三张图里的'歪曲'和'诚实'，全都出自这同一组数字。")

    crime1_truncated_axis(store_rev)
    crime2_pie_vs_bar(store_rev)
    crime3_inflated_line(df)
    print("[5] 结论: 图表的第一印象若与细读数字的结论不同，它就是坏图表。")


if __name__ == "__main__":
    main()
