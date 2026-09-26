"""
level05 — 相关关系与因果关系

第 1 部: 在咖啡店销售数据中观察广告费-销售额的真实相关。
第 2 部: 用合成数据制造混杂变量(炎热)带来的'假相关'，
        再用分层(按混杂变量区间分析)确认相关随之崩塌。

※ 图内文字统一用英文，避免因环境缺字体而乱码。
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def setup_plot_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False


def part1_real_correlation() -> None:
    """广告费与销售额的相关系数 + 散点图。"""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    daily = df.groupby("day_index").agg(ad=("ad_cost", "sum"), rev=("revenue", "sum"))

    r = np.corrcoef(daily["ad"], daily["rev"])[0, 1]
    print("[1] 第 1 部 — 每日广告费 vs 每日销售额")
    print(f"    皮尔逊相关系数 r = {r:.3f}")
    print("    -> 存在正相关。但仅凭这一点还不能说'广告拉动了销售额'。")
    print("       (旺季、周末这类混杂变量还没排除)")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(daily["ad"] / 1e6, daily["rev"] / 1e8, s=18, alpha=0.5, color="#4878cf")
    ax.set_title(f"Ad spend vs Revenue (r = {r:.2f}) — correlation != causation")
    ax.set_xlabel("Daily ad spend (million KRW)")
    ax.set_ylabel("Daily revenue (100M KRW)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ad_revenue.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] 散点图已保存: {path}")


def part2_confounder() -> None:
    """混杂变量实验: Z(炎热) -> X(冰淇淋), Z -> Y(玩水事故)。"""
    rng = np.random.default_rng(505)  # 固定 seed
    n = 400
    noise_scale = 2.0

    heat = rng.uniform(0, 10, size=n)                     # 混杂变量 Z: 炎热指数
    icecream = 20 + 8 * heat + rng.normal(0, 8 * noise_scale, n)   # X = f(Z)+噪声
    accidents = 1 + 0.9 * heat + rng.normal(0, 0.9 * noise_scale, n)  # Y = g(Z)+噪声
    # 注意: icecream 和 accidents 完全没有引用彼此!

    r_total = np.corrcoef(icecream, accidents)[0, 1]
    print()
    print("[3] 第 2 部 — 制造假相关")
    print("    生成规则: X(冰淇淋) = f(炎热)+噪声, Y(事故) = g(炎热)+噪声")
    print("    X 和 Y 是彼此独立生成的，然而...")
    print(f"    整体相关系数 r = {r_total:.3f}  <- 看起来像强相关!")

    # 分层: 把炎热指数切成 5 个区间，只在区间'内部'计算相关
    bins = np.linspace(0, 10, 6)
    labels = np.digitize(heat, bins[1:-1])                # 0~4 区间编号
    print()
    print("[4] 分层实验 — 只把炎热程度相近的日子放在一起看")
    inner_rs = []
    for k in range(5):
        mask = labels == k
        r_k = np.corrcoef(icecream[mask], accidents[mask])[0, 1]
        inner_rs.append(r_k)
        print(f"    炎热区间 {bins[k]:.0f}~{bins[k + 1]:.0f} ({mask.sum():3d}天): r = {r_k:+.3f}")
    print(f"    区间内相关的平均 = {np.mean(inner_rs):+.3f}  (塌到 0 附近)")
    print("    -> 整体相关不是两个变量之间的关系，而是'炎热'这具身体的作品。")

    # 图: 整体(歪曲的印象) vs 按区间着色(真相)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.scatter(icecream, accidents, s=14, alpha=0.5, color="#555555")
    left.set_title(f"All together: r = {r_total:.2f} (looks related)")
    left.set_xlabel("Ice cream sales")
    left.set_ylabel("Water accidents")

    cmap = ["#3b6bb5", "#5da05d", "#e1a03c", "#d1495b", "#7d4fa3"]
    for k in range(5):
        mask = labels == k
        right.scatter(icecream[mask], accidents[mask], s=14, alpha=0.6,
                      color=cmap[k], label=f"Heat {bins[k]:.0f}~{bins[k + 1]:.0f} (r={inner_rs[k]:+.2f})")
    right.set_title("Within heat bands: no order inside each blob")
    right.set_xlabel("Ice cream sales")
    right.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "confounder.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[5] 混杂变量图已保存: {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    part1_real_correlation()
    part2_confounder()
    print()
    print("[6] 结论: 相关是'值得调查的线索'，不是'可以据此行动的证据'。")
    print("    需要因果的证据时，就去做实验 (A/B 测试, level10)。")


if __name__ == "__main__":
    main()
