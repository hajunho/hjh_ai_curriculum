"""
level05 — 相関関係と因果関係

第 1 部: カフェの売上データで、広告費-売上の本物の相関を観察します。
第 2 部: 交絡因子 (暑さ) が作った「偽物の相関」を合成データで製造し、
         層化 (交絡因子の区間別分析) で相関が崩れることを確かめます。

※ 図中のテキストは、フォントの文字化けを避けるため英語にしています。
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
    """広告費と売上の相関係数 + 散布図。"""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    daily = df.groupby("day_index").agg(ad=("ad_cost", "sum"), rev=("revenue", "sum"))

    r = np.corrcoef(daily["ad"], daily["rev"])[0, 1]
    print("[1] 第 1 部 — 日別広告費 vs 日別売上")
    print(f"    ピアソン相関係数 r = {r:.3f}")
    print("    -> 正の相関があります。しかしこれだけでは「広告が売上を伸ばした」とは")
    print("       言えません。(繁忙期・週末のような交絡因子が残っています)")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(daily["ad"] / 1e6, daily["rev"] / 1e8, s=18, alpha=0.5, color="#4878cf")
    ax.set_title(f"Ad spend vs Revenue (r = {r:.2f}) — correlation != causation")
    ax.set_xlabel("Daily ad spend (million KRW)")
    ax.set_ylabel("Daily revenue (100M KRW)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ad_revenue.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] 散布図を保存: {path}")


def part2_confounder() -> None:
    """交絡因子の実験: Z(暑さ) -> X(アイスクリーム)、Z -> Y(水難事故)。"""
    rng = np.random.default_rng(505)  # seed 固定
    n = 400
    noise_scale = 2.0

    heat = rng.uniform(0, 10, size=n)                     # 交絡因子 Z: 暑さ指数
    icecream = 20 + 8 * heat + rng.normal(0, 8 * noise_scale, n)   # X = f(Z)+ノイズ
    accidents = 1 + 0.9 * heat + rng.normal(0, 0.9 * noise_scale, n)  # Y = g(Z)+ノイズ
    # 注目: icecream と accidents は互いをまったく参照していません!

    r_total = np.corrcoef(icecream, accidents)[0, 1]
    print()
    print("[3] 第 2 部 — 偽物の相関の製造")
    print("    生成ルール: X(アイスクリーム) = f(暑さ)+ノイズ、Y(事故) = g(暑さ)+ノイズ")
    print("    X と Y は互いに独立に作られたのに...")
    print(f"    全体の相関係数 r = {r_total:.3f}  <- 強い相関に見えます!")

    # 層化: 暑さ指数を 5 つの区間に分け、区間の「中」だけで相関を計算
    bins = np.linspace(0, 10, 6)
    labels = np.digitize(heat, bins[1:-1])                # 0〜4 の区間番号
    print()
    print("[4] 層化の実験 — 暑さが似た日どうしだけで見直すと")
    inner_rs = []
    for k in range(5):
        mask = labels == k
        r_k = np.corrcoef(icecream[mask], accidents[mask])[0, 1]
        inner_rs.append(r_k)
        print(f"    暑さ区間 {bins[k]:.0f}~{bins[k + 1]:.0f} ({mask.sum():3d}日): r = {r_k:+.3f}")
    print(f"    区間内の相関の平均 = {np.mean(inner_rs):+.3f}  (0 の近くまで崩れる)")
    print("    -> 全体の相関は 2 つの変数の関係ではなく、「暑さ」という胴体の仕業でした。")

    # 図: 全体 (歪んだ印象) vs 区間別の色分け (真実)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.scatter(icecream, accidents, s=14, alpha=0.5, color="#555555")
    left.set_title(f"All data: r = {r_total:.2f} (looks related)")
    left.set_xlabel("Ice cream sales")
    left.set_ylabel("Water accidents")

    cmap = ["#3b6bb5", "#5da05d", "#e1a03c", "#d1495b", "#7d4fa3"]
    for k in range(5):
        mask = labels == k
        right.scatter(icecream[mask], accidents[mask], s=14, alpha=0.6,
                      color=cmap[k], label=f"Heat {bins[k]:.0f}~{bins[k + 1]:.0f} (r={inner_rs[k]:+.2f})")
    right.set_title("By heat band: chaos inside each cluster")
    right.set_xlabel("Ice cream sales")
    right.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "confounder.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[5] 交絡因子の図を保存: {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    part1_real_correlation()
    part2_confounder()
    print()
    print("[6] 結論: 相関は「調べてみる価値のある手がかり」であって「行動してよい証拠」では")
    print("    ありません。因果の証拠が必要なら、実験 (A/B テスト、level10) へ進みます。")


if __name__ == "__main__":
    main()
