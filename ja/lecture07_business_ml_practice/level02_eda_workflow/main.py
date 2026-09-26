"""
churn_table 自動 EDA レポート。
仮説の設定 -> 基礎問診 -> 単変量 -> 二変量(ターゲット別の差、相関) -> 仮説判定 の順で進め、
分布比較の図と相関ヒートマップの PNG 2 枚を outputs/ に保存します。
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 日本語フォントがあれば使用 (なければ英語ラベルだけを崩れないように維持)
for _f in ["Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP"]:
    if any(ft.name == _f for ft in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
NUM_COLS = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
TARGET = "churned"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))

    print("=" * 62)
    print(" churn_table 自動 EDA レポート")
    print("=" * 62)

    # [1] 仮説が先 -----------------------------------------------------
    hypotheses = [
        ("H1", "直近30日の利用日数が少ないほど解約が多いはず", "usage_days_30d", "low"),
        ("H2", "カスタマーセンターへの問い合わせが多いほど解約が多いはず", "support_calls_30d", "high"),
        ("H3", "契約期間が長いほど解約が少ないはず", "tenure_months", "low"),
    ]
    print("\n[1] 検証する仮説 (グラフより文章が先)")
    for hid, text, _, _ in hypotheses:
        print(f"    {hid}. {text}")

    # [2] 基礎問診 -----------------------------------------------------
    print("\n[2] 基礎問診")
    print(f"    行 x 列: {df.shape[0]} x {df.shape[1]}")
    print(f"    欠損値の合計: {int(df.isna().sum().sum())}個")
    print(f"    ターゲット(churned)比率: {df[TARGET].mean():.1%}  <- 不均衡! 精度での評価は禁止")

    # [3] 単変量 --------------------------------------------------------
    print("\n[3] 単変量の要約 (数値型)")
    desc = df[NUM_COLS].describe().T[["mean", "std", "min", "50%", "max"]]
    print(desc.round(2).to_string())

    # [4] 二変量: ターゲットのグループ別平均 + 相関 -------------------------------
    print("\n[4] 二変量: 解約(1)/残留(0) グループ別の平均比較")
    grp = df.groupby(TARGET)[NUM_COLS].mean().T
    grp.columns = ["残留(0)", "解約(1)"]
    grp["差の倍率"] = (grp["解約(1)"] / grp["残留(0)"]).round(2)
    print(grp.round(2).to_string())

    corr = df[NUM_COLS + [TARGET]].corr()
    rank = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\n    ターゲットとの相関係数ランキング (絶対値の大きい順):")
    for name, v in rank.items():
        print(f"      {name:<18} {v:+.3f}")
    if rank.abs().max() > 0.9:
        print("      ! 相関 0.9 超の変数を発見 -> リーク疑い、生成時点の確認が必要")

    # [5] 仮説判定 -----------------------------------------------------
    print("\n[5] 仮説の判定")
    for hid, text, col, direction in hypotheses:
        churn_mean, stay_mean = grp.loc[col, "解約(1)"], grp.loc[col, "残留(0)"]
        supported = churn_mean < stay_mean if direction == "low" else churn_mean > stay_mean
        verdict = "支持" if supported else "棄却"
        print(f"    {hid} [{verdict}] 解約 {churn_mean:.2f} vs 残留 {stay_mean:.2f} — {text}")

    # [6] 図の保存 ------------------------------------------------------
    plot_cols = ["usage_days_30d", "support_calls_30d", "tenure_months", "plan_changes"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.ravel(), plot_cols):
        stay = df.loc[df[TARGET] == 0, col]
        churn = df.loc[df[TARGET] == 1, col]
        ax.hist(stay, bins=20, alpha=0.6, label="stay(0)", density=True)
        ax.hist(churn, bins=20, alpha=0.6, label="churn(1)", density=True)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Distribution by churn group")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "dist_by_target.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "correlation_heatmap.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)

    print("\n[6] 図の保存完了")
    print(f"    {p1}")
    print(f"    {p2}")
    print("\n    教訓: EDA はお絵描きではなく、「仮説を文章で書いて判定する手順」です。")


if __name__ == "__main__":
    main()
