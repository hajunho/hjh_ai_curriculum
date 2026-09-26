"""
churn_table 自动 EDA 报告。
按"提出假设 -> 基础问诊 -> 单变量 -> 双变量(按目标分组的差异、相关) -> 假设判定"的顺序推进，
并把分布对比图和相关热力图两张 PNG 保存到 outputs/。
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

# 有中文字体就用 (没有也没关系: 图内标签全部是英文, 不会乱码)
for _f in ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"]:
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
    print(" churn_table 自动 EDA 报告")
    print("=" * 62)

    # [1] 假设先行 -----------------------------------------------------
    hypotheses = [
        ("H1", "最近 30 天使用天数越少, 流失应该越多", "usage_days_30d", "low"),
        ("H2", "客服来电越多, 流失应该越多", "support_calls_30d", "high"),
        ("H3", "入网时间越长, 流失应该越少", "tenure_months", "low"),
    ]
    print("\n[1] 待验证的假设 (先有句子, 后有图)")
    for hid, text, _, _ in hypotheses:
        print(f"    {hid}. {text}")

    # [2] 基础问诊 -----------------------------------------------------
    print("\n[2] 基础问诊")
    print(f"    行 x 列: {df.shape[0]} x {df.shape[1]}")
    print(f"    缺失值总数: {int(df.isna().sum().sum())} 个")
    print(f"    目标(churned)比例: {df[TARGET].mean():.1%}  <- 不平衡! 禁止用准确率评估")

    # [3] 单变量 --------------------------------------------------------
    print("\n[3] 单变量汇总 (数值型)")
    desc = df[NUM_COLS].describe().T[["mean", "std", "min", "50%", "max"]]
    print(desc.round(2).to_string())

    # [4] 双变量: 按目标分组的平均 + 相关 -------------------------------
    print("\n[4] 双变量: 流失(1)/留存(0) 分组平均对比")
    grp = df.groupby(TARGET)[NUM_COLS].mean().T
    grp.columns = ["留存(0)", "流失(1)"]
    grp["差异倍率"] = (grp["流失(1)"] / grp["留存(0)"]).round(2)
    print(grp.round(2).to_string())

    corr = df[NUM_COLS + [TARGET]].corr()
    rank = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\n    与目标的相关系数排行 (按绝对值从大到小):")
    for name, v in rank.items():
        print(f"      {name:<18} {v:+.3f}")
    if rank.abs().max() > 0.9:
        print("      ! 发现相关系数超过 0.9 的变量 -> 疑似泄漏, 需确认生成时间点")

    # [5] 假设判定 -----------------------------------------------------
    print("\n[5] 假设判定")
    for hid, text, col, direction in hypotheses:
        churn_mean, stay_mean = grp.loc[col, "流失(1)"], grp.loc[col, "留存(0)"]
        supported = churn_mean < stay_mean if direction == "low" else churn_mean > stay_mean
        verdict = "支持" if supported else "否决"
        print(f"    {hid} [{verdict}] 流失 {churn_mean:.2f} vs 留存 {stay_mean:.2f} — {text}")

    # [6] 保存图片 ------------------------------------------------------
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

    print("\n[6] 图片保存完成")
    print(f"    {p1}")
    print(f"    {p2}")
    print("\n    教训: EDA 不是画图, 而是'把假设写成句子并给出判定'的流程。")


if __name__ == "__main__":
    main()
