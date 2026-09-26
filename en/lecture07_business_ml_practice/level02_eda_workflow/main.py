"""
Automated EDA report for churn_table.
Proceeds in the order: hypotheses -> basic intake -> univariate -> bivariate
(group differences vs the target, correlations) -> hypothesis verdicts,
and saves 2 PNGs to outputs/: a distribution comparison and a correlation heatmap.
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

plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
NUM_COLS = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
TARGET = "churned"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))

    print("=" * 62)
    print(" Automated EDA report for churn_table")
    print("=" * 62)

    # [1] Hypotheses first ----------------------------------------------
    hypotheses = [
        ("H1", "The fewer usage days in the last 30 days, the more churn", "usage_days_30d", "low"),
        ("H2", "The more support calls, the more churn", "support_calls_30d", "high"),
        ("H3", "The longer the tenure, the less churn", "tenure_months", "low"),
    ]
    print("\n[1] Hypotheses to verify (the sentence comes before the chart)")
    for hid, text, _, _ in hypotheses:
        print(f"    {hid}. {text}")

    # [2] Basic intake ----------------------------------------------------
    print("\n[2] Basic intake")
    print(f"    rows x cols: {df.shape[0]} x {df.shape[1]}")
    print(f"    total missing values: {int(df.isna().sum().sum())}")
    print(f"    target (churned) ratio: {df[TARGET].mean():.1%}  <- imbalanced! do not evaluate with accuracy")

    # [3] Univariate -------------------------------------------------------
    print("\n[3] Univariate summary (numeric)")
    desc = df[NUM_COLS].describe().T[["mean", "std", "min", "50%", "max"]]
    print(desc.round(2).to_string())

    # [4] Bivariate: group means by target + correlations -------------------
    print("\n[4] Bivariate: group means, churned (1) vs retained (0)")
    grp = df.groupby(TARGET)[NUM_COLS].mean().T
    grp.columns = ["stay(0)", "churn(1)"]
    grp["ratio"] = (grp["churn(1)"] / grp["stay(0)"]).round(2)
    print(grp.round(2).to_string())

    corr = df[NUM_COLS + [TARGET]].corr()
    rank = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\n    correlation with the target, ranked by absolute value:")
    for name, v in rank.items():
        print(f"      {name:<18} {v:+.3f}")
    if rank.abs().max() > 0.9:
        print("      ! variable with correlation above 0.9 found -> suspected leakage, check when it is created")

    # [5] Hypothesis verdicts -----------------------------------------------
    print("\n[5] Hypothesis verdicts")
    for hid, text, col, direction in hypotheses:
        churn_mean, stay_mean = grp.loc[col, "churn(1)"], grp.loc[col, "stay(0)"]
        supported = churn_mean < stay_mean if direction == "low" else churn_mean > stay_mean
        verdict = "SUPPORTED" if supported else "REJECTED"
        print(f"    {hid} [{verdict}] churn {churn_mean:.2f} vs stay {stay_mean:.2f} — {text}")

    # [6] Save figures --------------------------------------------------------
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

    print("\n[6] Figures saved")
    print(f"    {p1}")
    print(f"    {p2}")
    print("\n    Lesson: EDA is not drawing pictures — it is 'writing hypotheses as sentences and ruling on them'.")


if __name__ == "__main__":
    main()
