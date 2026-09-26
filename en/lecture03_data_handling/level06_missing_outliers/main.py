"""
Hands-on missing-value and outlier handling.
We diagnose the contamination (missing values, negatives) in the cafe chain's
sales data, find outliers with a domain rule and the IQR method, and then
compare three strategies — dropna / fill with the overall median / fill with
per-group medians — by their statistics, to see which strategy fits when.
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so figures can be saved even without a display
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def diagnose(df: pd.DataFrame) -> None:
    """Diagnose and print the contamination status."""
    n = len(df)
    n_missing = int(df["revenue"].isna().sum())
    n_negative = int((df["revenue"] < 0).sum())  # NaN drops out of comparisons automatically
    print(f"  Total rows          : {n:,}")
    print(f"  Missing (NaN) count : {n_missing:,} ({n_missing / n:.2%})")
    print(f"  Negative revenues   : {n_negative:,} ({n_negative / n:.2%})")
    desc = df["revenue"].describe()
    print(f"  describe() summary  : min={desc['min']:,.0f}  "
          f"median={desc['50%']:,.0f}  max={desc['max']:,.0f}")
    if desc["min"] < 0:
        print("  -> min is negative! Values that make no sense as revenue are mixed in.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    print("[1] Diagnosing the contamination — you know nothing until you count")
    rows = hjh_data.sales_table(n_days=365, seed=42)  # fixed seed: same data every time
    df = pd.DataFrame(rows)
    diagnose(df)

    # ------------------------------------------------------------------
    print("\n[2] Applying the domain rule — 'revenue cannot be negative'")
    df_rule = df.copy()  # preserve the original as evidence
    neg_mask = df_rule["revenue"] < 0
    print(f"  Marking {int(neg_mask.sum())} negatives as NaN (value unknown) "
          f"instead of deleting them.")
    df_rule.loc[neg_mask, "revenue"] = np.nan
    print(f"  Missing count after : {int(df_rule['revenue'].isna().sum()):,}"
          f" (originally missing + formerly negative)")

    # ------------------------------------------------------------------
    print("\n[3] The IQR method — finding statistically suspicious values")
    rev = df_rule["revenue"].dropna()
    q1, q3 = rev.quantile(0.25), rev.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((rev < low) | (rev > high)).sum())
    print(f"  Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}")
    print(f"  Normal range: [{low:,.0f}, {high:,.0f}]")
    print(f"  Outlier candidates outside the range: {n_out} ({n_out / len(rev):.2%})")
    print("  -> Weekend and peak-season sales may be in there, so these are only "
          "'candidates' — never delete automatically.")

    # ------------------------------------------------------------------
    print("\n[4] Comparing the strategies — same data, different numbers")
    # (a) drop the missing rows
    df_a = df_rule.dropna(subset=["revenue"])
    # (b) fill with the overall median
    overall_median = df_rule["revenue"].median()
    df_b = df_rule.copy()
    df_b["revenue"] = df_b["revenue"].fillna(overall_median)
    # (c) fill with the per store-and-category median (transform keeps the row count)
    group_median = df_rule.groupby(["store", "category"])["revenue"].transform("median")
    df_c = df_rule.copy()
    df_c["revenue"] = df_c["revenue"].fillna(group_median)

    report = pd.DataFrame({
        "rows": [len(df_a), len(df_b), len(df_c)],
        "total revenue (M KRW)": [df_a["revenue"].sum() / 1e6,
                                  df_b["revenue"].sum() / 1e6,
                                  df_c["revenue"].sum() / 1e6],
        "mean revenue": [df_a["revenue"].mean(),
                         df_b["revenue"].mean(),
                         df_c["revenue"].mean()],
    }, index=["(a) dropna", "(b) overall median", "(c) group median"])
    print(report.round(0).to_string())
    print("  Reading it: (a) loses rows, so its total revenue comes out smallest. "
          "Dangerous for a totals report!")
    print("        (b) and (c) preserve the rows, but (c) reflects store/product "
          "characteristics and is more precise.")

    # ------------------------------------------------------------------
    print("\n[5] Before/after summary + saving the boxplot")
    before = df["revenue"].describe()
    after = df_c["revenue"].describe()
    summary = pd.DataFrame({"before": before, "after (c)": after}).round(0)
    print(summary.loc[["count", "mean", "min", "50%", "max"]].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].boxplot(df["revenue"].dropna())
    axes[0].set_title("before (with negatives)")
    axes[1].boxplot(df_c["revenue"])
    axes[1].set_title("after (rule + group median)")
    for ax in axes:
        ax.set_ylabel("revenue (KRW)")
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "boxplot_before_after.png")
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    print(f"  Boxplot saved: {png_path}")
    print("\nRecap: diagnose -> infer the cause -> apply the rule -> compare strategies. "
          "The order is what makes the quality.")


if __name__ == "__main__":
    main()
