"""
level01 — Mean, Median, Variance, Standard Deviation

Using the cafe-chain sales data (hjh_data), we compute and cross-check the
main summary statistics, then run a step-by-step experiment showing how far
a single extreme value (one giant group order) drags the mean, the median,
and the standard deviation.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_clean_sales() -> pd.DataFrame:
    """Load the sales data and remove missing and negative contaminated values."""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.dropna(subset=["revenue"])          # drop missing values
    df = df[df["revenue"] > 0].copy()           # drop negative contamination
    df["revenue"] = df["revenue"].astype(float)
    print(f"[1] Data cleaning: {before:,} rows -> {len(df):,} rows "
          f"({before - len(df)} missing/negative rows removed)")
    return df


def describe_revenue(df: pd.DataFrame) -> None:
    """Cross-check pandas results against hand-rolled numpy calculations."""
    rev = df["revenue"].to_numpy()
    mean_np = rev.sum() / len(rev)              # the mean, by hand
    var_np = ((rev - mean_np) ** 2).mean()      # the (population) variance, by hand
    std_np = var_np ** 0.5

    print()
    print("[2] Summary statistics of revenue per transaction — pandas vs numpy by hand")
    print(f"    Mean     : {df['revenue'].mean():>14,.0f} KRW | by hand {mean_np:>14,.0f} KRW")
    print(f"    Median   : {df['revenue'].median():>14,.0f} KRW")
    print(f"    Variance : {df['revenue'].var(ddof=0):>14,.0f} KRW^2 | by hand {var_np:>14,.0f} KRW^2")
    print(f"    Std dev  : {df['revenue'].std(ddof=0):>14,.0f} KRW | by hand {std_np:>14,.0f} KRW")
    print("    -> Mean > median : the revenue distribution has a long tail toward large values.")


def by_store(df: pd.DataFrame) -> None:
    """Compare mean, median, standard deviation, and coefficient of variation per store."""
    g = df.groupby("store")["revenue"].agg(["mean", "median", "std"])
    g["cv"] = g["std"] / g["mean"]              # coefficient of variation
    print()
    print("[3] Summary statistics per store (CV = std dev / mean)")
    print(f"    {'store':<12} {'mean':>12} {'median':>12} {'std dev':>12} {'CV':>8}")
    for store, row in g.sort_values("mean", ascending=False).iterrows():
        print(f"    {store:<12} {row['mean']:>12,.0f} {row['median']:>12,.0f} "
              f"{row['std']:>12,.0f} {row['cv']:>8.2f}")


def outlier_experiment(df: pd.DataFrame) -> None:
    """Test how much one extreme value moves the statistics of a 30-row sample."""
    rng = np.random.default_rng(7)              # fixed seed
    gangnam = df[df["store"] == "Downtown"]["revenue"].to_numpy()
    sample = rng.choice(gangnam, size=30, replace=False)

    def report(tag: str, values: np.ndarray) -> tuple[float, float, float]:
        m, md, sd = values.mean(), float(np.median(values)), values.std()
        print(f"    {tag:<26} mean {m:>12,.0f} | median {md:>12,.0f} | std {sd:>12,.0f}")
        return m, md, sd

    print()
    print("[4] Outlier experiment — add one giant group order to a 30-row Downtown sample")
    base = report("original sample (30 rows)", sample)
    outlier = 500_000_000.0                     # a KRW 500 million group order
    spiked = np.append(sample, outlier)
    after = report("with outlier (31 rows)", spiked)

    print()
    print("    What a SINGLE value changed:")
    print(f"      Mean     : {after[0] - base[0]:>+14,.0f} KRW  (dragged far away)")
    print(f"      Median   : {after[1] - base[1]:>+14,.0f} KRW  (barely moved = robust)")
    print(f"      Std dev  : {after[2] - base[2]:>+14,.0f} KRW  (explodes — it squares deviations)")
    print("    => For long-tailed data like salaries and revenue, always report the median too.")


def main() -> None:
    df = load_clean_sales()
    describe_revenue(df)
    by_store(df)
    outlier_experiment(df)


if __name__ == "__main__":
    main()
