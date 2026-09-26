"""
level02 — Distributions and Histograms

Saves the distribution of the cafe-chain sales data as three histogram PNGs.
1) An experiment showing how the number of bins changes the story (the shape)
2) The right tail (skew) and the mean/median disagreement
3) A two-peak distribution created by mixing weekdays and weekends
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save to files only, no display window
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def load_sales() -> pd.DataFrame:
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    df["revenue_man"] = df["revenue"] / 10_000  # in KRW 10,000s, easier to read
    return df


def describe(df: pd.DataFrame) -> None:
    rev = df["revenue_man"]
    mean, med = rev.mean(), rev.median()
    p95 = np.percentile(rev, 95)
    skew = rev.skew()
    print("[1] Revenue per transaction (KRW 10,000s) — distribution summary")
    print(f"    mean {mean:.1f} | median {med:.1f} | p95 {p95:.1f} | skewness {skew:.2f}")
    print(f"    -> mean > median, skewness > 0 : a distribution with a long right tail.")


def plot_bins_experiment(df: pd.DataFrame) -> None:
    """Draw the same data with 5 / 30 / 200 bins and compare."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, bins in zip(axes, [5, 30, 200]):
        ax.hist(df["revenue_man"], bins=bins, color="#4878cf", edgecolor="white")
        ax.set_title(f"bins = {bins}")
        ax.set_xlabel("Revenue per transaction (KRW 10,000s)")
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Same data, different bins — the scale changes the story")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_bins.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] Bin experiment saved: {path}")
    print("    bins=5 smears the peaks away; bins=200 draws every random bump.")


def plot_skew(df: pd.DataFrame) -> None:
    """Overlay mean/median vertical lines on the right-tailed distribution."""
    rev = df["revenue_man"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(rev, bins=40, color="#9fbce8", edgecolor="white")
    ax.axvline(rev.mean(), color="#d1495b", lw=2, label=f"mean {rev.mean():.0f}")
    ax.axvline(rev.median(), color="#2e6f40", lw=2, ls="--",
               label=f"median {rev.median():.0f}")
    ax.set_title("The right tail drags the mean along")
    ax.set_xlabel("Revenue per transaction (KRW 10,000s)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_skew.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] Skew figure saved: {path}")


def plot_bimodal(df: pd.DataFrame) -> None:
    """Split weekdays and weekends and overlay them — the hidden groups appear."""
    weekend = df[df["weekday"].isin(["Sat", "Sun"])]["revenue_man"]
    weekday = df[~df["weekday"].isin(["Sat", "Sun"])]["revenue_man"]
    bins = np.linspace(df["revenue_man"].min(), df["revenue_man"].max(), 40)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # density=True: compares fairly as proportions even with different sample sizes
    ax.hist(weekday, bins=bins, density=True, alpha=0.6, label="Weekdays", color="#4878cf")
    ax.hist(weekend, bins=bins, density=True, alpha=0.6, label="Weekends", color="#e1a03c")
    ax.set_title("The single-looking distribution was really the sum of two groups")
    ax.set_xlabel("Revenue per transaction (KRW 10,000s)")
    ax.set_ylabel("Proportion density")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_bimodal.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] Weekday/weekend split figure saved: {path}")
    print(f"    Weekday median {weekday.median():.0f} vs weekend median {weekend.median():.0f} (KRW 10,000s)")
    print("    -> Interpreting a mixed distribution as one lump squashes two stories into one.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_sales()
    describe(df)
    plot_bins_experiment(df)
    plot_skew(df)
    plot_bimodal(df)


if __name__ == "__main__":
    main()
