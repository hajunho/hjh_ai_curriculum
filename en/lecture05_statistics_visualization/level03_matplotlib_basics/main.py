"""
level03 — Basic Charts with Matplotlib

Following the figure/axes structure, we draw the cafe-chain sales as
a line chart (monthly trend), a bar chart (store comparison), and a
scatter plot (ad spend vs revenue), saving PNGs under outputs/.
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # must come BEFORE importing pyplot — save to file, no display
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def load_sales() -> pd.DataFrame:
    """Clean the sales data and add a month column."""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    df["month"] = df["date"].str[:7]            # '2025-03' style
    print(f"[1] Data ready: {len(df):,} rows, period {df['date'].min()} ~ {df['date'].max()}")
    return df


def plot_line_monthly(df: pd.DataFrame) -> None:
    """Line chart: change over time gets a line."""
    monthly = df.groupby("month")["revenue"].sum() / 1e8  # in KRW 100 millions

    fig, ax = plt.subplots(figsize=(8, 4.5))              # paper + paragraph
    ax.plot(monthly.index, monthly.values,                # the sentence (draw the line)
            marker="o", color="#4878cf", lw=2)
    for x, y in monthly.items():                          # value labels
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title("Monthly Total Revenue Trend")           # the heading
    ax.set_xlabel("Month")
    ax.set_ylabel("Total revenue (KRW 100M)")
    ax.set_ylim(0, monthly.max() * 1.2)                   # start at 0 (a level04 preview)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "line_monthly.png")
    fig.savefig(path, dpi=120)                            # submit (save)
    plt.close(fig)                                        # put the paper away
    print(f"[2] Line chart saved: {path}")


def plot_bar_stores(df: pd.DataFrame) -> None:
    """Bar chart: category comparison gets bars. Highlight only the #1."""
    stores = (df.groupby("store")["revenue"].sum() / 1e8).sort_values(ascending=False)
    colors = ["#d1495b" if i == 0 else "#9aa7b5" for i in range(len(stores))]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(stores.index, stores.values, color=colors)
    ax.set_title("Total Revenue by Store (highlight = top store)")
    ax.set_xlabel("Store")
    ax.set_ylabel("Total revenue (KRW 100M)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "bar_stores.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] Bar chart saved: {path} (top store: {stores.index[0]})")


def plot_scatter_ad(df: pd.DataFrame) -> None:
    """Scatter plot: the relationship of two quantities gets dots. (A preview of level05.)"""
    daily = df.groupby("day_index").agg(
        ad=("ad_cost", "sum"), rev=("revenue", "sum"))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(daily["ad"] / 1e6, daily["rev"] / 1e8,
               s=18, alpha=0.5, color="#2e6f40")
    ax.set_title("Daily Ad Spend vs Daily Revenue")
    ax.set_xlabel("Ad spend (KRW millions)")
    ax.set_ylabel("Revenue (KRW 100M)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "scatter_ad.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] Scatter plot saved: {path}")
    print("    Does the dot cloud lean up and to the right? We dissect that 'relationship' in level05.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_sales()
    plot_line_monthly(df)
    plot_bar_stores(df)
    plot_scatter_ad(df)
    print("[5] Every chart was built in five steps: paper -> paragraph -> sentence -> heading -> save.")


if __name__ == "__main__":
    main()
