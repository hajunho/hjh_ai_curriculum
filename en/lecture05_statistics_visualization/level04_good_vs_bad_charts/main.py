"""
level04 — Good Charts vs Bad Charts

We draw a 'distorted chart' and an 'honest chart' side by side from
exactly the same data.
1) Axis truncation: making a 3% gap look like a landslide
2) A 3D-style pie chart vs a sorted horizontal bar chart
3) Manipulating the y-axis range to turn tiny wiggles into a roller coaster
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


def load_data():
    rows = hjh_data.sales_table(n_days=90, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    store_rev = (df.groupby("store")["revenue"].sum() / 1e8)  # KRW 100 millions
    return df, store_rev


def crime1_truncated_axis(store_rev: pd.Series) -> None:
    """Crime #1: truncating the bar chart's y-axis."""
    two = store_rev.sort_values(ascending=False).iloc[[1, 2]]  # two stores with a small gap
    a, b = two.index
    gap_pct = (two[a] - two[b]) / two[b] * 100

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#d1495b", "#9aa7b5"]

    bad.bar(two.index, two.values, color=colors)
    bad.set_ylim(two.min() * 0.985, two.max() * 1.005)   # axis truncation!
    bad.set_title(f"[Distorted] Truncated axis — a {gap_pct:.1f}% gap looks like a rout")
    bad.set_ylabel("Total revenue (KRW 100M)")

    good.bar(two.index, two.values, color=colors)
    good.set_ylim(0, two.max() * 1.15)                   # start at 0
    good.set_title("[Honest] Bars start at 0 — the real gap")
    good.set_ylabel("Total revenue (KRW 100M)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "truncated_axis.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] Truncated-axis comparison saved: {path}")
    print(f"    The real gap is only {gap_pct:.1f}%. Compare that with the left panel's first impression.")


def crime2_pie_vs_bar(store_rev: pd.Series) -> None:
    """Crime #2: a pie chart of similar values vs a sorted horizontal bar chart."""
    share = store_rev / store_rev.sum() * 100

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.8))
    # shadow + exploded slice = a recreation of the conference-room 3D pie
    bad.pie(share.values, labels=share.index, shadow=True,
            explode=[0.08 if i == 0 else 0 for i in range(len(share))],
            startangle=90)
    bad.set_title("[Distorted] Pie — can you tell which slice is #2?")

    ordered = share.sort_values()
    good.barh(ordered.index, ordered.values, color="#4878cf")
    for y, v in enumerate(ordered.values):
        good.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    good.set_title("[Honest] Sorted bars — rank and gaps visible at once")
    good.set_xlabel("Revenue share (%)")
    good.set_xlim(0, ordered.max() * 1.25)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pie_vs_bar.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] Pie vs bar saved: {path}")


def crime3_inflated_line(df: pd.DataFrame) -> None:
    """Crime #3: squeezing the y-axis range so tiny wiggles look like a roller coaster."""
    daily = df.groupby("day_index")["revenue"].sum() / 1e8
    weekly = daily.rolling(7).mean().dropna()            # smooth with a 7-day moving average

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.2))
    bad.plot(weekly.index, weekly.values, color="#d1495b", lw=2)
    bad.set_ylim(weekly.min() * 0.998, weekly.max() * 1.002)  # range manipulation!
    bad.set_title("[Distorted] Zoomed axis — a drama of crash and comeback?")
    bad.set_xlabel("Day")
    bad.set_ylabel("Daily revenue, 7-day avg (KRW 100M)")

    good.plot(weekly.index, weekly.values, color="#4878cf", lw=2)
    good.set_ylim(0, weekly.max() * 1.2)
    good.set_title("[Honest] Range with context — essentially stable")
    good.set_xlabel("Day")
    good.set_ylabel("Daily revenue, 7-day avg (KRW 100M)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "inflated_line.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    swing = (weekly.max() - weekly.min()) / weekly.mean() * 100
    print(f"[4] Line-chart range manipulation saved: {path}")
    print(f"    The real swing is only about ±{swing / 2:.1f}% around the mean.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df, store_rev = load_data()

    print("[1] The real numbers behind it all (total revenue per store, KRW 100M)")
    for store, v in store_rev.sort_values(ascending=False).items():
        print(f"    {store:<12} {v:8.2f}")
    print("    -> Every 'distorted' and 'honest' picture below comes from these same numbers.")

    crime1_truncated_axis(store_rev)
    crime2_pie_vs_bar(store_rev)
    crime3_inflated_line(df)
    print("[5] Conclusion: if a chart's first impression differs from a careful reading of the numbers, it is a bad chart.")


if __name__ == "__main__":
    main()
