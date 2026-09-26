"""
level05 — Correlation and Causation

Part 1: observe the real ad-spend/revenue correlation in the cafe sales data.
Part 2: manufacture a 'spurious correlation' created by a confounder (heat)
        with synthetic data, then watch the correlation collapse under
        stratification (analyzing within confounder bands).
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


def part1_real_correlation() -> None:
    """Correlation coefficient of ad spend vs revenue + a scatter plot."""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    daily = df.groupby("day_index").agg(ad=("ad_cost", "sum"), rev=("revenue", "sum"))

    r = np.corrcoef(daily["ad"], daily["rev"])[0, 1]
    print("[1] Part 1 — daily ad spend vs daily revenue")
    print(f"    Pearson correlation coefficient r = {r:.3f}")
    print("    -> There is a positive correlation. But this alone does not let us say")
    print("       'advertising raised revenue'. (Confounders like peak season and weekends remain.)")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(daily["ad"] / 1e6, daily["rev"] / 1e8, s=18, alpha=0.5, color="#4878cf")
    ax.set_title(f"Ad spend vs revenue (r = {r:.2f}) — correlation ≠ causation")
    ax.set_xlabel("Daily ad spend (KRW millions)")
    ax.set_ylabel("Daily revenue (KRW 100M)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ad_revenue.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] Scatter plot saved: {path}")


def part2_confounder() -> None:
    """Confounder experiment: Z(heat) -> X(ice cream), Z -> Y(swimming accidents)."""
    rng = np.random.default_rng(505)  # fixed seed
    n = 400
    noise_scale = 2.0

    heat = rng.uniform(0, 10, size=n)                     # confounder Z: heat index
    icecream = 20 + 8 * heat + rng.normal(0, 8 * noise_scale, n)   # X = f(Z)+noise
    accidents = 1 + 0.9 * heat + rng.normal(0, 0.9 * noise_scale, n)  # Y = g(Z)+noise
    # Note: icecream and accidents never reference each other at all!

    r_total = np.corrcoef(icecream, accidents)[0, 1]
    print()
    print("[3] Part 2 — manufacturing a spurious correlation")
    print("    Generation rule: X(ice cream) = f(heat)+noise, Y(accidents) = g(heat)+noise")
    print("    X and Y were generated independently of each other, and yet...")
    print(f"    overall correlation r = {r_total:.3f}  <- looks like a strong correlation!")

    # Stratify: split the heat index into 5 bands and compute correlation only WITHIN bands
    bins = np.linspace(0, 10, 6)
    labels = np.digitize(heat, bins[1:-1])                # band numbers 0–4
    print()
    print("[4] Stratification experiment — comparing only days with similar heat")
    inner_rs = []
    for k in range(5):
        mask = labels == k
        r_k = np.corrcoef(icecream[mask], accidents[mask])[0, 1]
        inner_rs.append(r_k)
        print(f"    heat band {bins[k]:.0f}~{bins[k + 1]:.0f} ({mask.sum():3d} days): r = {r_k:+.3f}")
    print(f"    mean within-band correlation = {np.mean(inner_rs):+.3f}  (collapses toward 0)")
    print("    -> The overall correlation was not a relationship between the two variables —")
    print("       it was the work of the hidden body called 'heat'.")

    # Figure: overall (distorted impression) vs colored by band (the truth)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.scatter(icecream, accidents, s=14, alpha=0.5, color="#555555")
    left.set_title(f"Seen as a whole: r = {r_total:.2f} (looks related)")
    left.set_xlabel("Ice cream sales")
    left.set_ylabel("Swimming accidents")

    cmap = ["#3b6bb5", "#5da05d", "#e1a03c", "#d1495b", "#7d4fa3"]
    for k in range(5):
        mask = labels == k
        right.scatter(icecream[mask], accidents[mask], s=14, alpha=0.6,
                      color=cmap[k], label=f"heat {bins[k]:.0f}~{bins[k + 1]:.0f} (r={inner_rs[k]:+.2f})")
    right.set_title("Seen per heat band: chaos inside each cluster")
    right.set_xlabel("Ice cream sales")
    right.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "confounder.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[5] Confounder figure saved: {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    part1_real_correlation()
    part2_confounder()
    print()
    print("[6] Conclusion: a correlation is 'a lead worth investigating', not 'evidence you may")
    print("    act on'. When you need causal evidence, run an experiment (A/B test, level10).")


if __name__ == "__main__":
    main()
