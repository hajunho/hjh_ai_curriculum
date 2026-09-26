"""
level08 — Samples and Confidence Intervals

Treating the entire cafe sales as the 'population', we
1) build a 95% confidence interval from an n=50 sample, 100 times, and count
   how often the true mean is captured,
2) confirm that the interval width shrinks in proportion to 1/sqrt(n), and
3) show that a biased sample (surveying weekends only) stays wrong no matter
   how large it grows.
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def confidence_interval(sample: np.ndarray, level: float = 0.95):
    """A t-distribution based confidence interval (safe even for small samples)."""
    n = len(sample)
    mean = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(n)          # standard error
    t_crit = stats.t.ppf((1 + level) / 2, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def ring_toss(pop: np.ndarray, mu: float) -> None:
    """[2][3] Ring toss: of 100 confidence intervals, how many capture mu?"""
    rng = np.random.default_rng(808)
    n_trials, n_sample = 100, 50
    intervals, hits = [], 0
    for _ in range(n_trials):
        sample = rng.choice(pop, size=n_sample, replace=False)
        lo, hi = confidence_interval(sample)
        contains = lo <= mu <= hi
        hits += contains
        intervals.append((lo, hi, contains))

    print()
    print(f"[2] Ring toss — building 95% confidence intervals from n={n_sample} samples, {n_trials} times")
    print(f"    intervals containing the true mean μ: {hits}/{n_trials} = {hits / n_trials:.0%}")
    print("    -> '95%' is not the probability of one interval — it is the long-run hit rate")
    print("       of this interval-building METHOD.")

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (lo, hi, ok) in enumerate(intervals):
        color = "#9aa7b5" if ok else "#d1495b"
        ax.plot([lo / 1e4, hi / 1e4], [i, i], color=color, lw=1.6)
    ax.axvline(mu / 1e4, color="#2e6f40", lw=2,
               label=f"true mean μ = {mu / 1e4:.1f} (KRW 10,000s)")
    ax.set_title(f"{n_trials - hits} of {n_trials} intervals missed μ (red intervals)")
    ax.set_xlabel("Revenue per transaction (KRW 10,000s)")
    ax.set_ylabel("Experiment number")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ci_rings.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] Ring figure saved: {path}")


def width_vs_n(pop: np.ndarray) -> None:
    """[4] Interval width by sample size — confirming the 1/sqrt(n) law."""
    rng = np.random.default_rng(809)
    sizes = [20, 50, 200, 800]
    widths = []
    print()
    print("[4] Sample size and interval width (to halve the width, quadruple the sample)")
    for n in sizes:
        # repeat 200 times at each size to measure the average width (removes one-off luck)
        w = [np.subtract(*confidence_interval(rng.choice(pop, n, replace=False))[::-1])
             for _ in range(200)]
        widths.append(np.mean(w))
        print(f"    n = {n:>4}: average interval width {np.mean(w) / 1e4:8.2f} (KRW 10,000s)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, [w / 1e4 for w in widths], marker="o", color="#4878cf", label="measured width")
    ref = widths[0] * np.sqrt(sizes[0]) / np.sqrt(np.array(sizes))
    ax.plot(sizes, ref / 1e4, ls="--", color="#d1495b", label="theory: 1/√n curve")
    ax.set_title("Confidence interval width scales as 1/√n")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("95% interval width (KRW 10,000s)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ci_width.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Width figure saved: {path}")


def biased_sampling(df: pd.DataFrame, mu: float) -> None:
    """[5] Biased sample: survey only weekends and you stay wrong at any size."""
    rng = np.random.default_rng(810)
    weekend = df[df["weekday"].isin(["Sat", "Sun"])]["revenue"].to_numpy()
    print()
    print("[5] Bias experiment — confidence intervals from a 'weekends-only' survey")
    for n in [50, 400, 2000]:
        # keep surveying weekend customers only (sampling with replacement mimics scaling up)
        sample = rng.choice(weekend, size=n, replace=True)
        lo, hi = confidence_interval(sample)
        verdict = "captured" if lo <= mu <= hi else "missed!"
        print(f"    n = {n:>4}: [{lo / 1e4:7.1f}, {hi / 1e4:7.1f}] (KRW 10,000s) -> μ {verdict}")
    print(f"    (true μ = {mu / 1e4:.1f}, KRW 10,000s)")
    print("    -> The intervals get narrower and more confident — and stay consistently wrong.")
    print("       Bias is not fixed by sample size. 'How was it drawn?' comes first.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    pop = df["revenue"].to_numpy(dtype=float)
    mu = pop.mean()
    print(f"[1] Population ready: {len(pop):,} sales rows, true mean μ = {mu / 1e4:.1f} (KRW 10,000s)")
    print("    (In real life μ is unknown — today we know it, for the sake of the experiment.)")

    ring_toss(pop, mu)
    width_vs_n(pop)
    biased_sampling(df, mu)


if __name__ == "__main__":
    main()
