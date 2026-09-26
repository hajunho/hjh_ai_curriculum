"""
level00 — Summarizing Reality with Numbers

We build four revenue datasets (A–D) that all share the same mean of 100
and confirm that the summary value (the mean) alone cannot tell these
completely different realities apart.
We then draw the 'shape' of each dataset with a text histogram.
"""

import random
import statistics


def make_datasets() -> dict[str, list[float]]:
    """Four daily-revenue datasets engineered so every mean equals 100 (in KRW 10,000s)."""
    rng = random.Random(500)  # fixed seed for reproducibility
    n = 200

    # A. Steady: clustered evenly around 100 (a shop with loyal regulars)
    a = [rng.gauss(100, 8) for _ in range(n)]

    # B. Two clusters: weekday crowd (~60) and weekend crowd (~140), half and half
    b = [rng.gauss(60, 7) for _ in range(n // 2)] + \
        [rng.gauss(140, 7) for _ in range(n // 2)]

    # C. Outliers: usually around 80, a few big contracts drag the mean up
    c = [rng.gauss(80, 6) for _ in range(n - 4)] + [1200, 1150, 900, 850]

    # D. Hit-or-miss: spread widely between 40 and 160
    d = [rng.uniform(40, 160) for _ in range(n)]

    data = {"A(steady)": a, "B(two clusters)": b, "C(outliers)": c, "D(hit-or-miss)": d}
    # Correct for tiny random error so each mean is exactly 100.
    for name, values in data.items():
        shift = 100 - statistics.mean(values)
        data[name] = [v + shift for v in values]
    return data


def summarize(values: list[float]) -> dict[str, float]:
    """Compute the most basic summary statistics."""
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def ascii_hist(values: list[float], lo: float, hi: float,
               n_bins: int = 12, width: int = 40) -> None:
    """A text histogram: split into bins, count frequencies, draw asterisks."""
    counts = [0] * n_bins
    step = (hi - lo) / n_bins
    for v in values:
        idx = int((v - lo) / step)
        idx = max(0, min(n_bins - 1, idx))  # out-of-range values go to the end bins
        counts[idx] += 1
    peak = max(counts) or 1
    for i, cnt in enumerate(counts):
        left = lo + i * step
        bar = "*" * round(cnt / peak * width)
        print(f"  {left:7.1f} ~ {left + step:7.1f} | {bar} ({cnt})")


def main() -> None:
    data = make_datasets()

    print("[1] Looking at the summary (mean) alone, the four shops are identical.")
    for name, values in data.items():
        print(f"    {name:<16} mean daily revenue = {statistics.mean(values):7.1f} (KRW 10,000s)")

    print()
    print("[2] Add a few more summary values and the differences start to show.")
    header = f"    {'dataset':<16} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name, values in data.items():
        s = summarize(values)
        print(f"    {name:<16} {s['mean']:>8.1f} {s['median']:>8.1f} "
              f"{s['min']:>8.1f} {s['max']:>8.1f}")
    print("    -> For C, the mean (100) and the median (about 80) disagree badly.")
    print("       That is the signal that a few extreme values pulled the mean up.")

    print()
    print("[3] Look at the shapes directly — text histograms (range 0–200)")
    for name, values in data.items():
        print(f"  <{name}>")
        ascii_hist(values, lo=0, hi=200)
        print()

    print("[4] Questions a summary cannot answer")
    print("    Q. Which shop needs more weekend-evening staff?      -> B (two clusters)")
    print("    Q. Which shop should worry about big-contract risk?  -> C (outliers)")
    print("    Q. Which shop needs a cash-flow buffer?              -> D (hit-or-miss)")
    print("    All four shops share the same 'mean' of 100, yet the right decision differs every time.")
    print("    => A summary is only a starting point — always check spread, shape, and outliers.")


if __name__ == "__main__":
    main()
