"""
level07 — The Normal Distribution and the Central Limit Theorem (CLT)

1) Verify the 68-95-99.7 rule with a million random numbers.
2) From a uniform and an exponential distribution, draw sample means of
   size n=1/5/30, 5,000 times each, and save one figure (a 2x3 grid,
   outputs/clt_grid.png) showing the distribution of means converging to
   a bell shape regardless of the original's shape.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SAMPLE_SIZES = [1, 5, 30]
N_REPEAT = 5_000  # how many times to draw a sample mean


def rule_68_95_997() -> None:
    """Count the 68-95-99.7 rule with a million normal random numbers."""
    rng = np.random.default_rng(707)
    z = rng.normal(0, 1, size=1_000_000)
    print("[1] Verifying the 68-95-99.7 rule (1,000,000 standard normal draws)")
    for k, expect in [(1, 68.3), (2, 95.4), (3, 99.7)]:
        ratio = (np.abs(z) <= k).mean() * 100
        print(f"    share within ±{k}σ: {ratio:5.2f}%  (theory about {expect}%)")


def normal_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """The normal probability density function (by hand, no scipy)."""
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def clt_experiment() -> None:
    """Draw the grid of sample-mean distributions for uniform and exponential."""
    rng = np.random.default_rng(708)

    # (name, sampling function, population mean, population std dev)
    uniform_spec = ("Uniform U(0,1)", lambda size: rng.uniform(0, 1, size),
                    0.5, 1 / np.sqrt(12))
    expo_spec = ("Exponential (mean 1)", lambda size: rng.exponential(1.0, size),
                 1.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    print()
    print("[2][3] Sample-mean experiment — each panel records 'the mean of n draws', 5,000 times")
    print(f"    {'source distribution':<22} {'n':>4} {'mean of means':>14} {'std of means':>14} {'theory σ/√n':>12}")

    for row, (name, sampler, mu, sigma) in enumerate([uniform_spec, expo_spec]):
        for col, n in enumerate(SAMPLE_SIZES):
            # Key move: build an (N_REPEAT, n) table and average each row -> N_REPEAT sample means
            means = sampler((N_REPEAT, n)).mean(axis=1)
            se_theory = sigma / np.sqrt(n)
            print(f"    {name:<22} {n:>4} {means.mean():>14.4f} "
                  f"{means.std():>14.4f} {se_theory:>12.4f}")

            ax = axes[row][col]
            ax.hist(means, bins=50, density=True, color="#9fbce8", edgecolor="white")
            xs = np.linspace(means.min(), means.max(), 200)
            ax.plot(xs, normal_pdf(xs, mu, se_theory), color="#d1495b", lw=2,
                    label="CLT theory curve")
            ax.set_title(f"{name}, n={n}")
            if col == 0:
                ax.set_ylabel("Density")
            if row == 0 and col == 2:
                ax.legend(fontsize=9)

    fig.suptitle("The Central Limit Theorem — whatever the source, the 'means' gather into a bell",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "clt_grid.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print()
    print(f"[4] CLT grid saved: {path}")
    print("    The left column (n=1) keeps the source's shape; the right column (n=30) hugs the bell.")
    print("[5] Check in the table above that the 'std of means' matches the theoretical σ/√n.")
    print("    -> You need 4x the sample to halve the error (the economics of √n).")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rule_68_95_997()
    clt_experiment()


if __name__ == "__main__":
    main()
