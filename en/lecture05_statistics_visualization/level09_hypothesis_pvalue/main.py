"""
level09 — Hypothesis Testing and p-values

We put the per-transaction revenue gap between two stores (Airport vs
University) on trial: is it chance?
1) Permutation test: shuffle the store labels 10,000 times and count the
   p-value directly from the 'chance-only differences' distribution
2) Compare with scipy's Welch t-test
3) A control experiment with samples that truly differ by nothing
   (the meaning of a Type I error)
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
N_PERM = 10_000


def load_two_stores() -> tuple[np.ndarray, np.ndarray]:
    rows = hjh_data.sales_table(n_days=60, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    rng = np.random.default_rng(909)
    a = rng.choice(df[df["store"] == "Airport"]["revenue"].to_numpy(), 150, replace=False)
    b = rng.choice(df[df["store"] == "University"]["revenue"].to_numpy(), 150, replace=False)
    return a.astype(float), b.astype(float)


def permutation_test(a: np.ndarray, b: np.ndarray, seed: int = 910) -> tuple[float, np.ndarray, float]:
    """Label-shuffling test: build the distribution of 'chance-only differences' by hand."""
    rng = np.random.default_rng(seed)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    fake_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        rng.shuffle(pooled)                      # the H0 world: labels mean nothing
        fake_diffs[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
    # two-sided test: the share of chance differences at least as large in absolute value
    p = (np.abs(fake_diffs) >= abs(observed)).mean()
    return p, fake_diffs, observed


def trial_of_two_stores() -> None:
    a, b = load_two_stores()
    print("[1] The defendant: 'the Airport vs University revenue gap is chance' (null hypothesis H0)")
    print(f"    Airport sample mean    {a.mean() / 1e4:8.1f} (KRW 10,000s, n={len(a)})")
    print(f"    University sample mean {b.mean() / 1e4:8.1f} (KRW 10,000s, n={len(b)})")
    print(f"    Observed difference    {(a.mean() - b.mean()) / 1e4:+8.1f} (KRW 10,000s)")

    p_perm, fake_diffs, observed = permutation_test(a, b)
    print()
    print(f"[2] Permutation test — compare with the 'chance difference' distribution from {N_PERM:,} label shuffles")
    print(f"    chance-difference distribution: mean {fake_diffs.mean() / 1e4:+.2f}, "
          f"std {fake_diffs.std() / 1e4:.2f} (KRW 10,000s)")
    print(f"    p-value (permutation) = {p_perm:.4f}")

    t_stat, p_scipy = stats.ttest_ind(a, b, equal_var=False)   # Welch t-test
    print()
    print("[3] The one-line scipy test — stats.ttest_ind(a, b, equal_var=False)")
    print(f"    t = {t_stat:.3f}, p-value (t-test) = {p_scipy:.4f}")
    print(f"    -> simulated p({p_perm:.4f}) ≈ formula p({p_scipy:.4f}). They compute the same thing.")
    verdict = ("rejected — hard to call it chance (difference accepted)" if p_scipy < 0.05
               else "not rejected — chance alone can explain it")
    print(f"    Verdict (5% significance level): H0 {verdict}")

    # Figure: the chance distribution + where the observed difference sits
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(fake_diffs / 1e4, bins=60, color="#9fbce8", edgecolor="white",
            label="differences from chance alone (10,000 label shuffles)")
    ax.axvline(observed / 1e4, color="#d1495b", lw=2,
               label=f"actual observed difference {observed / 1e4:+.1f} (KRW 10,000s)")
    ax.axvline(-observed / 1e4, color="#d1495b", lw=1, ls="--")
    ax.set_title(f"p-value = share of the tail beyond the red lines = {p_perm:.4f}")
    ax.set_xlabel("Difference of store means (KRW 10,000s)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "permutation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] Permutation distribution figure saved: {path}")


def control_experiment() -> None:
    """Two samples with NO real difference — split one store's revenue in half and test."""
    rows = hjh_data.sales_table(n_days=60, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    rng = np.random.default_rng(911)
    hongdae = df[df["store"] == "Riverside"]["revenue"].to_numpy(dtype=float).copy()
    rng.shuffle(hongdae)
    half1, half2 = hongdae[:100], hongdae[100:200]

    t_stat, p = stats.ttest_ind(half1, half2, equal_var=False)
    print()
    print("[5] Control experiment — randomly split the same Riverside revenue in half and test")
    print(f"    (by definition, no real difference) t = {t_stat:.3f}, p = {p:.4f}")
    print("    -> A large p = 'chance is a sufficient explanation'. But that is NOT")
    print("       'proof of no difference'. An acquittal is not a proof of innocence.")
    print("    -> Even this 'no-difference test' comes out p<0.05 about 5 times in 100.")
    print("       That is the meaning of the 5% significance level — a contract accepting")
    print("       Type I errors (false alarms).")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    trial_of_two_stores()
    control_experiment()


if __name__ == "__main__":
    main()
