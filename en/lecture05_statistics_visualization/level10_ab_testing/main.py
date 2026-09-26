"""
level10 — Designing and Interpreting A/B Tests

A conversion-rate A/B test simulator.
1) z-tests for a real effect (3.0% vs 3.6%) and for no effect (A/A)
2) Repeat the no-effect test 2,000 times -> verify the Type I error rate
   is 5%, as designed
3) Peeking (checking mid-test and stopping early when significant)
   -> an experiment showing the false-positive rate exploding
"""

import os

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def z_test_two_proportions(conv_a: int, n_a: int, conv_b: int, n_b: int) -> tuple[float, float]:
    """A z-test for the difference of two conversion rates (by hand). Returns (z, p)."""
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)          # H0: the two rates are equal
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    p_value = 2 * stats.norm.sf(abs(z))               # two-sided test
    return z, p_value


def run_single_test(rate_a: float, rate_b: float, n: int, seed: int, label: str) -> None:
    """Run one test with n users per group and give the verdict."""
    rng = np.random.default_rng(seed)
    conv_a = rng.binomial(n, rate_a)
    conv_b = rng.binomial(n, rate_b)
    z, p = z_test_two_proportions(conv_a, n, conv_b, n)
    verdict = "declare B the winner (significant)" if p < 0.05 else "withhold judgment (not significant)"
    print(f"    {label}: A {conv_a / n:.3%} vs B {conv_b / n:.3%} "
          f"(difference {(conv_b - conv_a) / n:+.3%})")
    print(f"      z = {z:+.2f}, p = {p:.4f} -> {verdict}")


def type1_error_and_peeking() -> None:
    """2,000 no-effect (A/A) tests — the honest test vs peeking."""
    rng = np.random.default_rng(1010)
    n_sims = 2_000
    rate = 0.03
    total_n = 20_000          # final sample per group
    step = 1_000              # peeking check interval
    checkpoints = np.arange(step, total_n + 1, step)

    honest_fp = 0            # honest: test once, at the end
    peeking_fp = 0           # peeking: stop the moment it looks significant
    sample_trajectories = []  # a few p-value trajectories for the figure

    for sim in range(n_sims):
        # cumulative conversions as each batch of `step` users arrives (no effect in either group)
        inc_a = rng.binomial(step, rate, size=len(checkpoints))
        inc_b = rng.binomial(step, rate, size=len(checkpoints))
        cum_a, cum_b = np.cumsum(inc_a), np.cumsum(inc_b)

        p_traj = np.array([
            z_test_two_proportions(ca, n, cb, n)[1]
            for ca, cb, n in zip(cum_a, cum_b, checkpoints)
        ])
        if p_traj[-1] < 0.05:
            honest_fp += 1
        if (p_traj < 0.05).any():
            peeking_fp += 1
        if sim < 12:
            sample_trajectories.append(p_traj)

    print()
    print(f"[3] Verifying the Type I error rate — {n_sims:,} repetitions of a no-effect test")
    print(f"    Honest test (once, at the end):  false positives {honest_fp:>4} "
          f"= {honest_fp / n_sims:.1%}  (near the designed 5%)")
    print()
    print(f"[4] Peeking — check every {step:,} users, 'declare victory and stop' the moment it is significant")
    print(f"    Peeking policy:                  false positives {peeking_fp:>4} "
          f"= {peeking_fp / n_sims:.1%}  (about {peeking_fp / max(honest_fp, 1):.1f}x explosion!)")
    print("    -> Same data, same significance level — yet 'when you look' alone multiplies the")
    print("       false alarms. The stopping time must be fixed before seeing any results.")

    # Figure: p-value trajectories — the moments they briefly pierce the 0.05 line are the bait
    fig, ax = plt.subplots(figsize=(9, 5))
    for traj in sample_trajectories:
        dipped = (traj < 0.05).any()
        ax.plot(checkpoints, traj, lw=1.6 if dipped else 1.0,
                color="#d1495b" if dipped else "#9aa7b5", alpha=0.85)
    ax.axhline(0.05, color="black", ls="--", lw=1, label="significance level 0.05")
    ax.set_ylim(0, 1)
    ax.set_title("p-value trajectories of 12 tests with NO effect — the red ones are peeking's prey")
    ax.set_xlabel("Cumulative sample per group")
    ax.set_ylabel("p-value at that moment")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "peeking.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[5] p-value trajectory figure saved: {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] With a real effect — true conversion A 3.0% vs B 3.6%, 20,000 users per group")
    run_single_test(0.030, 0.036, 20_000, seed=101, label="main test")
    print()
    print("[2] With no effect (A/A) — both at 3.0%, 20,000 users per group")
    run_single_test(0.030, 0.030, 20_000, seed=102, label="A/A test")

    type1_error_and_peeking()
    print()
    print("[6] Field summary: fix the metric, minimum effect, sample size, and stopping rule in a")
    print("    document BEFORE the experiment, and do not look at the p-value until the sample is")
    print("    full — the cheapest false-alarm prevention there is.")


if __name__ == "__main__":
    main()
