"""
level11 — Bayesian Thinking and Communicating Uncertainty

1) Saves beta_update.png showing a belief about a conversion rate (a Beta
   distribution) narrowing as data accumulates (prior -> evidence -> posterior).
2) From the posteriors of options A and B, computes P(B>A), the expected
   lift, and the expected loss by Monte Carlo, and auto-generates an
   executive-ready report.
"""

import os

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def belief_growth() -> None:
    """[1] A service with a true 3.2% conversion rate: belief narrowing as visitors accumulate."""
    rng = np.random.default_rng(1111)
    true_rate = 0.032
    visitors = rng.random(10_000) < true_rate      # conversion outcome per visitor
    stages = [0, 100, 1_000, 10_000]
    colors = ["#9aa7b5", "#e1a03c", "#4878cf", "#d1495b"]

    print("[1] The growth of a belief — starting from the prior Beta(1,1), adding observations")
    print(f"    (true conversion rate {true_rate:.1%} — unknown in real life)")
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0, 0.10, 800)
    for n, color in zip(stages, colors):
        s = int(visitors[:n].sum())               # conversions
        f = n - s                                 # non-conversions
        alpha, beta = 1 + s, 1 + f                # the update rule: just add!
        post = stats.beta(alpha, beta)
        lo, hi = post.ppf(0.025), post.ppf(0.975)  # 95% credible interval
        label = (f"n={n:,} (conv {s}) -> Beta({alpha},{beta})")
        ax.plot(xs, post.pdf(xs), color=color, lw=2, label=label)
        print(f"    n={n:>6,}: conversions {s:>3} | posterior Beta({alpha:>4},{beta:>5}) | "
              f"95% credible interval [{lo:.3%}, {hi:.3%}]")
    ax.axvline(true_rate, color="black", ls=":", lw=1.5, label="true conversion rate 3.2%")
    ax.set_xlim(0, 0.10)
    ax.set_title("As data accumulates, the belief distribution narrows (prior → posterior)")
    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Belief density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "beta_update.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Distribution update figure saved: {path}")
    print("    -> A credible interval is one you MAY describe as 'the true value is in here")
    print("       with 95% probability'.")


def ab_bayesian() -> tuple[float, float, float]:
    """[2] A/B posterior Monte Carlo: P(B>A), expected lift, expected loss."""
    rng = np.random.default_rng(1112)
    n_a, s_a = 20_000, 610      # option A: 20k visits, 610 conversions (3.05%)
    n_b, s_b = 20_000, 668      # option B: 20k visits, 668 conversions (3.34%)

    post_a = stats.beta(1 + s_a, 1 + n_a - s_a)
    post_b = stats.beta(1 + s_b, 1 + n_b - s_b)

    # Monte Carlo: draw 100k pairs from the two belief distributions and 'count'
    draws_a = post_a.rvs(100_000, random_state=rng)
    draws_b = post_b.rvs(100_000, random_state=rng)
    diff = draws_b - draws_a

    p_b_better = (diff > 0).mean()
    expected_lift = diff.mean()
    # Expected loss: the average damage if we pick B but A was actually better (pp)
    expected_loss = np.maximum(-diff, 0).mean()

    print()
    print("[2] Bayesian A/B verdict — A: 610/20,000 (3.05%) vs B: 668/20,000 (3.34%)")
    print(f"    P(B beats A)              = {p_b_better:.1%}")
    print(f"    expected lift             = {expected_lift * 100:+.3f} pp")
    print(f"    expected loss if we pick B = {expected_loss * 100:.4f} pp (the size of the risk when wrong)")

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0.025, 0.045, 800)
    ax.plot(xs, post_a.pdf(xs), color="#9aa7b5", lw=2, label="posterior of A (3.05%)")
    ax.fill_between(xs, post_a.pdf(xs), color="#9aa7b5", alpha=0.25)
    ax.plot(xs, post_b.pdf(xs), color="#d1495b", lw=2, label="posterior of B (3.34%)")
    ax.fill_between(xs, post_b.pdf(xs), color="#d1495b", alpha=0.25)
    ax.set_title(f"The two beliefs — the overlap is 'the uncertainty still left' (P(B>A)={p_b_better:.0%})")
    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Belief density")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ab_posterior.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Posterior comparison figure saved: {path}")
    return p_b_better, expected_lift, expected_loss


def executive_summary(p_b: float, lift: float, loss: float) -> None:
    """[3] Auto-generate an executive report in the format of section 3.4."""
    print()
    print("[3] Executive report (auto-generated)")
    print("    ------------------------------------------------------------")
    print(f"    1. On current data, the probability that option B is superior is {p_b:.0%}.")
    print(f"    2. The expected lift is {lift * 100:+.2f} pp of conversion; should B turn out")
    print(f"       inferior, the expected loss is limited to {loss * 100:.3f} pp.")
    print("    3. Switching now carries little risk; one more week of data would raise our")
    print("       confidence further. We request a decision on the switch date.")
    print("       (prior belief: uninformative Beta(1,1))")
    print("    ------------------------------------------------------------")
    print("    -> Without a single p-value, the probability, size, and risk a decision needs are all here.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    belief_growth()
    p_b, lift, loss = ab_bayesian()
    executive_summary(p_b, lift, loss)


if __name__ == "__main__":
    main()
