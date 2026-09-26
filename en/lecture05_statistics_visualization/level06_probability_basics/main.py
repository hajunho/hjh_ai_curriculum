"""
level06 — Probability Fundamentals

Three simulations pit intuition against calculation.
1) The law of large numbers: coin-flip ratios converging
2) The disease-testing paradox: of those testing positive, how many are
   actually sick? (a taste of Bayes)
3) The Monty Hall problem: win rates of switching vs staying
"""

import numpy as np


def law_of_large_numbers() -> None:
    """As the number of coin flips grows, the heads ratio approaches 0.5."""
    rng = np.random.default_rng(606)  # fixed seed
    flips = rng.integers(0, 2, size=10_000)  # 1=heads
    print("[1] Law of large numbers — convergence of the heads ratio")
    for n in [10, 100, 1_000, 10_000]:
        ratio = flips[:n].mean()
        print(f"    {n:>6,} flips: heads ratio {ratio:.4f} (distance from 0.5: {abs(ratio - 0.5):.4f})")
    print("    -> A probability of 0.5 is not a prophecy about 'the next flip' —")
    print("       it is a promise about the long-run ratio.")


def disease_test_paradox() -> None:
    """Run a test with 1% prevalence, 99% sensitivity, 95% specificity on 100,000 people."""
    rng = np.random.default_rng(607)
    n = 100_000
    prevalence = 0.01     # prevalence: prior probability of being sick
    sensitivity = 0.99    # P(positive|sick) — probability of catching a sick person
    specificity = 0.95    # P(negative|healthy) — probability of clearing a healthy person

    sick = rng.random(n) < prevalence
    # Sick people test positive with the sensitivity; healthy people get an
    # unfair positive with probability (1 - specificity)
    positive = np.where(sick,
                        rng.random(n) < sensitivity,
                        rng.random(n) < (1 - specificity))

    n_pos = positive.sum()
    n_true = (sick & positive).sum()          # positives who are actually sick
    n_false = (~sick & positive).sum()        # unfair positives (false alarms)
    p_sick_given_pos = n_true / n_pos

    # Compare with the theoretical value from Bayes' theorem
    theory = (prevalence * sensitivity) / (
        prevalence * sensitivity + (1 - prevalence) * (1 - specificity))

    print()
    print("[2] The disease-testing paradox — simulating a population of 100,000")
    print(f"    sick {sick.sum():,} / healthy {(~sick).sum():,} (prevalence {prevalence:.0%})")
    print(f"    tested positive {n_pos:,} = actually sick {n_true:,} + unfair positives {n_false:,}")
    print(f"    P(sick | positive), simulated   = {p_sick_given_pos:.3f}")
    print(f"    P(sick | positive), Bayes theory = {theory:.3f}")
    print("    -> Even with a '99% accurate test', a positive result means only ~17% chance of disease!")
    print("       Healthy people vastly outnumber the sick, so their 5% false-alarm rate")
    print("       swamps the number of actual patients.")
    print("       (P(positive|sick)=0.99 and P(sick|positive)=0.17 are completely different probabilities)")


def monty_hall(n_games: int = 10_000) -> None:
    """Monty Hall: 3 doors, and the host knowingly opens a losing door."""
    rng = np.random.default_rng(608)
    prize = rng.integers(0, 3, size=n_games)       # the door hiding the prize
    first_pick = rng.integers(0, 3, size=n_games)  # the contestant's first choice

    # 'Stay' strategy: win if the first pick was correct
    stay_wins = (first_pick == prize).sum()
    # 'Switch' strategy: win if the first pick was NOT correct
    #   (the host opens the remaining losing door, so switching lands on the prize)
    switch_wins = (first_pick != prize).sum()

    print()
    print(f"[3] The Monty Hall problem — {n_games:,} simulated games")
    print(f"    Stay strategy   win rate: {stay_wins / n_games:.3f} (theory 1/3 = 0.333)")
    print(f"    Switch strategy win rate: {switch_wins / n_games:.3f} (theory 2/3 = 0.667)")
    print("    -> The host can 'choose' to open a losing door, so that action itself is information.")
    print("       Intuition may insist on 50:50, but the ledger of ten thousand games says 2/3.")


def main() -> None:
    law_of_large_numbers()
    disease_test_paradox()
    monty_hall()
    print()
    print("[4] Today's lesson: when a probability argument breaks out, run a simulation before")
    print("    reaching for formulas. 'Build it and count' is a probability calculator that always works.")


if __name__ == "__main__":
    main()
