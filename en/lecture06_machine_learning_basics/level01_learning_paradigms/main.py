"""
level01 — A mini tour of supervised, unsupervised, and reinforcement learning

We run each of the three learning paradigms on a toy-sized example.
  [1] Supervised     = private tutoring    (train churn prediction with answer labels)
  [2] Unsupervised   = self-directed study (group customers with no answers given)
  [3] Reinforcement  = puppy training      (a bandit that finds the best coupon from rewards alone)
"""

import pathlib
import random
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "usage_days_30d", "support_calls_30d", "auto_pay"]


def demo_supervised() -> None:
    """[1] Supervised: teach with the problems (customer features) AND the answer key (churned)."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["churned"] for r in rows])

    scaler = StandardScaler()
    model = LogisticRegression(random_state=0)
    model.fit(scaler.fit_transform(X), y)          # <- we pass the answers y (tutoring)

    print("[1] Supervised = private tutoring (learn patterns from the answer key)")
    print("    Training: 2000 customers' features + churn labels")
    print("    Test: predict churn probability for 3 never-seen customers")
    new_customers = [
        ("Loyal heavy user", [36, 28, 0, 1]),
        ("Low usage + call flood", [3, 2, 5, 0]),
        ("Average customer", [18, 15, 1, 1]),
    ]
    for name, feat in new_customers:
        p = model.predict_proba(scaler.transform([feat]))[0, 1]
        verdict = "churn risk" if p >= 0.5 else "likely to stay"
        print(f"      {name:22s} -> churn probability {p:5.1%} ({verdict})")
    print()


def demo_unsupervised() -> None:
    """[2] Unsupervised: remove the answer column and ask only 'group similar customers'."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    cols = ["tenure_months", "usage_days_30d", "support_calls_30d", "monthly_fee"]
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float)  # no answers!

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=0)
    labels = km.fit_predict(Xs)                     # <- there is no y (self-directed study)

    print("[2] Unsupervised = self-directed study (find structure with no answers)")
    print("    k-means grouped 2000 customers into 3 clusters. Average profile per group:")
    print("      group  size   tenure_mo  usage_days  calls  monthly_fee")
    for g in range(3):
        member = X[labels == g]
        m = member.mean(axis=0)
        print(f"      {g:>2}    {len(member):>4}   {m[0]:8.1f}  {m[1]:9.1f}  {m[2]:5.2f}  {m[3]:10.0f}")
    print("    -> Naming each group (e.g. 'high-value', 'dormancy risk') is the human's job.")
    print()


def demo_reinforcement() -> None:
    """[3] Reinforcement: not knowing which coupon works, learn from sending and watching rewards.
    A multi-armed bandit with epsilon-greedy, in pure Python."""
    rng = random.Random(42)
    true_rates = {"A. 5% off": 0.05, "B. Free shipping": 0.12, "C. Buy 1 get 1": 0.08}
    coupons = list(true_rates)
    counts = {c: 0 for c in coupons}       # how many times each coupon was sent
    wins = {c: 0 for c in coupons}         # how many times each coupon was redeemed
    EPSILON = 0.1                          # 10% chance to explore (try anything)
    history = []

    for _ in range(1000):
        if rng.random() < EPSILON or not any(counts.values()):
            choice = rng.choice(coupons)                       # explore
        else:
            choice = max(coupons, key=lambda c: wins[c] / max(counts[c], 1))  # exploit
        reward = 1 if rng.random() < true_rates[choice] else 0  # customer response = the treat
        counts[choice] += 1
        wins[choice] += reward
        history.append(reward)

    print("[3] Reinforcement = puppy training (try -> reward -> better behavior)")
    print("    The 3 coupons' true redemption rates are secret. Learning across 1000 sends:")
    for c in coupons:
        est = wins[c] / max(counts[c], 1)
        print(f"      {c:16s} sent {counts[c]:>4}x, estimated rate {est:5.1%} (truth {true_rates[c]:.0%})")
    early = sum(history[:100]) / 100
    late = sum(history[-100:]) / 100
    print(f"    Avg reward, first 100 sends {early:.2f}  ->  last 100 sends {late:.2f}")
    print("    -> Nobody ever taught it the answer, yet it converges on the best action (B).")
    print()


if __name__ == "__main__":
    np.random.seed(0)   # reproducibility
    demo_supervised()
    demo_unsupervised()
    demo_reinforcement()

    print("[4] Summary")
    print("    Paradigm       Given                  Learned")
    print("    Supervised     inputs + answer labels input -> answer mapping (tutoring)")
    print("    Unsupervised   inputs only            hidden structure/groups (self-study)")
    print("    Reinforcement  environment + rewards  reward-maximizing actions (puppy training)")
    print("    Most real-world ML is supervised learning; we dive in properly at level03.")
