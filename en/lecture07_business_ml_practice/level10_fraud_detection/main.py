"""
Case study 3 — fraud detection: supervised learning alongside unsupervised (IsolationForest).
Fairly compares the detection power of both approaches under the same alert budget,
inspects the 'suspected novel scheme zone' (rated low by supervised, high by unsupervised),
then designs the operating threshold with a recall-vs-alert-budget marginal-benefit table.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Same assumption as level07: no history-aggregation (tx_count_1h) system yet,
# so we fight with payment-time fields + level03's derived features.
FEATURES = ["log_amount", "hour", "is_foreign", "is_night"]


def topk_stats(scores, y_true, k):
    """(frauds caught, precision, recall) when alerting only the top-k suspicion scores."""
    idx = np.argsort(scores)[::-1][:k]
    caught = int(np.asarray(y_true)[idx].sum())
    total = int(np.asarray(y_true).sum())
    return caught, caught / k, caught / total


def main() -> None:
    print("=" * 68)
    print(" Case study 3: fraud detection — the album (supervised) + instinct (unsupervised) + budget (operations)")
    print("=" * 68)

    # [1] Data and features --------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    df["log_amount"] = np.log1p(df["amount"])
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)
    X, y = df[FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    print(f"\n[1] {len(df)} transactions (fraud {y.mean():.2%}) / test set {len(y_te)} with {y_te.sum()} fraud")
    print(f"    features: {FEATURES}")

    # [2] Supervised: the veteran who learns from the label album --------------
    print("\n[2] Supervised — trained on historical caught-fraud labels (logistic regression)")
    sup = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=42, class_weight="balanced",
                                           max_iter=1000))
    sup.fit(X_tr, y_tr)
    sup_score = sup.predict_proba(X_te)[:, 1]
    print("    -> produced a fraud probability (suspicion score) per transaction.")

    # [3] Unsupervised: the rookie who spots 'different from usual' without labels ----
    print("\n[3] Unsupervised — IsolationForest, using no labels at all")
    iso = IsolationForest(n_estimators=200, contamination=0.015, random_state=42)
    iso.fit(X_tr)                      # no y_tr! learns only the shape of normal transactions
    iso_score = -iso.score_samples(X_te)   # larger = more anomalous (sign flipped)
    print("    -> produced anomaly scores from 'how few cuts until isolated'.")

    # [4] A fair comparison under the same alert budget ---------------------------
    budget = 20
    print(f"\n[4] Fair comparison: alert budget of {budget} out of the {len(y_te)} test transactions")
    for name, sc in [("supervised (uses labels)", sup_score), ("IsolationForest (no labels)", iso_score)]:
        caught, prec, rec = topk_stats(sc, y_te, budget)
        print(f"    {name:<28} frauds caught {caught:2d} | precision {prec:5.1%} | recall {rec:5.1%}")
    print("    => Labels are powerful. But the unsupervised model looks from a different")
    print("       angle — 'schemes not in the labels'. Not winner-take-all: a division of labor.")

    # [5] Suspected novel scheme zone: low supervised, high unsupervised ----------
    print("\n[5] Suspected novel scheme zone — bottom 50% supervised score but top 5% unsupervised score")
    sup_rank = pd.Series(sup_score).rank(pct=True)
    iso_rank = pd.Series(iso_score).rank(pct=True)
    novel = (sup_rank < 0.5) & (iso_rank > 0.95)
    zone = X_te.reset_index(drop=True)[novel]
    zone_y = y_te.reset_index(drop=True)[novel]
    print(f"    matching transactions: {novel.sum()} (of which {int(zone_y.sum())} actual fraud)")
    if len(zone) > 0:
        show = zone.head(3).copy()
        show["actual_fraud"] = zone_y.head(3).values
        print(show.to_string())
    print("    => Send this zone to a small sampled-investigation queue. The findings become")
    print("       new labels that re-teach the supervised model — that virtuous cycle is the core of real operations.")

    # [6] Marginal benefit per alert budget and the operating threshold -------------
    print("\n[6] How much more do we catch as the alert budget grows (supervised model)")
    print(f"    {'K':>4} | {'caught':>6} | {'precis':>6} | {'recall':>6} | threshold at that budget")
    prev_caught = 0
    for k in [10, 20, 30, 40, 60]:
        caught, prec, rec = topk_stats(sup_score, y_te, k)
        th = np.sort(sup_score)[::-1][k - 1]
        gain = caught - prev_caught
        print(f"    {k:>4} | {caught:>6} | {prec:6.1%} | {rec:6.1%} | "
              f"score >= {th:.3f} (+{gain} vs previous)")
        prev_caught = caught
    print("\n    Example reporting sentences:")
    c20, p20, r20 = topk_stats(sup_score, y_te, 20)
    c40, p40, r40 = topk_stats(sup_score, y_te, 40)
    print(f'    "With current staffing (20 alerts) we catch {r20:.0%} of fraud. Doubling the')
    print(f'     investigation team (40 alerts) raises recall to {r40:.0%} but precision falls to {p40:.0%}."')
    print("\n    Lesson: the threshold does not come from statistics. It comes from workforce planning (the budget).")


if __name__ == "__main__":
    main()
