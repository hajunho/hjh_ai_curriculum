"""
A comparison of techniques for imbalanced data (1.5% fraud).
Checks the recall-precision trade-off across four setups —
baseline model / class_weight / oversampling / threshold tuning —
and picks a reasonable threshold under a 'daily alert-handling budget' constraint.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             average_precision_score, precision_recall_curve)

# Assume no history-aggregation system exists yet, so only 'fields known at payment time' are usable
# (how much easier the problem gets with the aggregate feature tx_count_1h is a Try-it-yourself exercise)
FEATURES = ["amount", "hour", "is_foreign"]


def new_model(**kw):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=42, max_iter=1000, **kw))


def show(name, y_true, y_pred):
    print(f"    {name:<28} accuracy {accuracy_score(y_true, y_pred):.3f} | "
          f"recall {recall_score(y_true, y_pred):.3f} | "
          f"precision {precision_score(y_true, y_pred, zero_division=0):.3f} | "
          f"alerts {int(y_pred.sum())}")


def main() -> None:
    print("=" * 66)
    print(" Imbalanced data: four ways to catch the 1.5% of fraud")
    print("=" * 66)

    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    X, y = df[FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    print(f"\n    data: {len(df)} transactions, {y.sum()} fraud ({y.mean():.2%}) / "
          f"test set {len(y_te)} with {y_te.sum()} fraud")

    # [1] Default settings: the lazy model ------------------------------------
    print("\n[1] Default settings (threshold 0.5, no weights)")
    base = new_model()
    base.fit(X_tr, y_tr)
    show("baseline logistic regression", y_te, base.predict(X_te))
    missed = int(y_te.sum()) - int((base.predict(X_te) & y_te).sum())
    print(f"    => The 99%-range accuracy looks great, yet {missed} of the {int(y_te.sum())} frauds are missed.")
    print("       (at the extreme, predicting 'all normal' also scores 98.6% accuracy — accuracy is not this problem's metric)")

    # [2] class_weight: rewriting the penalty table ------------------------------
    print("\n[2] class_weight='balanced' — big penalty for minority-class mistakes")
    weighted = new_model(class_weight="balanced")
    weighted.fit(X_tr, y_tr)
    show("weighted model", y_te, weighted.predict(X_te))

    # [3] Oversampling: training data only! ----------------------------------------
    print("\n[3] Manual oversampling — duplicate the training data's fraud rows only, AFTER the split")
    rng = np.random.default_rng(42)
    pos_idx = y_tr[y_tr == 1].index
    ratio = int((y_tr == 0).sum() / (y_tr == 1).sum())  # a multiple that roughly balances the classes
    dup_idx = rng.choice(pos_idx, size=len(pos_idx) * (ratio - 1), replace=True)
    X_bal = pd.concat([X_tr, X_tr.loc[dup_idx]])
    y_bal = pd.concat([y_tr, y_tr.loc[dup_idx]])
    print(f"    training data {len(y_tr)} rows -> {len(y_bal)} rows (fraud share {y_bal.mean():.1%})")
    over = new_model()
    over.fit(X_bal, y_bal)
    show("oversampled model", y_te, over.predict(X_te))
    print("    => Similar effect to [2]. Both are variants of 'rewriting the penalty table'.")
    print("       (never doctor the test ratio — the exam happens at the real world's ratio)")

    # [4] Threshold tuning: same model, different operating point --------------------
    print("\n[4] Threshold tuning — one baseline model, only the operating point moves")
    proba = base.predict_proba(X_te)[:, 1]
    print(f"    {'thresh':>6} | {'recall':>6} | {'precis':>6} | alerts")
    rows = []
    for th in [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]:
        pred = (proba >= th).astype(int)
        r = recall_score(y_te, pred)
        p = precision_score(y_te, pred, zero_division=0)
        rows.append((th, r, p, int(pred.sum())))
        print(f"    {th:6.2f} | {r:6.1%} | {p:6.1%} | {int(pred.sum()):4d}")
    print("    => Going down the table, recall rises and precision falls. There is no free lunch.")

    # [5] PR-curve summary and the alert budget --------------------------------------
    print("\n[5] PR-curve summary and choosing an operating point with an 'alert budget'")
    pr_auc = average_precision_score(y_te, proba)
    prec_c, rec_c, th_c = precision_recall_curve(y_te, proba)
    print(f"    PR-AUC (average precision) = {pr_auc:.3f}  (the overall score across all thresholds)")
    budget = 15  # the number of alerts the investigation team can handle over this test period
    order = np.argsort(proba)[::-1]
    top = order[:budget]
    caught = int(y_te.iloc[top].sum())
    th_budget = proba[order[budget - 1]]
    print(f"    constraint: investigation-team capacity = {budget} alerts")
    print(f"    => alert only on the top {budget} suspicion scores (threshold about {th_budget:.2f})")
    print(f"       frauds caught {caught} / total {int(y_te.sum())} "
          f"(recall {caught / y_te.sum():.1%}, precision {caught / budget:.1%})")
    print("\n    Lesson: which operating point to choose is decided by the business (people, cost), not the model.")
    print("            The model's job ends at producing a good curve.")


if __name__ == "__main__":
    main()
