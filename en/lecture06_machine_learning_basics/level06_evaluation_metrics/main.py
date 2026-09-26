"""
level06 — Evaluation metrics: the accuracy trap

On card-transaction data with a ~1.5% fraud rate we
  - demonstrate the trap where a hollow model that always shouts "normal"
    reaches 98.5% accuracy, and
  - read the model properly with the confusion matrix / precision / recall /
    F1 / ROC-AUC.
We verify AUC's probabilistic reading ("probability that a random fraud/normal
pair has the fraud scored higher") directly with a random-pair simulation.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# tx_count_1h separates fraud 'perfectly' in this synthetic data, so we exclude it.
# (In real life, a feature that perfect calls for leakage suspicion, not celebration!)
FEATURES = ["amount", "hour", "is_foreign"]


def print_metrics(name: str, y_true, y_pred) -> None:
    """Print the confusion matrix and the big-four metrics in one go."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / len(y_true)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"    {name}")
    print(f"      Confusion matrix: TP={tp:>3} (caught)       FN={fn:>3} (missed!)")
    print(f"                        FP={fp:>3} (false alarm)  TN={tn:>4} (passed)")
    print(f"      accuracy {acc:.1%} / precision {prec:.1%} / recall {rec:.1%} / F1 {f1:.3f}")


def auc_by_sampling(y_true, proba, n_pairs: int = 10_000, seed: int = 0) -> float:
    """Verify AUC's definition by simulation:
    draw one fraud and one normal at random; the fraction where the fraud scored higher."""
    rng = np.random.default_rng(seed)
    pos = proba[y_true == 1]
    neg = proba[y_true == 0]
    p = rng.choice(pos, n_pairs)
    n = rng.choice(neg, n_pairs)
    return float(np.mean((p > n) + 0.5 * (p == n)))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] Prepare the data -----------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=8000, seed=11))
    X, y = df[FEATURES].astype(float), df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)  # split preserving the imbalance
    print(f"[1] Data: {len(df):,} card transactions, fraud rate {y.mean():.2%}")
    print(f"    Train {len(X_tr):,} / test {len(X_te):,} (ratio preserved via stratify)\n")

    # [2] The hollow model — predict everything normal ----------------------------
    dummy_pred = np.zeros(len(y_te), dtype=int)
    print("[2] Hollow model: insisting everything is 'normal'")
    print_metrics("Predict all normal:", y_te, dummy_pred)
    print("      -> 98.5% accuracy with zero fraud caught. The 'accuracy trap' in the flesh.")
    print("         On imbalanced data, accuracy belongs in a footnote, not the headline.\n")

    # [3] The real model — logistic regression ------------------------------------
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(random_state=0))])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print("[3] Real model: logistic regression (threshold 0.5)")
    print_metrics("Logistic regression:", y_te, (proba >= 0.5).astype(int))
    print("      -> Accuracy differs from the hollow model by a few points, but the")
    print("         confusion matrices live in different worlds.\n")

    # [4] The threshold seesaw: precision vs recall --------------------------------
    print("[4] Sweeping the threshold: the precision-recall seesaw")
    print("    threshold   alarms   precision   recall")
    for th in [0.9, 0.7, 0.5, 0.3, 0.1]:
        pred = (proba >= th).astype(int)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        print(f"     {th:.1f}        {pred.sum():>4}    {prec:6.1%}   {rec:6.1%}")
    print("    -> Raise the sensitivity (threshold down) and misses shrink while false alarms grow.")
    print("       Where to sit is decided by whoever knows 'miss cost vs false-alarm cost'.\n")

    # [5] ROC-AUC: discernment independent of the threshold --------------------------
    auc_dummy = roc_auc_score(y_te, np.zeros(len(y_te)))
    auc_model = roc_auc_score(y_te, proba)
    auc_sim = auc_by_sampling(y_te.to_numpy(), proba)
    print("[5] ROC-AUC — the 'caliber of the model' before fixing a threshold")
    print(f"    Hollow model AUC     : {auc_dummy:.3f} (coin-flip level)")
    print(f"    Logistic AUC         : {auc_model:.3f}")
    print(f"    Simulation check     : out of 10,000 random (fraud, normal) pairs,")
    print(f"                           fraction with fraud scored higher = {auc_sim:.3f}  (matches AUC)")
    print()
    print("[6] Summary: the mandatory trio for reporting on imbalanced problems")
    print("    (1) baseline (hollow model) score  (2) confusion matrix at the operating threshold  (3) AUC")
    print("    Whenever you see a one-line '98.5% accuracy' report, demand these three.")
