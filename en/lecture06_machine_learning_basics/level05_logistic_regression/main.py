"""
level05 — Logistic regression: a subscription-churn probability model

From churn_table (2000 customers) we build a classifier that outputs
'churn probability'.
  - sigmoid: the funnel that turns a linear score z into a 0-1 probability
  - coefficients -> odds-ratio translation: "1 more unit of calls -> churn odds xN"
  - the threshold is set by the business, not by the model
"""

import math
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
FEATURE_KO = {"tenure_months": "tenure (months)", "monthly_fee": "monthly fee",
              "usage_days_30d": "usage days (30d)", "support_calls_30d": "support calls (30d)",
              "plan_changes": "plan changes", "auto_pay": "auto-pay enabled"}


def sigmoid(z: float) -> float:
    """The probability funnel: squeezes any score, however large, into 0-1."""
    return 1.0 / (1.0 + math.exp(-z))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] Prepare + split the data ---------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X = df[FEATURES].astype(float)     # customer_id is a meaningless column, excluded
    y = df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)  # split preserving the churn rate
    print("[1] Data: 2000 subscription customers, churn rate {:.1%}".format(y.mean()))
    print(f"    Train {len(X_tr)} / test {len(X_te)} (following level04's rules)\n")

    # [2] Watching the sigmoid funnel --------------------------------------------
    print("[2] Sigmoid: linear score z -> probability p")
    for z in [-4, -2, 0, 2, 4]:
        print(f"    z = {z:+d}  ->  p = {sigmoid(z):5.1%}")
    print("    -> A score of 0 means fifty-fifty (50%); at ±4 it's nearly certain.\n")

    # [3] Training ----------------------------------------------------------------
    # Pipeline: the scaler is fit on the training set only -> leakage prevented automatically (level04)
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(random_state=0))])
    model.fit(X_tr, y_tr)
    proba_te = model.predict_proba(X_te)[:, 1]        # churn probability
    pred_05 = (proba_te >= 0.5).astype(int)
    print("[3] Logistic regression trained — test scores (threshold 0.5)")
    print(f"    accuracy {accuracy_score(y_te, pred_05):.1%} / "
          f"precision {precision_score(y_te, pred_05):.1%} / "
          f"recall {recall_score(y_te, pred_05):.1%}\n")

    # [4] Coefficients -> odds-ratio translation ----------------------------------
    clf = model.named_steps["clf"]
    print("[4] Coefficient interpretation (standardized features: effect of a '1 std dev increase')")
    print("    Feature                  coef      odds ratio  reading")
    order = np.argsort(-np.abs(clf.coef_[0]))
    for i in order:
        coef = clf.coef_[0][i]
        orat = math.exp(coef)
        direction = "churn risk up" if coef > 0 else "churn risk down"
        print(f"    {FEATURE_KO[FEATURES[i]]:20s} {coef:+7.3f}   {orat:6.2f}x     {direction}")
    print("    -> Check whether the directions match the true signals planted in the")
    print("       data generator (usage days down, calls up, auto-pay minus).")
    print("       (Still correlation, not proof of causation.)\n")

    # [5] Individual predictions + threshold experiment -----------------------------
    print("[5] Per-customer predictions and the threshold as a business decision")
    samples = pd.DataFrame([
        {"tenure_months": 36, "monthly_fee": 9900, "usage_days_30d": 28,
         "support_calls_30d": 0, "plan_changes": 0, "auto_pay": 1},
        {"tenure_months": 3, "monthly_fee": 29900, "usage_days_30d": 2,
         "support_calls_30d": 4, "plan_changes": 2, "auto_pay": 0},
        {"tenure_months": 12, "monthly_fee": 14900, "usage_days_30d": 15,
         "support_calls_30d": 1, "plan_changes": 1, "auto_pay": 1},
    ])[FEATURES].astype(float)
    names = ["Loyal heavy user", "New + low use + call flood", "Average customer"]
    for name, p in zip(names, model.predict_proba(samples)[:, 1]):
        print(f"    {name:28s} churn probability {p:5.1%}")
    print()
    print("    Change the threshold and the same model makes different decisions:")
    print("    threshold   flagged as at-risk   precision   recall")
    for th in [0.5, 0.3]:
        pred = (proba_te >= th).astype(int)
        print(f"     {th:.1f}             {pred.sum():>4}         "
              f"{precision_score(y_te, pred):6.1%}   {recall_score(y_te, pred):6.1%}")
    print("    -> Lowering the threshold means fewer missed churners (recall up) but more")
    print("       wasted outreach (precision down). 0.5 is mere convention — outreach cost")
    print("       and customer value should set the threshold.")
