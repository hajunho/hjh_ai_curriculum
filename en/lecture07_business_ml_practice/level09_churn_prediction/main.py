"""
Case study 2 — the complete customer-churn-prediction project.
Problem spec -> mini-EDA -> features -> cross-validation -> final model ->
coefficient interpretation -> a 'top-10 at-risk customers + main driver +
recommended action' table: a deliverable the marketing team can execute as-is.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score

BASE = ["tenure_months", "monthly_fee", "usage_days_30d",
        "support_calls_30d", "plan_changes", "auto_pay"]
DERIVED = ["usage_per_tenure", "calls_plus_changes"]
FEAT_KO = {
    "tenure_months": "months of tenure", "monthly_fee": "monthly fee",
    "usage_days_30d": "usage days (30d)", "support_calls_30d": "support calls (30d)",
    "plan_changes": "plan changes", "auto_pay": "auto-pay",
    "usage_per_tenure": "activity vs tenure", "calls_plus_changes": "complaint signals",
}
# main driver -> recommended action (a response rulebook designed by humans, outside the model)
ACTION_MAP = {
    "usage_days_30d": "re-engagement content + 7-day free pass",
    "usage_per_tenure": "re-engagement content + 7-day free pass",
    "support_calls_30d": "priority CS consultation, resolve the complaint",
    "calls_plus_changes": "priority CS consultation, resolve the complaint",
    "plan_changes": "tailored plan-recommendation consultation",
    "monthly_fee": "tailored plan-recommendation consultation",
    "auto_pay": "20% off one month when switching to auto-pay",
    "tenure_months": "onboarding guide + first-month perks",
}


def main() -> None:
    print("=" * 70)
    print(" Case study 2: customer churn prediction — finishing with list, reason, action")
    print("=" * 70)

    # [1] Problem spec ------------------------------------------------------
    print("\n[1] Problem spec (before the code)")
    print("    target: subscription churn this month / use: weekly CRM campaign to top-risk customers")
    print("    target metric: precision 25%+ at recall 55%+ / baseline: random sends (hit rate = churn rate, about 15%)")

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))

    # [2] Mini-EDA -----------------------------------------------------------
    print(f"\n[2] Mini-EDA: {len(df)} customers, churn rate {df['churned'].mean():.1%}")
    grp = df.groupby("churned")[BASE].mean()
    gap = ((grp.loc[1] - grp.loc[0]) / grp.loc[0]).sort_values(key=abs, ascending=False)
    print("    signals where the churn group differs most from the stay group (mean gap ratio):")
    for name, v in gap.head(3).items():
        print(f"      {FEAT_KO[name]:<20} {v:+.0%}")

    # [3] Feature preparation ---------------------------------------------------
    print("\n[3] Features: 6 raw + 2 derived")
    df["usage_per_tenure"] = df["usage_days_30d"] / (df["tenure_months"] + 1)
    df["calls_plus_changes"] = df["support_calls_30d"] + df["plan_changes"]
    features = BASE + DERIVED
    X, y = df[features], df["churned"]

    # [4] Cross-validation + final model -------------------------------------------
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=42, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"\n[4] 5-fold cross-validation AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    idx_te = X_te.index
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rec, prec = recall_score(y_te, pred), precision_score(y_te, pred)
    print(f"    test performance: AUC {roc_auc_score(y_te, proba):.3f} / "
          f"recall {rec:.1%} / precision {prec:.1%}")
    goal = "MET" if (rec >= 0.55 and prec >= 0.25) else "NOT MET -> revisit threshold/features"
    print(f"    target-metric verdict (recall 55%+, precision 25%+): {goal}")
    print(f"    (precision is {prec/y_te.mean():.1f}x the random-send baseline hit rate of {y_te.mean():.1%})")

    # [5] Interpretation: standardized coefficients ------------------------------------
    print("\n[5] Model interpretation — what drives the risk up (standardized coefficients)")
    scaler = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    coefs = lr.coef_[0]
    for name, c in sorted(zip(features, coefs), key=lambda t: -abs(t[1])):
        arrow = "risk up" if c > 0 else "risk down"
        print(f"    {FEAT_KO[name]:<20} {c:+.2f} ({arrow}) {'#' * int(abs(c) * 6)}")

    # [6] Deliverable: risk top-10 + driver + action -----------------------------------
    print("\n[6] Final deliverable — the top-10 churn-risk list (test customers)")
    # An individual customer's 'main driver' = the feature with the largest risk-direction
    # contribution among standardized feature value x coefficient
    Z = scaler.transform(X_te)                      # standardized feature values
    contrib = Z * coefs                             # per-customer, per-feature risk contribution
    top10 = np.argsort(proba)[::-1][:10]
    print(f"    {'cust ID':<9} {'churn %':>7}  {'main driver':<20} recommended action")
    print("    " + "-" * 78)
    for i in top10:
        cust_id = df.loc[idx_te[i], "customer_id"]
        main_feat = features[int(np.argmax(contrib[i]))]
        action = ACTION_MAP[main_feat]
        print(f"    {cust_id:<9} {proba[i]:>6.1%}  {FEAT_KO[main_feat]:<20} {action}")
    print("\n    Campaign operating notes:")
    print("      - probability 80%+: phone consultation / 50-80%: coupon + message (intensity by probability band)")
    print("      - impact verification: hold out a random control group of at-risk customers and compare churn rates")
    print("\n    Lesson: what completes the project is not AUC but 'a table the marketing")
    print("            team can execute as-is on Monday morning'.")


if __name__ == "__main__":
    main()
