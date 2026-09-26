"""
Scaling and pipeline experiments.
While predicting churn with KNN on churn_table's mixed-unit features,
two comparisons — (1) without vs with scaling, (2) preprocessing outside
cross-validation vs inside a pipeline — show why the rule is
'all preprocessing goes inside the pipeline'.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def report(name: str, y_true, proba) -> None:
    """Print AUC and 'actual churners among the 100 highest-risk customers' (a campaign-view metric)."""
    auc = roc_auc_score(y_true, proba)
    top100 = np.asarray(y_true)[np.argsort(proba)[::-1][:100]].sum()
    print(f"    {name:<26} AUC {auc:.3f} / actual churners in top-100 risk list: {int(top100)}")


def main() -> None:
    print("=" * 62)
    print(" Scaling and pipelines: unify the units + shut down leakage at the source")
    print("=" * 62)

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)

    # [1] Check the feature units -----------------------------------------
    print("\n[1] Feature scales (are we measuring with the same ruler?)")
    for c in FEATURES:
        print(f"    {c:<18} range {X[c].min():>7,.0f} ~ {X[c].max():>7,.0f}")
    print("    => monthly_fee is hundreds to thousands of times larger than the rest: it monopolizes the distance.")

    # [2] KNN without scaling ------------------------------------------------
    print("\n[2] KNN without scaling (k=15)")
    knn_raw = KNeighborsClassifier(n_neighbors=15)
    knn_raw.fit(X_tr, y_tr)
    report("KNN (no scaling)", y_te, knn_raw.predict_proba(X_te)[:, 1])
    base = int(y_te.sum() / len(y_te) * 100)
    print(f"    (for reference: picking 100 people at random averages {base} churners)")
    print("    (a few hundred KRW of fee difference currently outweighs a 30-day difference in usage)")

    # [3] Pipeline: StandardScaler + KNN --------------------------------------
    print("\n[3] Pipeline = the wash (scaling) -> assembly (model) conveyor belt")
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    pipe.fit(X_tr, y_tr)          # the scaler is automatically fit on the training data only
    report("KNN + StandardScaler", y_te, pipe.predict_proba(X_te)[:, 1])
    print("    => Same model, same data — only the units were unified, and the performance changes.")

    # [4] Wrong order vs right order (inside cross-validation) -----------------
    print("\n[4] Where the preprocessing happens: 5-fold cross-validation, metric = AUC")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Wrong way: scale the FULL dataset first -> out-of-fold information leaks in
    X_scaled_all = pd.DataFrame(StandardScaler().fit_transform(X), columns=FEATURES)
    bad = cross_val_score(KNeighborsClassifier(n_neighbors=15),
                          X_scaled_all, y, cv=cv, scoring="roc_auc")

    # Right way: feed the whole pipeline -> re-fit per fold on that fold's training part only
    good = cross_val_score(make_pipeline(StandardScaler(),
                                         KNeighborsClassifier(n_neighbors=15)),
                           X, y, cv=cv, scoring="roc_auc")
    print(f"    wrong order (scale everything, then CV): {bad.mean():.4f} ± {bad.std():.4f}")
    print(f"    right order (pipeline inside the CV)   : {good.mean():.4f} ± {good.std():.4f}")
    print("    => For plain scaling the gap looks small, but for preprocessing that uses the")
    print("       answer/distribution heavily — target encoding, imputation, feature selection —")
    print("       scores can inflate badly. One rule: 'all preprocessing goes inside the pipeline'.")

    # [5] Deployment view: predict with one object ------------------------------
    print("\n[5] Deployment simulation: one pipeline object = preprocessing + model")
    new_customers = pd.DataFrame([
        {"tenure_months": 2, "monthly_fee": 29900, "usage_days_30d": 3,
         "support_calls_30d": 4, "plan_changes": 2, "auto_pay": 0},
        {"tenure_months": 36, "monthly_fee": 9900, "usage_days_30d": 28,
         "support_calls_30d": 0, "plan_changes": 0, "auto_pay": 1},
        {"tenure_months": 12, "monthly_fee": 14900, "usage_days_30d": 15,
         "support_calls_30d": 1, "plan_changes": 1, "auto_pay": 1},
    ])
    probs = pipe.predict_proba(new_customers)[:, 1]
    for i, p in enumerate(probs):
        print(f"    new customer {i+1}: churn probability {p:.1%}")
    print("    => Deployment code is one line: pipe.predict_proba(new data). A forgotten-preprocessing accident is impossible.")
    print("\n    Lesson: a pipeline is not a convenience feature — it is a safety device against leakage and skew accidents.")


if __name__ == "__main__":
    main()
