"""
level08 — Random forests and ensembles: a collective-intelligence experiment

We compare a single decision tree vs a random forest on churn_table.
  [2] Implement bagging (bootstrap + voting) in 15 lines to see the 'power of averaging'
  [3] Performance comparison (AUC / recall)
  [4] Stability comparison: change the split 12 times and measure the AUC wobble (std dev)
  [5] Feature importance: one tree vs the forest
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def bagged_predict(X_tr, y_tr, X_te, n_trees: int, seed: int = 0) -> np.ndarray:
    """Mini-bagging by hand: grow deep trees on bootstrap resamples, average their
    probabilities (a vote). sklearn's RandomForest = this plus 'random feature
    selection at every split'."""
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    probas = []
    for i in range(n_trees):
        idx = rng.choice(n, n, replace=True)          # bootstrap: same-size draw with replacement
        tree = DecisionTreeClassifier(random_state=i)  # unlimited depth (low bias, high variance)
        tree.fit(X_tr.iloc[idx], y_tr.iloc[idx])
        probas.append(tree.predict_proba(X_te)[:, 1])
    return np.mean(probas, axis=0)                     # averaging cancels the wobble (variance)


if __name__ == "__main__":
    np.random.seed(0)

    # [1] Prepare the data ------------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] Data: {len(df)} customers, churn rate {y.mean():.1%}\n")

    # [2] Mini-bagging by hand -----------------------------------------------------
    single = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    auc_single = roc_auc_score(y_te, single.predict_proba(X_te)[:, 1])
    print("[2] Bagging by hand — the principle of the ox-weighing contest")
    print(f"    1 deep tree                 test AUC = {auc_single:.3f}")
    for n_trees in [5, 25]:
        auc_bag = roc_auc_score(y_te, bagged_predict(X_tr, y_tr, X_te, n_trees))
        print(f"    vote of {n_trees:>2} such trees        test AUC = {auc_bag:.3f}")
    print("    -> Individuals (trees) overfit, yet the average of many with different")
    print("       life experiences grows strong.\n")

    # [3] Performance comparison: single tree vs random forest ----------------------
    print("[3] Performance comparison (test set)")
    models = {
        "Tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=0),
        "Tree (unlimited)": DecisionTreeClassifier(random_state=0),
        "Random forest (300 trees)": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1),
    }
    print("    Model                        AUC     recall (threshold 0.5)")
    forest = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        rec = recall_score(y_te, m.predict(X_te))
        print(f"    {name:25s} {auc:.3f}      {rec:6.1%}")
        if isinstance(m, RandomForestClassifier):
            forest = m
    print()

    # [4] Stability comparison: measuring the wobble across changing splits ----------
    print("[4] Stability comparison — AUC re-measured over 12 different splits")
    aucs_tree, aucs_rf = [], []
    for rep in range(12):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25,
                                          random_state=rep, stratify=y)
        t = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xa, ya)
        f = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1).fit(Xa, ya)
        aucs_tree.append(roc_auc_score(yb, t.predict_proba(Xb)[:, 1]))
        aucs_rf.append(roc_auc_score(yb, f.predict_proba(Xb)[:, 1]))
    aucs_tree, aucs_rf = np.array(aucs_tree), np.array(aucs_rf)
    print(f"    Single tree     AUC mean {aucs_tree.mean():.3f} ± std {aucs_tree.std():.3f}")
    print(f"    Random forest   AUC mean {aucs_rf.mean():.3f} ± std {aucs_rf.std():.3f}")
    print(f"    Times the forest won: {int((aucs_rf > aucs_tree).sum())}/12")
    print("    -> Not just mean performance: a small 'wobble' (std dev) is the heart")
    print("       of trust in production.\n")

    # [5] Feature importance: one tree vs the forest ----------------------------------
    print("[5] Feature-importance comparison (impurity-reduction share, sums to 1)")
    tree5 = models["Tree (depth 5)"]
    print("    Feature              1 tree     forest of 300")
    for i in np.argsort(-forest.feature_importances_):
        print(f"    {FEATURES[i]:18s}   {tree5.feature_importances_[i]:.3f}      "
              f"{forest.feature_importances_[i]:.3f}")
    print("    -> The forest's importances, being an average over many trees, are more stable.")
    print("       (Still 'how much it was used', not causation — the fair measurement is level11)")
