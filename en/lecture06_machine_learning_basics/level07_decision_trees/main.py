"""
level07 — Decision trees: finding churners with twenty questions

We train a decision tree on churn_table, and
  - compute Gini impurity by hand to understand what makes a 'good question'
  - print the learned tree as human-readable rule sentences and interpret it
  - vary the depth and observe the fork in the road to overfitting.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def gini(labels: np.ndarray) -> float:
    """Gini impurity: 0 if the room is one kind, 0.5 if fifty-fifty.
    These 8 lines are the entirety of the tree's math."""
    if len(labels) == 0:
        return 0.0
    p = labels.mean()               # churn share
    return 1.0 - (p ** 2 + (1 - p) ** 2)


def split_gain(y: np.ndarray, mask: np.ndarray) -> float:
    """How much did the size-weighted average impurity of the two rooms drop after the question (mask)?"""
    left, right = y[mask], y[~mask]
    after = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
    return gini(y) - after


if __name__ == "__main__":
    np.random.seed(0)

    # [1] Data prep (trees need no standardization: threshold-comparison model) --
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] Data: {len(df)} customers, churn rate {y.mean():.1%} — train {len(X_tr)} / test {len(X_te)}\n")

    # [2] Gini impurity by hand: what makes a good question? -----------------------
    y_arr = y_tr.to_numpy()
    print("[2] Gini impurity — 'how mixed is the room'")
    print(f"    Impurity of the whole room, before any question: {gini(y_arr):.4f}")
    for feat, th in [("usage_days_30d", 9.5), ("monthly_fee", 15000), ("support_calls_30d", 1.5)]:
        gain = split_gain(y_arr, (X_tr[feat] <= th).to_numpy())
        print(f"    Impurity reduction from asking \"{feat} <= {th}\": {gain:.4f}")
    print("    -> The tree tries every feature x every threshold and picks the max-reduction question.")
    print("       (Just level00's threshold search applied recursively — no magic.)\n")

    # [3] Train a depth-3 tree + print the rules as text -----------------------------
    # class_weight="balanced": weights churners (15%) more heavily than stayers
    # so that minority-class (churn) rules surface in the tree.
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30,
                                  class_weight="balanced", random_state=0)
    tree.fit(X_tr, y_tr)
    print("[3] Full rules of the depth-3 tree (export_text)")
    print(export_text(tree, feature_names=FEATURES))

    # Translate the leaf rooms into report sentences
    leaf_id = tree.apply(X_tr)
    print("    Leaf-room summary (on training data):")
    rows = []
    for leaf in np.unique(leaf_id):
        members = y_tr[leaf_id == leaf]
        rows.append((leaf, len(members), members.mean()))
    for leaf, n, rate in sorted(rows, key=lambda r: -r[2])[:3]:
        print(f"      Top-risk room #{leaf}: {n} customers, churn rate {rate:.1%}")
    print("    -> The path to a top room (trace it in the rule tree above) can be")
    print("       re-imported directly as a CRM rule.\n")

    # [4] Train/test scores by depth — the fork in the road to overfitting ------------
    print("[4] Depth and overfitting (revisiting level04's experiment A)")
    print("    Depth      train accuracy   test accuracy   test recall")
    for depth in [1, 3, 5, 10, None]:
        t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        rec = recall_score(y_te, t.predict(X_te))
        label = "unlimited" if depth is None else f"{depth:>4}"
        print(f"    {label:>9}      {t.score(X_tr, y_tr):6.1%}         {t.score(X_te, y_te):6.1%}       {rec:6.1%}")
    print("    -> Unlimited depth = memorizing the training set 100%, and losing on the test.\n")

    # [5] Feature importances ------------------------------------------------------
    print("[5] Feature importances (share of impurity reduction, sums to 1)")
    order = np.argsort(-tree.feature_importances_)
    for i in order:
        bar = "#" * int(tree.feature_importances_[i] * 40)
        print(f"    {FEATURES[i]:18s} {tree.feature_importances_[i]:.3f} {bar}")
    print("    -> 'Most leaned-on features', not a ranking of causes (interpret with care; level11).")
