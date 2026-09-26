"""
level11 — Gradient boosting and model interpretation

We build a boosting model on churn_table and go all the way to answering 'why?'.
  [2] performance comparison: logistic vs random forest vs HistGradientBoosting
  [3] mini-experiment: learning rate (how much of the mistake journal to absorb) x tree count
  [4] permutation importance: how far AUC collapses when one feature is shuffled
  [5] partial dependence: the shape of predicted probability vs a feature's value (text graph)
  [6] per-prediction explanation: decompose the grounds by swapping features to average values
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
FEATURE_KO = {"tenure_months": "tenure (months)", "monthly_fee": "monthly fee",
              "usage_days_30d": "usage days", "support_calls_30d": "support calls",
              "plan_changes": "plan changes", "auto_pay": "auto-pay"}


def explain_one(model, x_row: pd.DataFrame, baseline: pd.Series) -> list[tuple[str, float]]:
    """A 12-line per-prediction explainer: for each feature, measure how much the
    churn probability changes when it's swapped for the 'average customer' value.
    The features with the biggest swings = the grounds for the verdict."""
    p0 = model.predict_proba(x_row)[0, 1]
    contribs = []
    for f in FEATURES:
        x_mod = x_row.copy()
        x_mod[f] = baseline[f]
        p_mod = model.predict_proba(x_mod)[0, 1]
        contribs.append((f, p0 - p_mod))     # positive = this feature raised the risk
    return sorted(contribs, key=lambda t: -abs(t[1]))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] Data ---------------------------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] Data: {len(df)} customers, churn rate {y.mean():.1%}\n")

    # [2] Graduation exam for the three models -----------------------------------------
    print("[2] Performance comparison of the models learned so far (test AUC)")
    models = {
        "Logistic regression (level05)": Pipeline([
            ("s", StandardScaler()), ("c", LogisticRegression(random_state=0))]),
        "Random forest (level08)": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=0, early_stopping=True),
    }
    boost = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"    {name:30s} AUC = {auc:.3f}")
        if isinstance(m, HistGradientBoostingClassifier):
            boost = m
    print("    -> Surprise: logistic takes first place! This synthetic data's true structure")
    print("       was built as a 'linear scorecard' (a linear formula in the log-odds). Two lessons:")
    print("       (1) The strongest model doesn't always win — the model matching the data's structure wins.")
    print("       (2) On real data full of nonlinearity and interactions, boosting often pulls ahead.\n")

    # [3] Mini-experiment: learning rate x tree count -------------------------------------
    print("[3] Learning-rate experiment (how much mistake journal to absorb) — the brake vs periods trade")
    print("    learning rate   max trees   test AUC")
    for lr, n_iter in [(1.0, 50), (0.3, 100), (0.1, 200), (0.03, 500)]:
        m = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=n_iter, random_state=0,
            early_stopping=True).fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"    {lr:<12}    {n_iter:>5}      {auc:.3f}")
    print("    -> Lower the learning rate and more trees are needed, but results tend to stabilize.")
    print("       (learning_rate is level09's regularization philosophy arriving in boosting)\n")

    # [4] Permutation importance -----------------------------------------------------------
    print("[4] Permutation importance — test-AUC drop when one feature is shuffled")
    perm = permutation_importance(boost, X_te, y_te, scoring="roc_auc",
                                  n_repeats=10, random_state=0)
    print("    Feature            AUC drop (mean ± std)")
    for i in np.argsort(-perm.importances_mean):
        bar = "#" * max(0, int(perm.importances_mean[i] * 200))
        print(f"    {FEATURE_KO[FEATURES[i]]:16s}   {perm.importances_mean[i]:+.4f} ± {perm.importances_std[i]:.4f}  {bar}")
    print("    -> 'Interrogating the finished model with test data' works for any model type,")
    print("       and is more honest for reporting than impurity importance (levels 07-08).\n")

    # [5] Partial dependence ------------------------------------------------------------------
    print("[5] Partial dependence — average churn probability as one feature moves, others fixed")
    for feat in ["usage_days_30d", "support_calls_30d"]:
        # method="brute": computed via predict_proba -> results are in 'probability' units
        pd_res = partial_dependence(boost, X_te, [feat], kind="average",
                                    grid_resolution=7, method="brute")
        grid = pd_res["grid_values"][0]
        avg = pd_res["average"][0]
        print(f"    {FEATURE_KO[feat]} ({feat})")
        for g, v in zip(grid, avg):
            bar = "#" * int(v * 60)
            print(f"      value {g:6.1f} -> avg probability {v:5.1%} {bar}")
    print("    -> The curve's 'shape' (where it changes sharply) gives richer information")
    print("       than a single logistic coefficient. Still association, not causation.\n")

    # [6] Per-prediction explanation --------------------------------------------------------------
    proba_te = boost.predict_proba(X_te)[:, 1]
    idx = int(np.argmax(proba_te))                     # highest-risk customer in the test set
    x_row = X_te.iloc[[idx]]
    baseline = X_tr.mean()                             # the 'average customer'
    print("[6] Per-prediction explanation — decomposing the grounds for the highest-risk customer")
    print(f"    This customer's churn probability: {proba_te[idx]:.1%} (relative to the average-customer baseline)")
    print("    Feature (customer value -> swapped to average)      probability change")
    for f, delta in explain_one(boost, x_row, baseline):
        if abs(delta) < 0.005:
            direction = "negligible effect"
        else:
            direction = "ground that raised the risk" if delta > 0 else "factor that lowered the risk"
        print(f"    {FEATURE_KO[f]:16s} ({x_row[f].iloc[0]:>8.1f} -> {baseline[f]:>8.1f})   "
              f"{delta:+6.1%}p  {direction}")
    print("    -> Sentences like 'bring the calls back to the average level and the probability")
    print("       drops sharply' become the basis of a call script. (The professional tool SHAP")
    print("       is a refinement of this same principle.)")
    print()
    print("[7] lecture06 complete! Only when you can report performance (boosting) and explanation")
    print("    (importance, partial dependence, case explanations) as a set does a model get adopted.")
