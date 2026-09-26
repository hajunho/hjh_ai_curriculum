"""
An experiment proving the power of feature engineering.
On fraud_table, with the same model and the same split held fixed,
compare the performance (AUC, PR-AUC) of 'raw features only' vs 'raw + derived',
and finish with a demo of the fake performance a data-leakage feature creates.
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
from sklearn.metrics import roc_auc_score, average_precision_score

# 'Raw' = fields that arrive with the payment-authorization message itself (amount, hour, foreign flag)
RAW = ["amount", "hour", "is_foreign"]
# 'Derived' = features a human built from domain knowledge
# (the transaction-history aggregate tx_count_1h is left for the 'Try it yourself' exercise)
DERIVED = ["log_amount", "is_night", "foreign_night"]


def evaluate(X_tr, X_te, y_tr, y_te, label: str) -> tuple[float, float]:
    """Measure performance with the model and preprocessing fixed, swapping only the feature set (controlled experiment)."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    pr_auc = average_precision_score(y_te, proba)
    print(f"    {label:<24} AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")
    return auc, pr_auc


def main() -> None:
    print("=" * 62)
    print(" Raw features vs derived features — a fair fight on the same model")
    print("=" * 62)

    # [1] Raw data --------------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    print(f"\n[1] fraud_table: {len(df)} transactions, fraud rate {df['is_fraud'].mean():.2%}")
    print(f"    raw features (known the moment the payment happens): {RAW}")

    # [2] Domain knowledge -> derived features ------------------------------
    print("\n[2] Translating domain knowledge into numeric columns (3 derived features)")
    df["log_amount"] = np.log1p(df["amount"])                    # skewed amounts -> a sense of multiples
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)  # small-hours flag
    df["foreign_night"] = df["is_foreign"] * df["is_night"]       # foreign x night interaction
    recipes = {
        "log_amount": "log transform — make 'twice as large' the same-size step everywhere",
        "is_night": "range flag — the business common sense that 'small-hours transactions are suspicious'",
        "foreign_night": "interaction — risky only when foreign AND in the small hours at once",
    }
    for k, v in recipes.items():
        print(f"    - {k:<14}: {v}")

    # [3] Controlled experiment: split and model fixed, features swapped ------
    print("\n[3] Performance comparison (logistic regression, identical split, positive = fraud)")
    y = df["is_fraud"]
    idx_tr, idx_te = train_test_split(df.index, test_size=0.3,
                                      random_state=42, stratify=y)
    _, pr_raw = evaluate(df.loc[idx_tr, RAW], df.loc[idx_te, RAW],
                         y.loc[idx_tr], y.loc[idx_te], "raw 3")
    _, pr_full = evaluate(df.loc[idx_tr, RAW + DERIVED], df.loc[idx_te, RAW + DERIVED],
                          y.loc[idx_tr], y.loc[idx_te], "raw + derived 6")
    print(f"    => PR-AUC {pr_raw:.4f} -> {pr_full:.4f} "
          f"({(pr_full - pr_raw) / pr_raw * 100:+.1f}%) — not one character of the model changed")

    # [4] Which features do the work ------------------------------------------
    print("\n[4] Standardized coefficients of the derived-set model (bigger absolute value = bigger influence)")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
    )
    cols = RAW + DERIVED
    model.fit(df.loc[idx_tr, cols], y.loc[idx_tr])
    coefs = model.named_steps["logisticregression"].coef_[0]
    for name, c in sorted(zip(cols, coefs), key=lambda t: -abs(t[1])):
        bar = "#" * int(abs(c) * 4)
        print(f"    {name:<14} {c:+.2f} {bar}")

    # [5] Leakage demo: feed a shadow of the answer in as a feature -----------
    print("\n[5] Data-leakage demo — a fake feature called 'investigation score'")
    rng = np.random.default_rng(0)
    # A value knowable only AFTER a fraud ruling: answer + a little noise = the classic leaky feature
    df["inspection_score"] = df["is_fraud"] * 0.9 + rng.normal(0, 0.1, len(df))
    _, pr_leak = evaluate(df.loc[idx_tr, cols + ["inspection_score"]],
                          df.loc[idx_te, cols + ["inspection_score"]],
                          y.loc[idx_tr], y.loc[idx_te], "derived + leaky feature")
    print(f"    => PR-AUC {pr_leak:.4f}: unrealistically perfect = not a celebration but a 'bug alarm'.")
    print("       Screening question: \"Can I know this value at prediction time?\" — if no, drop it immediately.")
    print("\n    Lesson: the key to performance is not swapping models but translating domain knowledge into features.")


if __name__ == "__main__":
    main()
