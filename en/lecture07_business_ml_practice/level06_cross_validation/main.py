"""
Three cross-validation experiments.
(1) Show with 30 runs how much a single-split score wobbles with split luck,
(2) build a 'mean ± standard deviation' report with 5-fold cross-validation, and
(3) show that random splitting inflates scores on time-series data, compared with TimeSeriesSplit.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, KFold, TimeSeriesSplit)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def main() -> None:
    print("=" * 62)
    print(" Cross-validation: measuring — and taming — the luck in a performance number")
    print("=" * 62)

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

    # [1] 30 single splits: the score is a random variable ------------------
    print("\n[1] Same model, same data — measure AUC 30 times changing only the split seed")
    scores = []
    for seed in range(30):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                  random_state=seed, stratify=y)
        pipe.fit(X_tr, y_tr)
        scores.append(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))
    scores = np.array(scores)
    print(f"    min {scores.min():.4f} / max {scores.max():.4f} / "
          f"spread {scores.max()-scores.min():.4f} / std {scores.std():.4f}")
    # mini histogram
    bins = np.linspace(scores.min(), scores.max() + 1e-9, 7)
    counts, _ = np.histogram(scores, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], counts):
        print(f"    {lo:.3f}~{hi:.3f} | {'#' * c}")
    print("    => This much luck hides behind a 'single score of 0.87'.")

    # [2] 5-fold cross-validation ----------------------------------------------
    print("\n[2] 5-fold StratifiedKFold cross-validation (feeding in the whole pipeline)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print("    per-fold AUC:", " ".join(f"{s:.4f}" for s in cv_scores))
    print(f"    reporting format => AUC {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # [3] A fair comparison of two models ----------------------------------------
    print("\n[3] Model comparison: logistic regression vs random forest (same CV)")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"    logistic regression : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    random forest       : {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    diff = abs(cv_scores.mean() - rf_scores.mean())
    noise = max(cv_scores.std(), rf_scores.std())
    verdict = "hard to call a meaningful difference (mean gap < spread)" if diff < noise \
        else "the gap exceeds the spread, so it looks meaningful"
    print(f"    mean gap {diff:.4f} vs spread {noise:.4f} => {verdict}")

    # [4] The time-series trap: random KFold vs TimeSeriesSplit -------------------
    print("\n[4] Time-series data: predicting the past from the future inflates the score")
    sales = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    sales = sales.dropna(subset=["revenue"])
    sales = sales[sales["revenue"] > 0]
    daily = sales.groupby("day_index")["revenue"].sum().reset_index()
    # Lag features: previous day + moving average -> under random splits, validation info seeps into training features
    daily["lag1"] = daily["revenue"].shift(1)
    daily["ma7"] = daily["revenue"].shift(1).rolling(7).mean()
    daily = daily.dropna().reset_index(drop=True)
    Xs, ys = daily[["lag1", "ma7"]], daily["revenue"]

    model = Ridge(alpha=1.0)
    shuffled = cross_val_score(model, Xs, ys, scoring="r2",
                               cv=KFold(n_splits=5, shuffle=True, random_state=42))
    tssplit = TimeSeriesSplit(n_splits=5)
    ordered = cross_val_score(model, Xs, ys, scoring="r2", cv=tssplit)
    print(f"    random KFold (shuffle)        R2: {shuffled.mean():.4f} ± {shuffled.std():.4f}")
    print(f"    TimeSeriesSplit (order kept)  R2: {ordered.mean():.4f} ± {ordered.std():.4f}")
    print("    (R2 < 0 means 'worse than guessing the mean'. Yesterday's revenue alone makes")
    print("     a poor forecaster — that's the honest report card, and random splitting was hiding it.)")
    print("    TimeSeriesSplit's fold structure (training always earlier than validation):")
    for i, (tr, te) in enumerate(tssplit.split(Xs)):
        print(f"      fold{i+1}: train day {tr.min()}~{tr.max()} -> validate day {te.min()}~{te.max()}")
    print("\n    Lesson: measured once, performance is 'luck'; measured many times, it becomes 'skill ± error'.")
    print("            And when there's a time axis, always take the exam questions from the future.")


if __name__ == "__main__":
    main()
