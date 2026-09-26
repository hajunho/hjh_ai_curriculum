"""
level05 — Hands-On Text Classification: Review Sentiment Analysis

Vectorizes review_corpus (600 English reviews) with TF-IDF and classifies
positive/negative with logistic regression.
We then open the learned coefficients (the per-word scorecard) to interpret
the model's reasoning, and print an evidence report for new reviews.
"""

import sys
import pathlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

SEED = 42


def load_data():
    """[1] Load 600 reviews, then split train/test."""
    rows = hjh_data.review_corpus(600, seed=3)
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows])
    print(f"[1] Data prep: {len(texts)} reviews "
          f"({labels.sum()} positive / {(labels == 0).sum()} negative)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    print(f"    train {len(X_tr)} / test {len(X_te)} (ratio kept via stratify)")
    print(f"    example (positive): {X_tr[int(np.argmax(y_tr))]!r}")
    print(f"    example (negative): {X_tr[int(np.argmin(y_tr))]!r}\n")
    return X_tr, X_te, y_tr, y_te


def vectorize(X_tr, X_te):
    """[2] TF-IDF vectorization — vocabulary and IDF built from TRAINING data only."""
    vec = TfidfVectorizer()
    V_tr = vec.fit_transform(X_tr)      # train: fit + transform
    V_te = vec.transform(X_te)          # test: transform only (no data leakage)
    print(f"[2] Vectorized: {len(vec.get_feature_names_out())} vocabulary types, "
          f"train matrix {V_tr.shape}, test matrix {V_te.shape}")
    print("    Note: we never fit on the test data (leakage prevention).\n")
    return vec, V_tr, V_te


def train_and_eval(V_tr, y_tr, V_te, y_te):
    """[3] Train and evaluate logistic regression."""
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(V_tr, y_tr)
    pred = model.predict(V_te)
    acc = accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred)
    print(f"[3] Training done. Test accuracy: {acc:.1%}")
    print("    Confusion matrix (rows=actual, cols=predicted / 0=negative, 1=positive)")
    print(f"                 pred neg  pred pos")
    print(f"    actual neg   {cm[0, 0]:5d} {cm[0, 1]:9d}")
    print(f"    actual pos   {cm[1, 0]:5d} {cm[1, 1]:9d}\n")
    return model


def show_scorecard(model, vec, top: int = 8):
    """[4] Coefficients = the model's 'scorecard'. Reveal the evidence words."""
    words = vec.get_feature_names_out()
    coef = model.coef_[0]
    order = np.argsort(coef)
    print(f"[4] The scorecard the model learned (logistic-regression coefficients)")
    print("    positive evidence top8       | negative evidence top8")
    print("    " + "-" * 58)
    for pos_i, neg_i in zip(order[::-1][:top], order[:top]):
        print(f"    {words[pos_i]:12s} {coef[pos_i]:+6.2f}       | "
              f"{words[neg_i]:12s} {coef[neg_i]:+6.2f}")
    print("    -> No human wrote these rules — the data built the scorecard.\n")


def judge_new_reviews(model, vec):
    """[5] Judge new reviews + report the evidence words."""
    new_reviews = [
        "The staff were friendly and the packaging was careful, very satisfied",
        "Delivery took a whole week and the box arrived torn",
        "Performance was beyond my expectations, I would buy this again",
        "Nothing like the description and my inquiry never got an answer",
    ]
    words = vec.get_feature_names_out()
    coef = model.coef_[0]
    V = vec.transform(new_reviews)
    proba = model.predict_proba(V)[:, 1]
    print("[5] New-review verdict report (positive probability + evidence words)")
    for text, p, row in zip(new_reviews, proba, V.toarray()):
        present = np.where(row > 0)[0]                       # words appearing in this review
        contrib = row[present] * coef[present]               # per-word contribution
        order = np.argsort(np.abs(contrib))[::-1][:3]
        evidence = ", ".join(
            f"{words[present[i]]}({contrib[i]:+.2f})" for i in order)
        verdict = "positive" if p >= 0.5 else "negative"
        print(f"    [{verdict} {p:.0%}] {text}")
        print(f"        evidence: {evidence}")
    print("\nBottom line: we can now report not just 'how often it's right' but 'why it decided that way'.")


if __name__ == "__main__":
    print("=" * 70)
    print("Text classification — a TF-IDF + logistic-regression sentiment analyzer")
    print("=" * 70 + "\n")
    np.random.seed(SEED)
    X_tr, X_te, y_tr, y_te = load_data()
    vec, V_tr, V_te = vectorize(X_tr, X_te)
    model = train_and_eval(V_tr, y_tr, V_te, y_te)
    show_scorecard(model, vec)
    judge_new_reviews(model, vec)
