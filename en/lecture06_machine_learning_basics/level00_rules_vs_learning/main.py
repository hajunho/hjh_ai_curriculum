"""
level00 — Rules vs Learning

We solve the same classification problem (card-fraud detection) two ways and compare.
  A. Hand-made rule: an IF statement a human picked by gut feel
     ("500,000 KRW or more + small hours -> fraud")
  B. Learning: a search that tries every candidate threshold and lets the data
     pick the optimal value (implemented by hand)
Key message: traditional programming is "rules+data->answers";
machine learning is "data+answers->rules".
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

NIGHT_HOURS = {0, 1, 2, 3, 4, 23}          # small hours / late night


def score(y_true: list[int], y_pred: list[int]) -> dict:
    """The rule's report card. Fraud is only 1.4% of rows, so plain accuracy is
    an illusion; we grade on the average of the 'fraud hit rate (recall)' and
    the 'normal hit rate' (the balanced score)."""
    n_fraud = sum(y_true)
    hit_fraud = sum(p == 1 for t, p in zip(y_true, y_pred) if t == 1) / n_fraud
    hit_normal = sum(p == 0 for t, p in zip(y_true, y_pred) if t == 0) / (len(y_true) - n_fraud)
    n_alarm = sum(y_pred)
    precision = (sum(t == 1 for t, p in zip(y_true, y_pred) if p == 1) / n_alarm) if n_alarm else 0.0
    return {"balanced": (hit_fraud + hit_normal) / 2, "recall": hit_fraud,
            "precision": precision, "alarms": n_alarm}


def hand_rule(row: dict) -> int:
    """[Hand-made rule] The gut-feel rule you'd hear in a meeting room:
    'If big money — over 500,000 KRW — leaves in the small hours, it must be fraud.'"""
    return 1 if (row["amount"] >= 500_000 and row["hour"] in NIGHT_HOURS) else 0


def learn_threshold(rows: list[dict], feature: str) -> tuple[float, float]:
    """[Learning] Try every threshold T for the rule 'value >= T means fraud'.
    The human decides only the rule's 'shape'; the data picks the 'number' T."""
    y_true = [r["is_fraud"] for r in rows]
    best_t, best_s = None, -1.0
    for t in sorted({r[feature] for r in rows}):
        y_pred = [1 if r[feature] >= t else 0 for r in rows]
        s = score(y_true, y_pred)["balanced"]
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def learn_combo(rows: list[dict]) -> tuple[float, float]:
    """[Learning, extended] Search T for 'amount >= T AND small hours'.
    Same shape as the human rule, but the data picks the number."""
    y_true = [r["is_fraud"] for r in rows]
    best_t, best_s = None, -1.0
    for t in sorted({round(r["amount"], -4) for r in rows}):    # candidates in 10,000-KRW steps
        y_pred = [1 if (r["amount"] >= t and r["hour"] in NIGHT_HOURS) else 0
                  for r in rows]
        s = score(y_true, y_pred)["balanced"]
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def report(name: str, s: dict) -> None:
    print(f"    {name}")
    print(f"      balanced score {s['balanced']:.1%} / fraud hit rate {s['recall']:.1%} / "
          f"alarm precision {s['precision']:.1%} ({s['alarms']} alarms)")


if __name__ == "__main__":
    # [1] Prepare the data -------------------------------------------------
    rows = hjh_data.fraud_table(n=5000, seed=11)   # fixed seed -> always the same data
    y_true = [r["is_fraud"] for r in rows]
    print("[1] Data: card transactions,", len(rows), "rows (same structure as a spam filter)")
    hjh_data.head(rows, 3)
    print(f"    Fraud rate: {sum(y_true) / len(y_true):.1%}  (fraud=1, normal=0)\n")

    # [2] Approach A — hand-made rule ---------------------------------------
    pred_hand = [hand_rule(r) for r in rows]
    s_hand = score(y_true, pred_hand)
    print("[2] Approach A — hand-made rule (a human picks the number by gut feel)")
    print("    Rule: amount >= 500,000 KRW AND small hours -> fraud")
    report("Score:", s_hand)
    print("      -> Everything it flags is real, but it misses 70% of the fraud. The bar was too high.\n")

    # [3] Approach B — learning: the data picks the optimal threshold -------
    print("[3] Approach B — learning (try every candidate threshold T; the data chooses)")
    best_feat, best_t, best_s = None, None, -1.0
    for feat in ["amount", "hour", "is_foreign"]:
        t, s = learn_threshold(rows, feat)
        print(f"    Feature {feat:12s}: best T={t:>10,} -> balanced score {s:.1%}")
        if s > best_s:
            best_feat, best_t, best_s = feat, t, s
    pred_learn = [1 if r[best_feat] >= best_t else 0 for r in rows]
    s_learn = score(y_true, pred_learn)
    print(f"    => Rule the data chose: \"{best_feat} >= {best_t:,} means fraud\"")
    report("Score:", s_learn)
    print("      -> The data found a number the human never knew, but there are many false alarms (precision down).\n")

    # [4] Approach B extended — combining two features -----------------------
    t2, s2 = learn_combo(rows)
    pred_two = [1 if (r["amount"] >= t2 and r["hour"] in NIGHT_HOURS) else 0 for r in rows]
    s_combo = score(y_true, pred_two)
    print("[4] Approach B extended — same shape as the human rule, data picks the number")
    print(f"    Rule the data chose: \"amount >= {t2:,} KRW AND small hours -> fraud\"")
    report("Score:", s_combo)
    print("      -> The human's gut (500,000 KRW) vs the data's answer (about 20,000 KRW) — that far apart.\n")

    # [5] Overall comparison -------------------------------------------------
    print("[5] Overall comparison (balanced score = average of fraud hit rate and normal hit rate)")
    print("    Approach                      balanced   fraud hit    alarm precision")
    print(f"    A. Hand-made rule              {s_hand['balanced']:6.1%}     {s_hand['recall']:6.1%}      {s_hand['precision']:6.1%}")
    print(f"    B. Learning (1 feature)        {s_learn['balanced']:6.1%}     {s_learn['recall']:6.1%}      {s_learn['precision']:6.1%}")
    print(f"    B. Learning (2 features)       {s_combo['balanced']:6.1%}     {s_combo['recall']:6.1%}      {s_combo['precision']:6.1%}")
    print()
    print("    Key: traditional programming  rules + data -> answers")
    print("         machine learning         data + answers -> rules")
    print("    Today's 'learning' was a for-loop threshold search, but a neural net is the same at heart.")
    print("    (Caution: we graded on the training data, so these scores are optimistic -> level04)")
