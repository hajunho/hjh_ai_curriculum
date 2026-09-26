"""
A calculator that converts a confusion matrix into KRW amounts.
It attaches business unit prices to the TP/FP/FN/TN of a churn model
(logistic regression) and computes 'how much is this model worth?'
and 'how much money is 1 pp of recall?'.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# ---- Business unit prices (assume these come from Finance/Marketing) ----
VALUE_V = 179_000   # customer retention value: revenue kept if saved (KRW/person)
COST_C = 12_000     # intervention cost: coupon + outreach (KRW/person)
SUCCESS_S = 0.30    # intervention success rate: probability a couponed would-be churner stays

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def profit_of(tp: int, fp: int) -> float:
    """Confusion matrix -> expected profit (KRW). TPs save some customers; FPs only spend money."""
    return tp * (VALUE_V * SUCCESS_S - COST_C) - fp * COST_C


def main() -> None:
    print("=" * 62)
    print(" Confusion matrix -> KRW translator: how much is the churn model worth?")
    print("=" * 62)

    # [1] Prepare data and train the model ---------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    # class_weight="balanced": weight the minority class (churn) so it isn't missed (details in level07)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(random_state=42, class_weight="balanced"))
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"\n[1] Logistic regression trained (train {len(X_tr)} / test {len(X_te)}, "
          f"test churn rate {y_te.mean():.1%})")

    # [2] Confusion matrix -------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    rec = recall_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    print("\n[2] Test confusion matrix (positive = churn)")
    print(f"    TP (churn caught) ={tp:4d}  FN (churn missed)={fn:4d}")
    print(f"    FP (false alarm)  ={fp:4d}  TN (correct pass)={tn:4d}")
    print(f"    recall {rec:.1%} / precision {prec:.1%}")

    # [3] Attach a unit price to each cell ---------------------------
    print("\n[3] Converting with business unit prices")
    print(f"    prices: customer value V={VALUE_V:,} KRW, intervention cost C={COST_C:,} KRW, success rate s={SUCCESS_S:.0%}")
    unit_tp = VALUE_V * SUCCESS_S - COST_C
    print(f"    value of 1 TP = V*s - C = {unit_tp:+,.0f} KRW")
    print(f"    value of 1 FP = -C      = {-COST_C:+,} KRW")
    print(f"    1 FN = 0 KRW spent, but opportunity loss V*s = {VALUE_V*SUCCESS_S:,.0f} KRW")
    model_profit = profit_of(tp, fp)
    print(f"    => model expected profit (on the {len(X_te)}-person test set): {model_profit:+,.0f} KRW")

    # [4] Comparison: do nothing vs coupon everyone ------------------
    print("\n[4] Strategy comparison (same test customers)")
    n_pos = int(y_te.sum())
    do_nothing = profit_of(0, 0)
    give_all = profit_of(n_pos, len(y_te) - n_pos)  # intervene on everyone: all churners TP, everyone else FP
    print(f"    A. Do nothing               : {do_nothing:+13,.0f} KRW")
    print(f"    B. Coupon everyone          : {give_all:+13,.0f} KRW")
    print(f"    C. Coupon only model's picks: {model_profit:+13,.0f} KRW")
    best = max([("A", do_nothing), ("B", give_all), ("C", model_profit)], key=lambda t: t[1])
    print(f"    => Best strategy: {best[0]} — the model's value is its ability to choose WHO to spend on.")

    # [5] The value of 1 pp of recall ---------------------------------
    print("\n[5] How much money is '1 pp of recall'?")
    n_customers = 100_000            # scaled to a real service
    churn_rate = float(y.mean())     # churn rate estimated from the data
    value_1pp = n_customers * churn_rate * 0.01 * unit_tp
    print(f"    formula: N * churn rate * 0.01 * (V*s - C)")
    print(f"        = {n_customers:,} customers * {churn_rate:.1%} * 1 pp * {unit_tp:,.0f} KRW")
    print(f"        = about {value_1pp:,.0f} KRW (about {value_1pp*12:,.0f} KRW/year for a monthly campaign)")
    print("\n    Example reporting sentence:")
    print(f'    "Raising recall from {rec:.0%} to {rec+0.05:.0%} (5 pp) adds')
    print(f'     about {value_1pp*5*12/1e6:,.0f} million KRW of churn-defense profit per year."')
    print("\n    Lesson: the moment you attach unit prices, ML metrics become budget language.")


if __name__ == "__main__":
    main()
