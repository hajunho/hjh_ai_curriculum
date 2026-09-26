"""
From model to decision — the 'who gets the coupon' profit-maximization experiment.
Lays the expected-value calculation (p*V*s - C) on top of churn probabilities,
cross-checks the theoretical break-even threshold p* = C/(V*s) against the
empirical optimum, and produces a profit-curve PNG and a campaign-ROI scenario report.
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ---- Business unit prices (same as level01) ----
VALUE_V = 179_000   # customer retention value (KRW)
COST_C = 12_000     # intervention cost: coupon + outreach (KRW)
SUCCESS_S = 0.30    # intervention success rate

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def realized_profit(proba, y_true, threshold, s=SUCCESS_S):
    """Realized profit on the test data when intervening on customers at or above the threshold.
    A customer who really was going to churn saves V at success rate s; otherwise only the coupon cost is spent."""
    target = proba >= threshold
    y = np.asarray(y_true)
    tp = int((target & (y == 1)).sum())     # intervened on a would-be churner
    fp = int((target & (y == 0)).sum())     # intervened on a stayer (wasted cost)
    return tp * (VALUE_V * s - COST_C) - fp * COST_C, int(target.sum())


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 68)
    print(" From model to decision: finding the profit-maximizing coupon threshold")
    print("=" * 68)

    # [1] The churn model and its probabilities --------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"\n[1] Churn model trained. Churn probabilities computed for {len(y_te)} test customers")
    print(f"    (note: trained WITHOUT class_weight — expected-value math needs calibrated probabilities)")

    # [2] Theoretical break-even probability p* -----------------------------------
    p_star = COST_C / (VALUE_V * SUCCESS_S)
    print("\n[2] Theoretical break-even probability (one line of arithmetic)")
    print(f"    expected profit of intervening = p*V*s - C > 0  <=>  p > C/(V*s)")
    print(f"    p* = {COST_C:,} / ({VALUE_V:,} x {SUCCESS_S:.0%}) = {p_star:.3f}")
    print(f"    => Couponing only customers above {p_star:.1%} churn probability is the theoretical optimum.")
    print("       The conventional 0.5 threshold has no justification behind it.")

    # [3] Experiment: sweep the threshold -------------------------------------------
    print("\n[3] Experiment: sweep thresholds 0 -> 1 and compute realized test profit")
    grid = np.arange(0.0, 1.001, 0.01)
    profits = np.array([realized_profit(proba, y_te, t)[0] for t in grid])
    best_i = int(np.argmax(profits))
    best_th = float(grid[best_i])
    print(f"    empirical optimal threshold = {best_th:.2f} (realized profit {profits[best_i]:+,.0f} KRW)")
    print(f"    distance from theory {p_star:.3f} = {abs(best_th - p_star):.3f}")
    print("    => Close together means the probability calibration is usable. Far apart: check calibration first!")

    # [4] Strategy comparison table --------------------------------------------------
    print("\n[4] Strategy comparison (realized profit over the test customers)")
    strategies = [
        ("A. Do nothing", 1.01),
        ("B. Coupon everyone", 0.0),
        ("C. Convention 0.5", 0.5),
        ("D. Theoretical p*", p_star),
        ("E. Empirical best", best_th),
    ]
    for name, th in strategies:
        profit, n_target = realized_profit(proba, y_te, th)
        print(f"    {name:<20} intervene {n_target:>4} | profit {profit:>+12,.0f} KRW")
    c_profit, _ = realized_profit(proba, y_te, 0.5)
    d_profit, _ = realized_profit(proba, y_te, p_star)
    if c_profit > 0:
        print(f"    => Same model — moving the threshold from 0.5 to p* multiplies profit {d_profit/c_profit:.1f}x.")
    print("       This is why 'decision design' comes before 'model improvement'.")

    # [5] Profit-curve PNG -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, profits / 1e4, color="#4477aa", lw=2)
    ax.axvline(p_star, color="#228833", ls="--", lw=1.5,
               label=f"theory p* = {p_star:.3f}")
    ax.axvline(best_th, color="#cc6677", ls=":", lw=1.5,
               label=f"empirical best = {best_th:.2f}")
    ax.axvline(0.5, color="gray", ls="-.", lw=1, label="convention 0.5")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("coupon threshold (churn probability)")
    ax.set_ylabel("realized profit (10k KRW)")
    ax.set_title("Profit vs threshold: who should get the coupon?")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "profit_curve.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"\n[5] Profit curve saved: {png}")

    # [6] Campaign ROI scenarios (at 100,000-customer scale) ---------------------------
    print("\n[6] Campaign ROI report — 100,000-customer scale, three success-rate scenarios")
    n_scale = 100_000 / len(y_te)
    target = proba >= best_th
    n_target = int(target.sum() * n_scale)
    cost = n_target * COST_C
    print(f"    intervention targets: about {n_target:,} customers / coupon budget: {cost:,.0f} KRW")
    print(f"    {'scenario':<12} {'s':>5} | {'net profit':>16} | {'ROI':>7}")
    for label, s_val in [("conservative", 0.20), ("base", 0.30), ("optimistic", 0.40)]:
        profit, _ = realized_profit(proba, y_te, best_th, s=s_val)
        profit_scaled = profit * n_scale
        roi = profit_scaled / cost * 100
        print(f"    {label:<12} {s_val:>5.0%} | {profit_scaled:>+14,.0f} KRW | {roi:>6.0f}%")
    base_profit, _ = realized_profit(proba, y_te, best_th, s=0.30)
    base_roi = base_profit * n_scale / cost * 100
    print("\n    Example campaign-plan sentences:")
    print(f'    "Offering coupons to the roughly {n_target:,} customers above {best_th:.0%} churn probability')
    print(f'     yields an expected ROI of about {base_roi:.0f}% in the base scenario (30% success rate).')
    print(f'     The success rate will be measured against a campaign control group and recalculated next quarter."')
    print("\n    Lesson: the moment you multiply the model's probabilities by price tags (V, C, s),")
    print("            machine learning stops being statistics and becomes a management tool.")


if __name__ == "__main__":
    main()
