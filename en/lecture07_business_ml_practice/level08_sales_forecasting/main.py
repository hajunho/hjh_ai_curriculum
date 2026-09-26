"""
Case study 1 — the full sales-demand-forecasting pipeline.
Cleans and aggregates one year of cafe-chain sales, builds weekday/season/lag/
moving-average features, evaluates the last 4 weeks as a time-based backtest,
and looks at 'next week' in detail.
Compares against the baselines (same as yesterday / same weekday last week)
and saves a forecast-vs-actual PNG.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BACKTEST_DAYS = 28     # evaluation window: the last 4 weeks (7 days alone mixes in too much luck)
FOCUS_DAYS = 7         # the 'next week' examined in the table and the figure
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 66)
    print(" Case study 1: the next-week sales-forecasting pipeline (sales_table)")
    print("=" * 66)

    # [1] Clean + aggregate to daily -----------------------------------------
    raw = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n0 = len(raw)
    raw = raw.dropna(subset=["revenue"])
    raw = raw[raw["revenue"] > 0]
    daily = raw.groupby("day_index")["revenue"].sum().reset_index()
    print(f"\n[1] Cleaned {n0} -> {len(raw)} rows, aggregated to {len(daily)} days")
    print(f"    average daily revenue {daily['revenue'].mean():,.0f} KRW")

    # [2] Build time features (only values knowable at prediction time!) --------
    print("\n[2] Feature building: calendar + season + last week's info (forecasting 7 days ahead, so only lag>=7)")
    daily["weekday"] = daily["day_index"] % 7            # 0=Mon ... 6=Sun
    daily["is_weekend"] = (daily["weekday"] >= 5).astype(int)
    daily["season_sin"] = np.sin(2 * np.pi * daily["day_index"] / 365)
    daily["season_cos"] = np.cos(2 * np.pi * daily["day_index"] / 365)
    daily["lag_7"] = daily["revenue"].shift(7)           # same weekday last week
    daily["ma7_prev"] = daily["revenue"].shift(7).rolling(7).mean()   # last-7-day average, as of last week
    daily["ma28_prev"] = daily["revenue"].shift(7).rolling(28).mean() # last-4-week average, as of last week
    for wd in range(7):                                  # weekday one-hots
        daily[f"wd_{WEEKDAYS[wd]}"] = (daily["weekday"] == wd).astype(int)
    daily = daily.dropna().reset_index(drop=True)
    features = (["is_weekend", "season_sin", "season_cos", "lag_7", "ma7_prev", "ma28_prev"]
                + [f"wd_{w}" for w in WEEKDAYS])
    print(f"    {len(features)} features: 7 weekday one-hots + 3 weekend/season + 3 last-week lag/moving averages")

    # [3] Time-based split -------------------------------------------------------
    train = daily.iloc[:-BACKTEST_DAYS]
    test = daily.iloc[-BACKTEST_DAYS:]
    print(f"\n[3] Time-based split: train {len(train)} days (day~{train['day_index'].max()}) / "
          f"evaluate the last {BACKTEST_DAYS} days (random splits forbidden!)")
    print("    (assuming weekly forecast refreshes: each day's features use only information up to 7 days back)")

    # [4] Baselines vs the model ---------------------------------------------------
    print(f"\n[4] Baselines vs the model — backtest over the last {BACKTEST_DAYS} days (MAE / MAPE)")
    base_naive = daily["revenue"].shift(1).iloc[-BACKTEST_DAYS:].to_numpy()  # baseline 1: same as yesterday
    base_seasonal = test["lag_7"].to_numpy()             # baseline 2: same weekday last week

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train[features], train["revenue"])
    pred = model.predict(test[features])

    for name, p in [("baseline1 same as yesterday", base_naive),
                    ("baseline2 same weekday last week", base_seasonal),
                    ("RandomForest model", pred)]:
        print(f"    {name:<32} MAE {mean_absolute_error(test['revenue'], p):>11,.0f} KRW | "
              f"MAPE {mape(test['revenue'], p):5.2f}%")
    improve = (1 - mean_absolute_error(test["revenue"], pred)
               / mean_absolute_error(test["revenue"], base_seasonal)) * 100
    print(f"    => vs the strongest baseline the model changes MAE by {improve:+.1f}% "
          f"({'better' if improve > 0 else 'worse'}) — always report relative to a baseline.")

    # [5] Next week's daily forecast table + PNG ------------------------------------
    print(f"\n[5] 'Next week' (the last {FOCUS_DAYS} days): forecast vs actual")
    print(f"    {'day':>4} {'wday':>4} {'actual':>12} {'forecast':>12} {'error':>10}")
    focus = test.iloc[-FOCUS_DAYS:]
    focus_pred = pred[-FOCUS_DAYS:]
    for (_, row), p in zip(focus.iterrows(), focus_pred):
        wd = WEEKDAYS[int(row["weekday"])]
        err = p - row["revenue"]
        print(f"    {int(row['day_index']):>4} {wd:>4} {row['revenue']:>12,.0f} "
              f"{p:>12,.0f} {err:>+10,.0f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    recent = daily.iloc[-56:]                            # the recent 8-week trend
    ax.plot(recent["day_index"], recent["revenue"], label="actual", color="#4477aa")
    ax.plot(test["day_index"], pred, "o--", label="model forecast", color="#cc6677",
            markersize=4)
    ax.plot(focus["day_index"], base_seasonal[-FOCUS_DAYS:], "s:",
            label="seasonal naive (last week)", color="#999933", markersize=4)
    ax.axvline(train["day_index"].max() + 0.5, color="gray", ls="--", lw=1)
    ax.text(train["day_index"].max() + 0.7, ax.get_ylim()[1] * 0.97, "backtest start",
            fontsize=8, va="top")
    ax.set_xlabel("day_index")
    ax.set_ylabel("daily revenue (KRW)")
    ax.set_title("Next-week sales forecast vs actual")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "forecast_vs_actual.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"\n    figure saved: {png}")

    # [6] What did the model look at? -------------------------------------------------
    print("\n[6] Feature importances (top 6)")
    imp = sorted(zip(features, model.feature_importances_), key=lambda t: -t[1])[:6]
    for name, v in imp:
        print(f"    {name:<12} {v:.3f} {'#' * int(v * 40)}")
    print("\n    Lesson: demand forecasting = translating the manager's intuition (weekday, season, trend)")
    print("            into features and applying it automatically and consistently, every day.")


if __name__ == "__main__":
    main()
