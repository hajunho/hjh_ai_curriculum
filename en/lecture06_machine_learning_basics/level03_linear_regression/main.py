"""
level03 — Linear regression: ad spend -> revenue

From cafe-chain sales data, we compute 'how much extra revenue per 10,000 KRW
of ad spend' as a least-squares line. We calculate the same answer two ways:
  (1) NumPy formula: a = Cov(x,y)/Var(x), b = mean(y) - a*mean(x)
  (2) sklearn LinearRegression
The scatter plot + regression line is saved to outputs/regression.png.
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")   # allows saving figures on machines with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def load_daily_store_sales() -> pd.DataFrame:
    """[1] The raw data is (day x store x category) -> aggregate to (day x store).
    Missing revenue and negative contamination are removed before aggregating
    (a lecture03 refresher)."""
    raw = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n_before = len(raw)
    clean = raw.dropna(subset=["revenue"])          # drop missing values
    clean = clean[clean["revenue"] > 0]             # drop negatives (entry errors)
    print(f"    Cleaning: {n_before} rows -> {len(clean)} rows (removed {raw['revenue'].isna().sum()} missing, "
          f"{(raw['revenue'].dropna() <= 0).sum()} negative)")
    # ad_cost is one value per (day, store), so take first; revenue sums over categories
    daily = (clean.groupby(["day_index", "store"], as_index=False)
                  .agg(ad_cost=("ad_cost", "first"), revenue=("revenue", "sum")))
    return daily


if __name__ == "__main__":
    np.random.seed(0)  # reproducibility (this level uses no randomness, but we fix it by convention)

    # [1] Prepare the data ---------------------------------------------------
    print("[1] Data prep — cafe chain, 365 days x 5 stores of sales")
    daily = load_daily_store_sales()
    x = daily["ad_cost"].to_numpy(dtype=float)      # input: daily ad spend (KRW)
    y = daily["revenue"].to_numpy(dtype=float)      # answer: daily revenue (KRW)
    print(f"    Unit of analysis: (day, store), {len(daily)} rows / "
          f"ad spend range {x.min():,.0f}~{x.max():,.0f} KRW\n")

    # [2] Least-squares line by NumPy formula ----------------------------------
    slope_np = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)  # a = Cov/Var
    intercept_np = y.mean() - slope_np * x.mean()              # passes through (x̄,ȳ)
    print("[2] Computed directly with the NumPy formula (Cov/Var)")
    print(f"    slope a = {slope_np:.4f}   intercept b = {intercept_np:,.0f}")

    # [3] Solve the same problem with sklearn ----------------------------------
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)   # sklearn wants 2-D input (rows=samples, cols=features)
    slope_sk, intercept_sk = model.coef_[0], model.intercept_
    print("[3] sklearn LinearRegression")
    print(f"    slope a = {slope_sk:.4f}   intercept b = {intercept_sk:,.0f}")
    same = np.isclose(slope_np, slope_sk) and np.isclose(intercept_np, intercept_sk)
    print(f"    Do the two methods agree? {same} — the library is packaging around the same formula.\n")

    # [4] Business interpretation ----------------------------------------------
    y_hat = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_hat)
    print("[4] Business interpretation")
    print(f"    Per 10,000 KRW of ad spend: revenue +{slope_sk * 10_000:,.0f} KRW associated (correlation, not proof of causation!)")
    print(f"    R^2 = {r2:.3f} -> ad spend alone explains {r2:.1%} of the revenue fluctuation")
    ad = 300_000
    pred = model.predict([[ad]])[0]
    print(f"    Predicted revenue on a day with {ad:,} KRW ad spend: {pred:,.0f} KRW")
    print("    (Caution: trained and graded on the full data — an optimistic score -> level04)\n")

    # [5] Save scatter plot + regression line -----------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x / 10_000, y / 10_000, s=8, alpha=0.3, label="daily data")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs / 10_000, (slope_sk * xs + intercept_sk) / 10_000,
            color="crimson", linewidth=2,
            label=f"y = {slope_sk:.2f}x + {intercept_sk/10_000:,.0f}")
    ax.set_xlabel("ad cost (10k KRW)")
    ax.set_ylabel("daily revenue (10k KRW)")
    ax.set_title("Ad cost vs daily revenue (least squares fit)")
    ax.legend()
    fig.tight_layout()
    png = OUT_DIR / "regression.png"
    fig.savefig(png, dpi=120)
    print(f"[5] Figure saved: {png}")
    print("    Amid the scatter (noise), the line summarizes the 'average tendency'.")
