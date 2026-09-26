"""
Hands-on time series work.
We put the cafe chain's revenue onto the time axis and run, in order:
  - diagnosing bad date strings (to_datetime errors="coerce")
  - resampling daily -> weekly/monthly (resample)
  - the 7-day moving average (rolling) and week-over-week growth (shift / pct_change)
then save the trend chart into outputs/.
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # save figures to files only, no display needed.
import matplotlib.pyplot as plt
import pandas as pd

# Path setup so we can import the shared data module (hjh_data)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def load_clean_sales() -> pd.DataFrame:
    """Generate the sales data and remove the missing/negative contamination (level06 recap)."""
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    before = len(df)
    df = df.dropna(subset=["revenue"])          # drop missing revenue
    df = df[df["revenue"] > 0].copy()           # drop negative outliers
    print(f"    Cleaning: {before:,} rows -> {len(df):,} rows "
          f"({before - len(df)} missing/negative removed)")
    return df


def step1_diagnose_dates(df: pd.DataFrame) -> pd.DataFrame:
    """[1] Diagnose contaminated date strings and rebuild a trustworthy date."""
    print("\n[1] Diagnosing the dates — a string is not a date yet")
    parsed = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    n_bad = int(parsed.isna().sum())
    bad_examples = df.loc[parsed.isna(), "date"].unique()[:5]
    print(f"    Result of errors='coerce': {n_bad:,} NaT (missing dates) produced")
    print(f"    Examples of dates that don't exist on the calendar: {list(bad_examples)}")
    print("    -> This data's date column was stamped assuming 'every month has 31 days', "
          "so it cannot be trusted.")

    # Rebuild real dates from the trustworthy day_index (days since opening).
    df = df.copy()
    df["real_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["day_index"], unit="D")
    print(f"    Period rebuilt from day_index: {df['real_date'].min().date()} ~ "
          f"{df['real_date'].max().date()}")
    return df


def step2_daily_series(df: pd.DataFrame) -> pd.Series:
    """[2] Build the company-wide daily revenue series and check the DatetimeIndex."""
    print("\n[2] Building the company-wide daily revenue series")
    daily = df.groupby("real_date")["revenue"].sum().sort_index()
    print(f"    Days: {len(daily)} / index type: {type(daily.index).__name__}")
    print(f"    First 3 days:\n{(daily.head(3) / 1e6).round(1).to_string()}  (unit: M KRW)")
    # The dt accessor: pull parts like 'month' out of a date column.
    month_of_first_rows = df["real_date"].dt.month.head(3).tolist()
    print(f"    dt accessor example — month of the first 3 rows: {month_of_first_rows}")
    return daily


def step3_resample(daily: pd.Series) -> pd.Series:
    """[3] resample: change the time unit from daily -> weekly -> monthly."""
    print("\n[3] resample — pour the daily data into weekly and monthly boxes and sum")
    weekly = daily.resample("W").sum()
    monthly = daily.resample("ME").sum()
    print(f"    Weekly totals: {len(weekly)} weeks "
          f"(the first week is partial, so it may look small)")
    print(f"    Monthly totals (unit: 100M KRW):")
    for ts, val in monthly.items():
        print(f"      {ts.strftime('%Y-%m')}: {val / 1e8:6.2f}")
    return weekly


def step4_rolling(daily: pd.Series) -> pd.Series:
    """[4] rolling(7): erase the weekday chop with a 7-day moving average."""
    print("\n[4] rolling(7) — removing the noise with a moving average")
    ma7 = daily.rolling(7).mean()
    print(f"    The first 6 days cannot fill the window, so they are NaN: "
          f"leading NaN count = {int(ma7.isna().sum())}")
    sample_day = daily.index[9]
    print(f"    Example) {sample_day.date()} raw {daily.iloc[9]/1e6:.1f}M KRW "
          f"vs 7-day mean {ma7.iloc[9]/1e6:.1f}M KRW")
    weekend_std = daily.std()
    smooth_std = ma7.dropna().std()
    print(f"    Std comparison: raw {weekend_std/1e6:.1f} -> moving average "
          f"{smooth_std/1e6:.1f} (M KRW) — the chop has calmed down")
    return ma7


def step5_growth(weekly: pd.Series) -> None:
    """[5] shift / pct_change: compute week-over-week growth."""
    print("\n[5] Week-over-week growth — shift and pct_change")
    # The partial weeks at both ends (fewer than 7 days) distort growth, so trim them.
    full_weeks = weekly.iloc[1:-1]
    growth = full_weeks.pct_change() * 100
    # Reproduce the same computation by hand with shift, as a check.
    manual = (full_weeks - full_weeks.shift(1)) / full_weeks.shift(1) * 100
    assert ((growth - manual).abs().dropna() < 1e-9).all(), \
        "pct_change and the shift computation disagree"
    print("    Check: pct_change == (this week - last week) / last week  "
          "(reproduced with shift, identical)")
    top = growth.nlargest(3)
    bottom = growth.nsmallest(3)
    print("    Top 3 weeks by growth:")
    for ts, val in top.items():
        print(f"      week ending {ts.date()}: {val:+.1f}%")
    print("    Bottom 3 weeks by growth:")
    for ts, val in bottom.items():
        print(f"      week ending {ts.date()}: {val:+.1f}%")


def step6_plot(daily: pd.Series, ma7: pd.Series) -> None:
    """[6] Save the daily revenue + 7-day moving average chart as a PNG."""
    print("\n[6] Saving the chart")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values / 1e6, color="#9ecae1", linewidth=0.8,
            label="daily revenue")
    ax.plot(ma7.index, ma7.values / 1e6, color="#08519c", linewidth=2.0,
            label="7-day moving average")
    ax.set_title("Cafe chain daily revenue (2025)")
    ax.set_ylabel("revenue (million KRW)")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "daily_trend.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {out_path}")


def main() -> None:
    print("=" * 60)
    print("Level 09 — Working with Time Series Data")
    print("=" * 60)
    df = load_clean_sales()
    df = step1_diagnose_dates(df)
    daily = step2_daily_series(df)
    weekly = step3_resample(daily)
    ma7 = step4_rolling(daily)
    step5_growth(weekly)
    step6_plot(daily, ma7)
    print("\nDone! Open the chart and check whether the beverage high season "
          "(the springtime climb) shows up in the moving average.")


if __name__ == "__main__":
    main()
