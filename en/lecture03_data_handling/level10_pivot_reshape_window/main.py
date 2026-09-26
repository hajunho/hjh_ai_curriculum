"""
Hands-on pivoting, reshaping, and window operations.
We reproduce the Excel pivot table in pandas.
  - pivot_table (index/columns/values/aggfunc/margins)
  - melt to turn wide back into long
  - rank / pct_change / cumsum window operations and within-group ranks
The final pivot report is saved to outputs/ as a CSV.
"""

import os
import pathlib
import sys

import pandas as pd

# Path setup so we can import the shared data module (hjh_data)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_clean_sales() -> pd.DataFrame:
    """Generate the sales data, remove missing/negative rows, and attach real dates."""
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    df = df.dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    # The date strings are contaminated (see level09), so build dates from day_index.
    df["real_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["day_index"], unit="D")
    print(f"    After cleaning: {len(df):,} rows "
          f"(long format: one row = one day-store-category revenue record)")
    return df


def step1_store_category_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """[1] Store x category annual revenue pivot (margins = grand totals included)."""
    print("\n[1] Store x category revenue pivot — the pandas version of Excel's pivot table")
    pivot = pd.pivot_table(df, index="store", columns="category",
                           values="revenue", aggfunc="sum",
                           margins=True, margins_name="Total")
    print((pivot / 1e8).round(2).to_string())
    print("    (unit: 100M KRW; margins=True adds the 'Total' row/column automatically)")
    return pivot


def step2_weekday_pivot(df: pd.DataFrame) -> None:
    """[2] Weekday x store mean revenue pivot — change only aggfunc and answer a new question."""
    print("\n[2] Weekday x store 'mean' revenue pivot (aggfunc='mean')")
    pivot = pd.pivot_table(df, index="weekday", columns="store",
                           values="revenue", aggfunc="mean")
    pivot = pivot.reindex(WEEKDAY_ORDER)  # alphabetical -> restore the Mon-Sun business order
    print((pivot / 1e4).round(0).astype(int).to_string())
    print("    (unit: 10k KRW — check the 'weekend effect': the Sat/Sun rows "
          "run above the weekdays)")


def step3_melt(df: pd.DataFrame) -> None:
    """[3] melt: turn the wide pivot back into long format."""
    print("\n[3] melt — putting the dresser (wide) back on the clothes rack (long)")
    wide = pd.pivot_table(df, index="store", columns="category",
                          values="revenue", aggfunc="sum")  # the margins-free version
    long = wide.reset_index().melt(id_vars="store",
                                   var_name="category", value_name="revenue")
    print(f"    wide: {wide.shape[0]} rows x {wide.shape[1]} cols (easy for humans to read)")
    print(f"    long: {long.shape[0]} rows x {long.shape[1]} cols (easy for machines to handle)")
    print("    First 3 rows of long:")
    print(long.head(3).to_string(index=False))
    print("    -> Analysis, charts, and merges want long; report tables want wide.")


def step4_rankings(df: pd.DataFrame) -> None:
    """[4] rank: company-wide ranks and within-category ranks."""
    print("\n[4] Rankings — rank and groupby+rank")
    store_total = df.groupby("store")["revenue"].sum()
    ranking = store_total.rank(ascending=False).astype(int).sort_values()
    print("    Company-wide ranking of stores by total revenue:")
    for store, r in ranking.items():
        print(f"      #{r} {store} ({store_total[store] / 1e8:.1f} x100M KRW)")

    # Within-category store ranks: "where does Downtown rank in the Coffee segment?"
    cat_store = df.groupby(["category", "store"], as_index=False)["revenue"].sum()
    cat_store["rank_in_category"] = (cat_store.groupby("category")["revenue"]
                                     .rank(ascending=False).astype(int))
    coffee = cat_store[cat_store["category"] == "Coffee"].sort_values("rank_in_category")
    print("    Ranks within the 'Coffee' category:")
    for _, row in coffee.iterrows():
        print(f"      #{row['rank_in_category']} {row['store']} "
              f"({row['revenue'] / 1e8:.1f} x100M KRW)")


def step5_monthly_flow(df: pd.DataFrame) -> None:
    """[5] Attach pct_change (month over month) and cumsum (running total) to monthly revenue."""
    print("\n[5] The monthly flow — pct_change and cumsum")
    monthly = (df.groupby(df["real_date"].dt.to_period("M"))["revenue"].sum())
    report = pd.DataFrame({
        "revenue_100m": (monthly / 1e8).round(2),                 # monthly revenue (100M KRW)
        "mom_pct": (monthly.pct_change() * 100).round(1),         # vs previous month %
        "cum_100m": (monthly.cumsum() / 1e8).round(1),            # running total (100M KRW)
    })
    print(report.to_string())
    print("    (the NaN in mom_pct's first row is the honest statement "
          "'no previous month to compare against')")


def step6_save_report(pivot: pd.DataFrame) -> None:
    """[6] Save the finished pivot report as a CSV."""
    print("\n[6] Saving the report")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = OUT_DIR / "pivot_report.csv"
    (pivot / 1e8).round(3).to_csv(out_path, encoding="utf-8-sig")  # Excel-friendly encoding
    print(f"    Saved: {out_path}")
    print("    Open it in Excel and you'll see the pivot you used to build by hand, "
          "reproduced by one run of code.")


def main() -> None:
    print("=" * 60)
    print("Level 10 — Pivoting, Reshaping, and Window Operations")
    print("=" * 60)
    df = load_clean_sales()
    pivot = step1_store_category_pivot(df)
    step2_weekday_pivot(df)
    step3_melt(df)
    step4_rankings(df)
    step5_monthly_flow(df)
    step6_save_report(pivot)
    print("\nDone! Try answering different questions with the same pivot_table, "
          "changing only the aggfunc.")


if __name__ == "__main__":
    main()
