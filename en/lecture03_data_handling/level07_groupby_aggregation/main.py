"""
Hands-on groupby aggregation.
After cleaning a year of cafe-chain sales, we build a per-store / per-weekday /
per-category analysis report using the split-apply-combine pattern.
We practice the difference between agg (multiple aggregations) and transform
(keeps the original size), plus weekday reordering with reindex — the exact
flow of a real-world report.
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_clean_sales() -> pd.DataFrame:
    """Load the sales data and clean it level06-style (negatives -> NaN, drop missing)."""
    rows = hjh_data.sales_table(n_days=365, seed=42)  # fixed seed
    df = pd.DataFrame(rows)
    df.loc[df["revenue"] < 0, "revenue"] = np.nan  # domain rule: revenue cannot be negative
    df = df.dropna(subset=["revenue"])             # aggregation is the goal here, so simply drop
    return df


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_clean_sales()
    print(f"Cleaning done: {len(df):,} rows (negatives removed + missing dropped)\n")

    # ------------------------------------------------------------------
    print("[1] Per-store summary — total, mean, and count at once with agg")
    store_report = (
        df.groupby("store")
          .agg(total=("revenue", "sum"),
               avg=("revenue", "mean"),
               n=("revenue", "count"))
          .sort_values("total", ascending=False)
    )
    store_report["total_mil"] = (store_report["total"] / 1e6).round(1)  # millions of KRW
    print(store_report[["total_mil", "avg", "n"]].round(0).to_string())
    print("  -> Why Lakeside and University run big: a store-size effect is "
          "planted in the data.")

    # ------------------------------------------------------------------
    print("\n[2] Mean revenue per weekday — beating the alphabetical-order trap with reindex")
    by_weekday = df.groupby("weekday")["revenue"].mean().reindex(WEEKDAY_ORDER)
    print(by_weekday.round(0).to_string())
    weekend_avg = by_weekday[["Sat", "Sun"]].mean()
    weekday_avg = by_weekday[["Mon", "Tue", "Wed", "Thu", "Fri"]].mean()
    print(f"  Weekend mean {weekend_avg:,.0f} / weekday mean {weekday_avg:,.0f}"
          f" -> weekend multiple {weekend_avg / weekday_avg:.2f}x")
    print("  -> We recovered the weekend effect planted in the data (about 1.2x) "
          "through aggregation.")

    # ------------------------------------------------------------------
    print("\n[3] Total revenue and share (%) per category")
    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    share = (by_cat / by_cat.sum() * 100).round(1)
    cat_table = pd.DataFrame({"total (M KRW)": (by_cat / 1e6).round(1), "share (%)": share})
    print(cat_table.to_string())

    # ------------------------------------------------------------------
    print("\n[4] Store x category — multi-key groupby, top 5 combinations")
    combo = (
        df.groupby(["store", "category"])["revenue"].sum()
          .sort_values(ascending=False)
          .head(5)
    )
    print((combo / 1e6).round(1).to_string())
    print("  -> This is a MultiIndex result. Add reset_index() to use it as a plain table.")

    # ------------------------------------------------------------------
    print("\n[5] transform — building the derived column 'share within the store'")
    store_total = df.groupby("store")["revenue"].transform("sum")  # row count preserved!
    df = df.assign(share_in_store=df["revenue"] / store_total)
    sample = df.loc[df["store"] == "Downtown",
                    ["date", "category", "revenue", "share_in_store"]].head(3)
    print(sample.to_string(index=False))
    check = df.groupby("store")["share_in_store"].sum().round(6)
    print(f"  Check: per-store share totals = {[float(v) for v in check.unique()]} "
          f"(all should be 1.0)")

    # ------------------------------------------------------------------
    print("\n[6] Saving the final report")
    csv_path = os.path.join(OUT_DIR, "store_report.csv")
    store_report.reset_index().to_csv(csv_path, index=False, encoding="utf-8-sig")
    top_store = store_report.index[0]
    top_cat = by_cat.index[0]
    print(f"  Per-store summary saved: {csv_path}")
    print("  ---- One-line report ----")
    print(f"  Top store by revenue: {top_store} "
          f"({store_report.loc[top_store, 'total_mil']:,} M KRW)")
    print(f"  Top category by revenue: {top_cat} (share {share[top_cat]}%)")
    print(f"  Weekend effect: {weekend_avg / weekday_avg:.2f}x the weekday level")
    print("\nRecap: every 'what does it look like per ...?' question becomes "
          "one line of groupby.")


if __name__ == "__main__":
    main()
