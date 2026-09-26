"""
A hands-on dissection of the structure of Pandas Series and DataFrame.
We turn 90 days of cafe-chain sales (hjh_data.sales_table) into a DataFrame,
walk through shape / index / columns / dtypes / head / info / describe,
pull one column out as a Series, build a derived column, and finish by
diagnosing the missing-value counts.
"""

import io
import pathlib
import sys

import pandas as pd

# Add the path so the shared data module (common/hjh_data.py) can be imported.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 110)
    pd.set_option("display.max_columns", 10)

    # ------------------------------------------------------------------
    print("[1] List of dictionaries -> DataFrame")
    rows = hjh_data.sales_table(n_days=90, seed=42)   # fixed seed: same data every time
    print(f"    Source: a Python list of dicts, {len(rows)} entries "
          f"(90 days x 5 stores x 5 categories)")
    df = pd.DataFrame(rows)
    print(f"    Conversion: one line of pd.DataFrame(rows) -> type {type(df).__name__}")

    # ------------------------------------------------------------------
    print("\n[2] The 3 parts of a table: values / row labels (index) / column names (columns)")
    print(f"    shape   : {df.shape}  ({df.shape[0]} rows, {df.shape[1]} columns)")
    print(f"    index   : {df.index}")
    print(f"    columns : {list(df.columns)}")

    # ------------------------------------------------------------------
    print("\n[3] Each column carries exactly one dtype")
    print(df.dtypes.to_string())
    print("    -> Why revenue is float64: mix NaN into an integer column "
          "and it turns into floats.")

    # ------------------------------------------------------------------
    print("\n[4] head() — preview of the first 5 rows "
          "(like scrolling to the top in Excel)")
    print(df.head().to_string())

    # ------------------------------------------------------------------
    print("\n[5] info() — the data health check "
          "(non-null count per column + dtype + memory)")
    buf = io.StringIO()                 # info() returns nothing and only prints, so we capture it
    df.info(buf=buf)
    print(buf.getvalue())

    # ------------------------------------------------------------------
    print("[6] describe() — summary statistics of the numeric columns")
    print(df[["ad_cost", "revenue"]].describe().round(1).to_string())
    print("    -> revenue's min is negative! A signal that outliers are mixed in "
          "(handled in level06).")

    # ------------------------------------------------------------------
    print("\n[7] Pull out one column and you get a Series — "
          "a labeled 1-D bundle of values")
    revenue = df["revenue"]
    print(f"    type(df['revenue']) = {type(revenue).__name__}")
    print(f"    length {len(revenue)}, dtype {revenue.dtype}")
    print(f"    mean KRW {revenue.mean():,.0f} / max KRW {revenue.max():,.0f} / "
          f"min KRW {revenue.min():,.0f}")
    print("    First 3 values (left number is the index, right is the value):")
    print(revenue.head(3).to_string())

    # ------------------------------------------------------------------
    print("\n[8] Creating a new column — revenue per ad-spend unit (roas)")
    df["roas"] = df["revenue"] / df["ad_cost"]        # Excel's 'enter formula and drag' in one line
    print(df[["store", "category", "ad_cost", "revenue", "roas"]].head(3).round(2).to_string())
    print(f"    roas mean: {df['roas'].mean():.2f} (revenue per 1 KRW of ad spend)")

    # ------------------------------------------------------------------
    print("\n[9] value_counts() — composition of a categorical column "
          "(repeated Excel COUNTIFs in one line)")
    print("    Rows per store:")
    print(df["store"].value_counts().to_string())
    print("    Rows per weekday (90 days is not divisible by 7, "
          "so the day counts differ):")
    print(df["weekday"].value_counts().to_string())

    # ------------------------------------------------------------------
    print("\n[10] Diagnosing missing values — isna().sum()")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())
    ratio = df["revenue"].isna().mean() * 100
    print(f"    -> revenue missing ratio {ratio:.2f}%. "
          f"Handling strategies come in level06.")

    print("\nRecap: DataFrame = index + columns + values. Pull out one column -> Series.")
    print("When new data arrives, greet it in this order: shape -> head -> info -> describe.")


if __name__ == "__main__":
    main()
