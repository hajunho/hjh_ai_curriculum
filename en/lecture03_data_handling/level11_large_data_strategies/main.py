"""
Hands-on strategies for processing large datasets.
Working on 8 years of sales data (73,000 rows), we
  - measure per-column memory with memory_usage(deep=True)
  - compare before/after a dtype diet (category / int32 / float32)
  - run a chunked streaming aggregation with read_csv(chunksize=...) and verify it
  - compare the time/memory effect of reading with usecols/dtype specified
and finish by summarizing the criteria for "when to leave pandas".
"""

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

# Path setup so we can import the shared data module (hjh_data)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
CSV_PATH = OUT_DIR / "big_sales.csv"


def fmt_bytes(n: float) -> str:
    """Format a byte count into a human-friendly unit."""
    return f"{n / 1024 / 1024:.2f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"


def step1_measure(df: pd.DataFrame) -> int:
    """[1] Step on the scale: measure per-column memory with deep=True."""
    print(f"\n[1] memory_usage(deep=True) — per-column memory of {len(df):,} rows")
    mem = df.memory_usage(deep=True)
    for col, n_bytes in mem.items():
        dtype = "-" if col == "Index" else str(df[col].dtype)
        print(f"    {str(col):10s} ({dtype:7s}): {fmt_bytes(n_bytes):>10s}")
    total = int(mem.sum())
    print(f"    Total: {fmt_bytes(total)}")
    print("    -> The string (object) columns are the 'memory hippos', several times "
          "heavier than the numeric ones.")
    return total


def step2_dtype_diet(df: pd.DataFrame, before: int) -> pd.DataFrame:
    """[2] The dtype diet: convert to category / int32 / float32 and compute the savings."""
    print("\n[2] The dtype diet — same data, smaller clothes")
    opt = df.copy()
    # Highly repetitive strings -> category (compressed into a dictionary + integer codes)
    for col in ["date", "weekday", "store", "category"]:
        opt[col] = opt[col].astype("category")
    # Integer columns whose value range fits comfortably in int32 -> half the size
    opt["day_index"] = opt["day_index"].astype("int32")
    opt["ad_cost"] = opt["ad_cost"].astype("int32")
    # Analysis-grade float column -> float32 (keep float64 for accounting reports)
    opt["revenue"] = opt["revenue"].astype("float32")

    after = int(opt.memory_usage(deep=True).sum())
    saving = (1 - after / before) * 100
    print(f"    Before optimizing: {fmt_bytes(before)}")
    print(f"    After optimizing : {fmt_bytes(after)}  (savings {saving:.1f}%)")

    # Verify the values were not damaged (categories converted back to strings to compare)
    assert (opt["store"].astype(str) == df["store"]).all()
    assert (opt["day_index"].astype("int64") == df["day_index"]).all()
    ok = np.allclose(opt["revenue"].astype("float64"), df["revenue"], equal_nan=True)
    print(f"    Check: string/integer columns match exactly, revenue matches within "
          f"float32 precision ({ok})")
    print("    -> The category principle: instead of writing 'Downtown' 73,000 times, "
          "write it once in a dictionary and store only its number")
    return opt


def step3_chunked_aggregation(df: pd.DataFrame) -> None:
    """[3] Chunked streaming aggregation: verify reading in pieces matches the full aggregate."""
    print("\n[3] read_csv(chunksize) — the pattern for files bigger than memory")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"    CSV saved: {CSV_PATH} ({fmt_bytes(CSV_PATH.stat().st_size)})")

    # Iterate over the chunks: accumulate the partial aggregates (sums).
    total_by_store = None
    n_chunks = 0
    for chunk in pd.read_csv(CSV_PATH, chunksize=10_000):
        part = chunk.groupby("store")["revenue"].sum()
        total_by_store = part if total_by_store is None else total_by_store.add(part, fill_value=0)
        n_chunks += 1
    print(f"    Accumulated per-store totals across {n_chunks} chunks of 10,000 rows each.")

    # Verify against the result of reading everything at once
    full = pd.read_csv(CSV_PATH).groupby("store")["revenue"].sum()
    match = np.allclose(total_by_store.sort_index(), full.sort_index())
    print(f"    Check: chunked aggregate == full aggregate ? {match}")
    top = total_by_store.sort_values(ascending=False)
    print("    Per-store revenue totals (100M KRW): "
          + ", ".join(f"{s} {v / 1e8:.0f}" for s, v in top.items()))
    print("    -> Sums, counts, max/min work with this pattern; statistics that need "
          "the whole dataset, like the median, do not.")


def step4_read_optimized() -> None:
    """[4] Optimized reading: compare the time/memory effect of usecols + dtype."""
    print("\n[4] Optimized reading — only the columns you need, in the right types")
    t0 = time.perf_counter()
    naive = pd.read_csv(CSV_PATH)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    smart = pd.read_csv(CSV_PATH, usecols=["store", "revenue"],
                        dtype={"store": "category", "revenue": "float32"})
    t_smart = time.perf_counter() - t0

    mem_naive = naive.memory_usage(deep=True).sum()
    mem_smart = smart.memory_usage(deep=True).sum()
    print(f"    Full read      : {t_naive * 1000:6.0f} ms / memory {fmt_bytes(mem_naive)}")
    print(f"    usecols+dtype  : {t_smart * 1000:6.0f} ms / memory {fmt_bytes(mem_smart)}"
          f"  (memory saved {(1 - mem_smart / mem_naive) * 100:.0f}%)")
    print("    -> On a small file the time gap is small, but the memory savings "
          "are always real.")
    # Note: with the column-oriented Parquet format this is one line (needs pyarrow).
    #   df.to_parquet("big_sales.parquet")
    #   pd.read_parquet("big_sales.parquet", columns=["store", "revenue"])
    # Column pruning, type preservation, and compression are built in — the standard
    # for data you read repeatedly.


def step5_when_to_leave_pandas() -> None:
    """[5] The judgment call: when to leave pandas for a DB/Spark."""
    print("\n[5] When to leave pandas — the criteria in one table")
    rules = [
        ("data <= 1/3 of memory", "just use pandas"),
        ("fits in memory, but barely", "dtype diet + save as Parquet"),
        ("bigger than memory, goal is aggregation", "chunksize streaming, or Polars/DuckDB"),
        ("several people query/update concurrently", "a database (DB) — lecture04"),
        ("hundreds of millions of rows, multiple servers", "distributed processing like Spark"),
    ]
    for cond, action in rules:
        print(f"    {cond:44s} -> {action}")
    print("    -> The criterion is the nature of the work, not the GB count. "
          "A distributed system is a cost in itself.")


def main() -> None:
    print("=" * 60)
    print("Level 11 — Strategies for Processing Large Datasets")
    print("=" * 60)
    # 8 years (2,920 days) x 5 stores x 5 categories = 73,000 rows. Fixed seed, reproducible.
    df = pd.DataFrame(hjh_data.sales_table(n_days=2920, seed=42))
    before = step1_measure(df)
    step2_dtype_diet(df, before)
    step3_chunked_aggregation(df)
    step4_read_optimized()
    step5_when_to_leave_pandas()
    print("\nDone! That wraps up lecture03. Next: lecture04 — Databases and SQL.")


if __name__ == "__main__":
    main()
