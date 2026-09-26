"""
From Excel to Python — save the sales table to CSV, reload it, and
reproduce the SUM / SUMIF / auto-filter / sort you used to do in Excel,
using only the standard library.
The script reports how many missing/negative rows it skipped, so you
experience code as a 'verifiable procedure' and the value of reproducibility.
"""

import csv
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

BASE_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"


def load_clean_rows(csv_path: str) -> tuple[list[dict], int, int]:
    """Read the CSV and restore revenue to a number.
    Missing (empty string) and negative rows are skipped, and their counts
    are returned alongside the clean rows."""
    clean, n_missing, n_negative = [], 0, 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw = r["revenue"]
            if raw == "" or raw == "None":          # missing -> skip
                n_missing += 1
                continue
            revenue = int(raw)
            if revenue < 0:                          # negative contamination -> skip
                n_negative += 1
                continue
            r["revenue"] = revenue                   # restore the type (str -> int)
            r["ad_cost"] = int(r["ad_cost"])
            clean.append(r)
    return clean, n_missing, n_negative


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] Save 60 days of sales as CSV (a format Excel can open too)")
    rows = hjh_data.sales_table(n_days=60, seed=42)
    csv_path = hjh_data.to_csv(rows, str(OUT_DIR / "sales.csv"))
    print(f"    Saved: {csv_path} ({len(rows)} rows)")
    print()

    print("[2] Read the CSV back — every value comes in as a 'string'")
    with open(csv_path, newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    print(f"    First row: {first}")
    print(f"    Type of revenue: {type(first['revenue']).__name__} <- not a number!")
    print()

    print("[3] Restore types + skip contaminated rows (always report what you threw away)")
    clean, n_missing, n_negative = load_clean_rows(csv_path)
    print(f"    Kept {len(clean)} rows / skipped {n_missing} missing / "
          f"skipped {n_negative} negative")
    print()

    print("[4] SUM — the equivalent of Excel's =SUM(G:G)")
    total = sum(r["revenue"] for r in clean)
    print(f"    Total revenue over 60 days: KRW {total:,}")
    print()

    print("[5] SUMIF — totals per store (accumulating in a dictionary)")
    by_store: dict[str, int] = {}
    for r in clean:
        by_store[r["store"]] = by_store.get(r["store"], 0) + r["revenue"]
    for store, subtotal in sorted(by_store.items(), key=lambda kv: kv[1]):
        print(f"    {store}: KRW {subtotal:>15,}")
    print()

    print("[6] Auto filter — only 'Downtown & weekend' rows, compared with weekdays")
    gangnam = [r for r in clean if r["store"] == "Downtown"]
    weekend = [r for r in gangnam if r["weekday"] in ("Sat", "Sun")]
    weekday_rows = [r for r in gangnam if r["weekday"] not in ("Sat", "Sun")]
    avg_weekend = sum(r["revenue"] for r in weekend) / len(weekend)
    avg_weekday = sum(r["revenue"] for r in weekday_rows) / len(weekday_rows)
    print(f"    Downtown weekend average: KRW {avg_weekend:>12,.0f} ({len(weekend)} rows)")
    print(f"    Downtown weekday average: KRW {avg_weekday:>12,.0f} ({len(weekday_rows)} rows)")
    print(f"    -> Weekends run at {avg_weekend / avg_weekday:.2f}x the weekday level.")
    print()

    print("[7] Sort — top 5 by revenue (Excel's descending sort)")
    top5 = sorted(clean, key=lambda r: r["revenue"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        print(f"    #{i} | {r['date']} {r['weekday']} | {r['store']} "
              f"{r['category']} | KRW {r['revenue']:,}")
    print()

    print("[8] Reproducibility — the real value of this script")
    print("    Every number you just saw came from a 'fixed seed + a recorded procedure'.")
    print("    Run it again tomorrow, or let someone else run it — the result is identical.")
    print("    Excel clicks live in someone's memory; code lives on as a document.")


if __name__ == "__main__":
    main()
