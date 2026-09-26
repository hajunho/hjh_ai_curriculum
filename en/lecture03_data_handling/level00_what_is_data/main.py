"""
What is data? — dissecting the structure of a table using a plain list of dicts.
We confirm with our own eyes that a row = one case and a column = one attribute,
then extract columns, find rows, inspect types/missing values, and build a mini
schema summary using nothing but the standard library.
At the end we compare against unstructured text to see why a 'table' is so
much easier to aggregate.
"""

import pathlib
import sys

# Path setup so we can import the shared data module (hjh_data)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def extract_column(rows: list[dict], col: str) -> list:
    """Pull one column out of the table as a list. (column = values of the same attribute)"""
    return [r[col] for r in rows]


def summarize_schema(rows: list[dict]) -> list[dict]:
    """Count the types and missing values per column to build a 'mini schema summary'."""
    summary = []
    for col in rows[0].keys():
        values = extract_column(rows, col)
        none_count = sum(1 for v in values if v is None)
        # Collect the type names of the non-None values
        type_names = sorted({type(v).__name__ for v in values if v is not None})
        summary.append({
            "column": col,
            "types": "/".join(type_names),
            "missing": none_count,
            "example": next(v for v in values if v is not None),
        })
    return summary


def main() -> None:
    # Generate 30 days of fictional cafe sales (fixed seed -> same result every time)
    rows = hjh_data.sales_table(n_days=30, seed=42)

    print("[1] Table size and schema (column names)")
    print(f"    Number of rows (cases): {len(rows)}")
    print(f"    Columns (attributes): {list(rows[0].keys())}")
    print()

    print("[2] Preview — always eyeball a table before anything else")
    hjh_data.head(rows, n=5)
    print()

    print("[3] Dissecting one row — what exactly is 'one row' in this table?")
    first = rows[0]
    for key, value in first.items():
        print(f"    {key:>10} = {value!r}  ({type(value).__name__})")
    print("    -> One row = one day's revenue for a specific date, store, and category.")
    print()

    print("[4] Extracting a column — just the revenue column")
    revenues = extract_column(rows, "revenue")
    print(f"    Length of the revenue column: {len(revenues)} (same as the row count)")
    print(f"    First 8 values: {revenues[:8]}")
    print()

    print("[5] Finding rows by condition — only Downtown's Coffee revenue")
    gangnam_coffee = [r for r in rows
                     if r["store"] == "Downtown" and r["category"] == "Coffee"]
    print(f"    Matching rows: {len(gangnam_coffee)} (30 days, so 30 is correct)")
    hjh_data.head(gangnam_coffee, n=3)
    print()

    print("[6] Mini schema summary — types and missing values (None) per column")
    schema = summarize_schema(rows)
    for s in schema:
        print(f"    {s['column']:>10} | types: {s['types']:<8} | "
              f"missing: {s['missing']:>2} | example: {s['example']!r}")
    missing_total = sum(s["missing"] for s in schema)
    negative_count = sum(1 for v in revenues if v is not None and v < 0)
    print(f"    -> Hiding in this table: {missing_total} missing cells and "
          f"{negative_count} negative revenue rows.")
    print("      (Deliberately planted contamination. level06 teaches how to handle it.)")
    print()

    print("[7] Structured vs unstructured — same information, different shape")
    unstructured = ("Went to the Downtown store yesterday, the coffee was great. "
                    "Pretty crowded, so they probably had a good sales day!")
    structured = {"date": "2025-01-01", "store": "Downtown",
                  "category": "Coffee", "revenue": 512000}
    print(f"    Unstructured (free text): {unstructured!r}")
    print(f"    Structured (one table row): {structured!r}")
    print("    -> You cannot run SUM on a sentence, but the table's revenue column adds up instantly.")
    print("      To analyze unstructured data, you eventually convert it into a table too.")


if __name__ == "__main__":
    main()
