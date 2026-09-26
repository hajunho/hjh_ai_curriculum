"""
Hands-on practice bringing data from files into a DataFrame.
(1) Save/read a healthy CSV, (2) repair a troublemaker CSV with comma-formatted
numbers, '-' as missing, and semicolon separators using read_csv options,
(3) reproduce and fix a cp949 encoding error, (4) round-trip records-style JSON —
four real-world 'customs incidents', all solved in code.
"""

import os
import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parent
OUT = BASE / "outputs"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 110)

    # ------------------------------------------------------------------
    print("[1] Healthy CSV — save it and read it back")
    rows = hjh_data.sales_table(n_days=30, seed=42)   # fixed seed
    df = pd.DataFrame(rows)
    csv_path = OUT / "sales_30d.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")   # don't forget index=False!
    print(f"    Saved: {csv_path} ({len(df)} rows)")
    loaded = pd.read_csv(csv_path)
    print(f"    Read back: {loaded.shape[0]} rows x {loaded.shape[1]} cols, dtype summary:")
    print("    " + ", ".join(f"{c}={t}" for c, t in loaded.dtypes.items()))
    print("    -> revenue, which has missing values, reads back as float64.")

    # ------------------------------------------------------------------
    print("\n[2] Troublemaker CSV — comma numbers, '-' as missing, semicolon separators")
    messy_path = OUT / "messy.csv"
    with open(messy_path, "w", encoding="utf-8") as f:
        f.write("date;store;revenue\n")
        f.write("2025-01-01;Downtown;1,234,000\n")
        f.write("2025-01-02;Downtown;-\n")            # a system that writes missing as '-'
        f.write("2025-01-03;Riverside;987,500\n")
        f.write("2025-01-04;Airport;1,050,000\n")
    print(f"    Created: {messy_path}")

    naive = pd.read_csv(messy_path)                  # what if we read it with no options?
    print(f"    Read with no options -> {naive.shape[1]} column(s) "
          f"(it missed the semicolons: one big lump!)")
    print(f"      columns = {list(naive.columns)}")

    fixed = pd.read_csv(messy_path, sep=";", thousands=",", na_values="-")
    print("    After the 3 options (sep=';', thousands=',', na_values='-'):")
    print(fixed.to_string(index=False))
    print(f"      revenue dtype = {fixed['revenue'].dtype} -> "
          f"sum KRW {fixed['revenue'].sum():,.0f} is now computable")

    # ------------------------------------------------------------------
    print("\n[3] Encoding incident — opening a cp949 file with the utf-8 key?")
    # Our fictional chain's legacy Korean POS system exports store names in
    # Korean and saves in cp949 — a scenario you WILL meet with older systems.
    cp949_path = OUT / "sales_cp949.csv"
    with open(cp949_path, "w", encoding="cp949") as f:
        f.write("date,store,category,revenue\n")
        f.write("2025-01-01,강남점,Coffee,541509\n")
        f.write("2025-01-01,강남점,Bakery,402678\n")
        f.write("2025-01-02,홍대점,Coffee,389180\n")
    print(f"    Created: {cp949_path} (saved as cp949, the legacy Korean encoding)")
    try:
        pd.read_csv(cp949_path, encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"    Reading as utf-8 -> UnicodeDecodeError!")
        print(f"      Message excerpt: {str(e)[:70]}...")
    rescued = pd.read_csv(cp949_path, encoding="cp949")
    print("    Read again with encoding='cp949' and the text is intact:")
    print(rescued[["date", "store", "category", "revenue"]].head(3).to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[4] JSON — save in records form and read it back")
    json_path = OUT / "sales_records.json"
    sample = df.head(3)[["date", "store", "category", "revenue"]]
    sample.to_json(json_path, orient="records", force_ascii=False)
    print(f"    Saved: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        raw = f.read()
    print(f"    Start of the file: {raw[:80]}...")
    from_json = pd.read_json(json_path, orient="records")
    print("    Table read back with pd.read_json:")
    print(from_json.to_string(index=False))
    print("    -> One row = one dictionary. The standard shape for passing data "
          "between systems.")

    # Note: Excel files are read like below after installing openpyxl (concept only here).
    #   df = pd.read_excel("report.xlsx", sheet_name="January", header=2)

    print("\nRecap: never fix the source file by hand — solve it with read_csv options.")
    print("      sep / encoding / thousands / na_values cover 90% of the incidents.")


if __name__ == "__main__":
    main()
