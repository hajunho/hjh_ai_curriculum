"""
Hands-on merging (merge) and stacking (concat).
We join three tables — the sales table + the store info table + the targets
table — to build a ranking of target achievement per store.
We also witness the row-count differences of inner/left/outer, the indicator
audit, and the row-explosion trap caused by duplicate keys.
"""

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def build_performance() -> pd.DataFrame:
    """Clean a year of sales and build the per-store total performance table."""
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))  # fixed seed
    df.loc[df["revenue"] < 0, "revenue"] = np.nan  # domain-rule cleaning
    df = df.dropna(subset=["revenue"])
    perf = (df.groupby("store", as_index=False)
              .agg(actual=("revenue", "sum")))
    return df, perf


def main() -> None:
    # ------------------------------------------------------------------
    print("[1] Per-store performance (level07 recap)")
    df, perf = build_performance()
    perf["actual_mil"] = (perf["actual"] / 1e6).round(1)  # millions of KRW
    print(perf[["store", "actual_mil"]].to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[2] Preparing the side tables — the facilities roster and the "
          "planning team's targets (built in code)")
    # Airport is deliberately missing, and the brand-new 'Parkview' store,
    # which has no sales yet, is included instead.
    store_info = pd.DataFrame({
        "store": ["Downtown", "Riverside", "University", "Lakeside", "Parkview"],
        "region": ["Metro", "Metro", "Central", "Metro", "Metro"],
        "open_year": [2015, 2017, 2019, 2021, 2025],
        "manager": ["Rachel Adams", "Peter Shaw", "Dylan Reed", "Hannah Cole", "Mason Frey"],
    })
    targets = pd.DataFrame({
        "store": ["Downtown", "Riverside", "Airport", "University", "Lakeside"],
        "target_mil": [800, 850, 900, 1000, 1100],  # annual targets (millions of KRW)
    })
    print(f"  store_info {len(store_info)} rows (Airport missing, Parkview new)")
    print(f"  targets    {len(targets)} rows")

    # ------------------------------------------------------------------
    print("\n[3] Comparing the how's — same two tables, different row counts")
    for how in ["inner", "left", "outer"]:
        merged = pd.merge(perf, store_info, on="store", how=how)
        print(f"  how='{how:5s}' -> {len(merged)} rows")
    print("  (right is just left with the tables swapped, so it is skipped)")
    audit = pd.merge(perf, store_info, on="store", how="outer", indicator=True)
    print("  indicator audit (the _merge column):")
    print(audit["_merge"].value_counts().to_string())
    only_left = audit.loc[audit["_merge"] == "left_only", "store"].tolist()
    only_right = audit.loc[audit["_merge"] == "right_only", "store"].tolist()
    print(f"  -> Has sales but missing from the roster: {only_left} / "
          f"only on the roster: {only_right}")

    # ------------------------------------------------------------------
    print("\n[4] Merging the targets -> achievement rate and ranking")
    result = pd.merge(perf, targets, on="store", how="left")
    result["achieve_pct"] = (result["actual_mil"] / result["target_mil"] * 100).round(1)
    result = result.sort_values("achieve_pct", ascending=False).reset_index(drop=True)
    result.index = result.index + 1  # display starting from rank 1
    print(result[["store", "actual_mil", "target_mil", "achieve_pct"]].to_string())
    best = result.iloc[0]
    print(f"  -> #1 {best['store']}: {best['achieve_pct']}% of target achieved")

    # ------------------------------------------------------------------
    print("\n[5] The duplicate-key trap — rows multiply without a sound")
    dup_targets = pd.concat(
        [targets, targets.iloc[[0]]], ignore_index=True)  # Downtown's target twice!
    print(f"  Contaminated targets table: {len(dup_targets)} rows "
          f"({int(dup_targets['store'].duplicated().sum())} duplicate for Downtown)")
    boom = pd.merge(perf, dup_targets, on="store", how="left")
    print(f"  Merge result: {len(perf)} rows -> {len(boom)} rows "
          f"(Downtown cloned into 2 rows!)")
    total_ok = perf["actual_mil"].sum()
    total_boom = boom["actual_mil"].sum()
    print(f"  Total performance: clean {total_ok:,.1f} vs contaminated "
          f"{total_boom:,.1f} M KRW (+{total_boom - total_ok:,.1f} inflated)")
    safe = pd.merge(perf, dup_targets.drop_duplicates(subset="store"),
                    on="store", how="left")
    print(f"  Prevention: merge after drop_duplicates -> {len(safe)} rows (back to normal)")
    print("  Habit: check duplicated() before merging + compare len() before and after!")

    # ------------------------------------------------------------------
    print("\n[6] concat — not joining sideways but 'stacking downward'")
    first_half = (df[df["day_index"] < 182].groupby("store", as_index=False)
                  .agg(revenue_mil=("revenue", "sum")))
    second_half = (df[df["day_index"] >= 182].groupby("store", as_index=False)
                   .agg(revenue_mil=("revenue", "sum")))
    first_half["half"], second_half["half"] = "H1", "H2"
    stacked = pd.concat([first_half, second_half], ignore_index=True)
    stacked["revenue_mil"] = (stacked["revenue_mil"] / 1e6).round(1)
    print(f"  H1 {len(first_half)} rows + H2 {len(second_half)} rows"
          f" = concat {len(stacked)} rows (same column structure, stacked vertically)")
    print(stacked.head(3).to_string(index=False))
    print("\nRecap: with a base table use left; when auditing use outer+indicator;")
    print("      and check for duplicate keys before merging — those three prevent "
          "the merge accidents.")


if __name__ == "__main__":
    main()
