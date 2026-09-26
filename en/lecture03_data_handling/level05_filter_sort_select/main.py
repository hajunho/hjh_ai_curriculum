"""
Hands-on practice answering business questions with filtering, sorting, and selecting.
Working on 180 days of cafe-chain sales, we use boolean masks, condition
combinations (&, |, ~, isin), loc/iloc, and sort_values/nlargest to translate
five real questions — like "top 10 weekend Coffee sales at Downtown" — into code.
"""

import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 120)

    # ------------------------------------------------------------------
    print("[0] Data prep — 180 days of sales; missing rows simply dropped for this level")
    df = pd.DataFrame(hjh_data.sales_table(n_days=180, seed=42))  # fixed seed
    n_missing = df["revenue"].isna().sum()
    df = df.dropna(subset=["revenue"]).copy()   # proper handling strategies come in level06
    print(f"    Dropped {n_missing} missing revenue rows -> starting with {len(df)} rows.")

    # ------------------------------------------------------------------
    print("\n[Q1] Pull the top 10 weekend Coffee sales at the Downtown store.")
    q1 = df[
        (df["store"] == "Downtown")
        & (df["weekday"].isin(["Sat", "Sun"]))   # one of several values -> isin
        & (df["category"] == "Coffee")
    ]
    top10 = q1.nlargest(10, "revenue")           # for 'top N', nlargest is the clean way
    print(f"    Top 10 out of {len(q1)} matching rows:")
    print(top10[["date", "weekday", "revenue"]].to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[Q2] Which 'efficient' rows had revenue of at least KRW 500k "
          "on ad spend of at most KRW 200k?")
    q2 = df[(df["revenue"] >= 500_000) & (df["ad_cost"] <= 200_000)]
    print(f"    {len(q2)} matching rows ({len(q2) / len(df) * 100:.1f}% of the total)")
    print("    Top 3 by revenue:")
    print(q2.nlargest(3, "revenue")[["date", "store", "category", "ad_cost", "revenue"]]
          .to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[Q3] Only the Dessert and Bakery results for the Airport and University "
          "stores, please. (isin twice)")
    q3 = df[df["store"].isin(["Airport", "University"])
            & df["category"].isin(["Dessert", "Bakery"])]
    print(f"    {len(q3)} matching rows. Mean revenue per store x category combination:")
    summary = q3.groupby(["store", "category"])["revenue"].mean().round(0)
    print(summary.to_string())
    print("    (groupby is taught properly in level07. This is just a taste!)")

    # ------------------------------------------------------------------
    print("\n[Q4] How do loc (labels) and iloc (positions) differ?")
    busan = df[df["store"] == "Airport"]
    first_label = busan.index[0]
    print(f"    Start of the index after filtering to Airport: {list(busan.index[:5])}")
    print("    -> Not 0,1,2... — it still carries the original row numbers!")
    print(f"    busan.iloc[0]  : the 'first row' by position -> date={busan.iloc[0]['date']}, "
          f"revenue={busan.iloc[0]['revenue']:,.0f}")
    print(f"    busan.loc[{first_label}]: the row labeled {first_label} -> the same row.")
    print("    And busan.loc[0]? If label 0 is not in the Airport table, you get a KeyError.")
    # Row condition + column selection at once: the most common idiom at work
    picked = df.loc[df["revenue"] >= 900_000, ["date", "store", "category", "revenue"]]
    print(f"    df.loc[condition, columns] example: revenue >= 900k, "
          f"{len(picked)} rows, first 3:")
    print(picked.head(3).to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[Q5] Which rows had the best revenue per ad spend (roas)?")
    df["roas"] = df["revenue"] / df["ad_cost"]      # derived column
    q5 = df.sort_values("roas", ascending=False)    # descending sort
    print("    Top 5 by roas:")
    print(q5.head(5)[["date", "store", "category", "ad_cost", "revenue", "roas"]]
          .round(2).to_string(index=False))
    print("    Multi-key sort example: store ascending + revenue descending, "
          "top 1 for each store")
    multi = df.sort_values(["store", "revenue"], ascending=[True, False])
    print(multi.groupby("store").head(1)[["store", "date", "category", "revenue"]]
          .to_string(index=False))

    print("\nRecap: a business question = (pick rows by combined conditions) "
          "+ (sort / top N) + (select columns with loc).")
    print("      Parentheses around every condition, &/| instead of and/or, "
          "and iloc[0] for the first row — those three habits prevent most errors.")


if __name__ == "__main__":
    main()
