"""
フィルタリング・ソート・選択で業務の質問に答える実習です。
カフェチェーンの180日分の売上で、ブールマスク、条件の結合 (&, |, ~, isin)、
loc/iloc、sort_values/nlargest を使って
「渋谷店の週末のコーヒー売上上位10」のような実際の質問5つをコードへ翻訳します。
"""

import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 120)

    # ------------------------------------------------------------------
    print("[0] データ準備 — 180日分の売上、欠損行はこのレベルでは簡単に除去")
    df = pd.DataFrame(hjh_data.sales_table(n_days=180, seed=42))  # seed 固定
    n_missing = df["revenue"].isna().sum()
    df = df.dropna(subset=["revenue"]).copy()   # 正式な処理戦略は level06 で
    print(f"    欠損 revenue {n_missing}件を除去 -> {len(df)}行でスタートします。")

    # ------------------------------------------------------------------
    print("\n[質問1] 渋谷店の週末のコーヒー売上、上位10件を出してください。")
    q1 = df[
        (df["store"] == "渋谷店")
        & (df["weekday"].isin(["土", "日"]))     # 複数の値のどれか -> isin
        & (df["category"] == "コーヒー")
    ]
    top10 = q1.nlargest(10, "revenue")           # 「上位 N」は nlargest がすっきり
    print(f"    条件に合う行 {len(q1)}件のうち上位10件:")
    print(top10[["date", "weekday", "revenue"]].to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[質問2] 売上50万ウォン以上で、広告費は20万ウォン以下だった「効率の良い」行は?")
    q2 = df[(df["revenue"] >= 500_000) & (df["ad_cost"] <= 200_000)]
    print(f"    該当行 {len(q2)}件 (全体の {len(q2) / len(df) * 100:.1f}%)")
    print("    売上上位3件:")
    print(q2.nlargest(3, "revenue")[["date", "store", "category", "ad_cost", "revenue"]]
          .to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[質問3] 名古屋店・京都店のデザート・ベーカリーの実績だけを選んでください。(isin を2回)")
    q3 = df[df["store"].isin(["名古屋店", "京都店"])
            & df["category"].isin(["デザート", "ベーカリー"])]
    print(f"    該当行 {len(q3)}件。店舗 x カテゴリの組み合わせ別の平均売上:")
    summary = q3.groupby(["store", "category"])["revenue"].mean().round(0)
    print(summary.to_string())
    print("    (groupby は level07 で正式に学びます。ここでは味見!)")

    # ------------------------------------------------------------------
    print("\n[質問4] loc (名札) と iloc (位置) はどう違うのですか?")
    busan = df[df["store"] == "名古屋店"]
    first_label = busan.index[0]
    print(f"    名古屋店だけをフィルタした表の index の先頭部分: {list(busan.index[:5])}")
    print("    -> 0,1,2... ではなく、元の表の行番号をそのまま持っています!")
    print(f"    busan.iloc[0]  : 位置基準の「先頭の行」 -> date={busan.iloc[0]['date']}, "
          f"revenue={busan.iloc[0]['revenue']:,.0f}")
    print(f"    busan.loc[{first_label}]: 名札 {first_label}番の行 -> 同じ行です。")
    print("    busan.loc[0] は? 名札0番が名古屋店の表になければ KeyError になります。")
    # 行の条件 + 列の選択を同時に: 実務で最頻出の文型
    picked = df.loc[df["revenue"] >= 900_000, ["date", "store", "category", "revenue"]]
    print(f"    df.loc[条件, 列リスト] の例: 売上90万以上 {len(picked)}件のうち先頭3件")
    print(picked.head(3).to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[質問5] 広告費に対する売上効率 (roas) が最も良かった行はどこでしたか?")
    df["roas"] = df["revenue"] / df["ad_cost"]      # 派生列
    q5 = df.sort_values("roas", ascending=False)    # 降順ソート
    print("    roas 上位5件:")
    print(q5.head(5)[["date", "store", "category", "ad_cost", "revenue", "roas"]]
          .round(2).to_string(index=False))
    print("    複数基準ソートの例: 店舗昇順 + 売上降順で、各店舗の上位1件ずつを確認")
    multi = df.sort_values(["store", "revenue"], ascending=[True, False])
    print(multi.groupby("store").head(1)[["store", "date", "category", "revenue"]]
          .to_string(index=False))

    print("\nまとめ: 業務の質問 = (条件の結合で選び出す) + (ソート/上位N) + (loc で列を選択)。")
    print("      条件ごとにかっこ、and/or の代わりに &/|、先頭の行は iloc[0] — この3つを守るだけでエラーの大部分を予防できます。")


if __name__ == "__main__":
    main()
