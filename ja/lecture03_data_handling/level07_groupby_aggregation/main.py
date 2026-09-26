"""
groupby 集計の実習。
1年分のカフェチェーンの売上をクレンジングした後、split-apply-combine パターンで
店舗別・曜日別・カテゴリ別の分析レポートを作ります。
agg (複数集計) と transform (元の行数を維持) の違い、
曜日の並べ替え (reindex) まで、実務のレポート作成の流れそのままに練習します。
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
WEEKDAY_ORDER = ["月", "火", "水", "木", "金", "土", "日"]


def load_clean_sales() -> pd.DataFrame:
    """売上データを読み込んで level06 の方式でクレンジングします (マイナス->NaN、欠損は削除)。"""
    rows = hjh_data.sales_table(n_days=365, seed=42)  # seed 固定
    df = pd.DataFrame(rows)
    df.loc[df["revenue"] < 0, "revenue"] = np.nan  # ドメインルール: 売上はマイナスになり得ない
    df = df.dropna(subset=["revenue"])             # このレベルは集計が目的なので単純に削除
    return df


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_clean_sales()
    print(f"クレンジング完了: {len(df):,}行 (マイナスを除去 + 欠損を削除)\n")

    # ------------------------------------------------------------------
    print("[1] 店舗別の要約 — agg で総売上・平均・件数を一度に")
    store_report = (
        df.groupby("store")
          .agg(total=("revenue", "sum"),
               avg=("revenue", "mean"),
               n=("revenue", "count"))
          .sort_values("total", ascending=False)
    )
    store_report["total_mil"] = (store_report["total"] / 1e6).round(1)  # 百万ウォン単位
    print(store_report[["total_mil", "avg", "n"]].round(0).to_string())
    print("  -> 福岡・名古屋のほうが大きい理由: データに店舗規模の効果が仕込まれています。")

    # ------------------------------------------------------------------
    print("\n[2] 曜日別の平均売上 — 文字コード順の罠を reindex で解決")
    by_weekday = df.groupby("weekday")["revenue"].mean().reindex(WEEKDAY_ORDER)
    print(by_weekday.round(0).to_string())
    weekend_avg = by_weekday[["土", "日"]].mean()
    weekday_avg = by_weekday[["月", "火", "水", "木", "金"]].mean()
    print(f"  週末の平均 {weekend_avg:,.0f} / 平日の平均 {weekday_avg:,.0f}"
          f" -> 週末倍率 {weekend_avg / weekday_avg:.2f}倍")
    print("  -> データに仕込まれた週末効果 (約1.2倍) を集計で復元しました。")

    # ------------------------------------------------------------------
    print("\n[3] カテゴリ別の総売上と比率(%)")
    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    share = (by_cat / by_cat.sum() * 100).round(1)
    cat_table = pd.DataFrame({"総売上(百万ウォン)": (by_cat / 1e6).round(1), "比率(%)": share})
    print(cat_table.to_string())

    # ------------------------------------------------------------------
    print("\n[4] 店舗 x カテゴリ — 複数キーの groupby、上位5つの組み合わせ")
    combo = (
        df.groupby(["store", "category"])["revenue"].sum()
          .sort_values(ascending=False)
          .head(5)
    )
    print((combo / 1e6).round(1).to_string())
    print("  -> MultiIndex の結果です。表として使うなら reset_index() を付けてください。")

    # ------------------------------------------------------------------
    print("\n[5] transform — 「店舗内の売上比率」という派生列を作る")
    store_total = df.groupby("store")["revenue"].transform("sum")  # 行数を維持!
    df = df.assign(share_in_store=df["revenue"] / store_total)
    sample = df.loc[df["store"] == "渋谷店",
                    ["date", "category", "revenue", "share_in_store"]].head(3)
    print(sample.to_string(index=False))
    check = df.groupby("store")["share_in_store"].sum().round(6)
    print(f"  検証: 店舗別の比率の合計 = {[float(v) for v in check.unique()]} (すべて 1.0 なら正常)")

    # ------------------------------------------------------------------
    print("\n[6] 最終レポートの保存")
    csv_path = os.path.join(OUT_DIR, "store_report.csv")
    store_report.reset_index().to_csv(csv_path, index=False, encoding="utf-8-sig")
    top_store = store_report.index[0]
    top_cat = by_cat.index[0]
    print(f"  店舗別の要約表を保存: {csv_path}")
    print("  ---- 一行要約レポート ----")
    print(f"  売上トップの店舗: {top_store} ({store_report.loc[top_store, 'total_mil']:,}百万ウォン)")
    print(f"  売上トップのカテゴリ: {top_cat} (比率 {share[top_cat]}%)")
    print(f"  週末効果: 平日比 {weekend_avg / weekday_avg:.2f}倍")
    print("\nまとめ: 「~別に見ると?」という質問は、すべて groupby 一行に変わります。")


if __name__ == "__main__":
    main()
