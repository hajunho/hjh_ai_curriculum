"""
ピボット・リシェイプ・ウィンドウ演算の実習。
Excel のピボットテーブルを pandas で再現します。
  - pivot_table (index/columns/values/aggfunc/margins)
  - melt で wide -> long に戻す
  - rank / pct_change / cumsum のウィンドウ演算とグループ内順位
最終のピボットレポートは outputs/ に CSV で保存します。
"""

import os
import pathlib
import sys

import pandas as pd

# 共通データモジュール (hjh_data) を読み込むためのパス設定
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
WEEKDAY_ORDER = ["月", "火", "水", "木", "金", "土", "日"]


def load_clean_sales() -> pd.DataFrame:
    """売上データを生成した後、欠損とマイナスを除去して本物の日付を付けます。"""
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    df = df.dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    # date の文字列は汚染されているので (level09 参照)、day_index から日付を作ります。
    df["real_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["day_index"], unit="D")
    print(f"    クレンジング後 {len(df):,}行 (long フォーマット: 1行 = ある日・ある店舗・あるカテゴリの売上1件)")
    return df


def step1_store_category_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """[1] 店舗 x カテゴリの年間総売上ピボット (margins=総合計を含む)。"""
    print("\n[1] 店舗 x カテゴリの総売上ピボット — Excel ピボットテーブルの pandas 版")
    pivot = pd.pivot_table(df, index="store", columns="category",
                           values="revenue", aggfunc="sum",
                           margins=True, margins_name="合計")
    print((pivot / 1e8).round(2).to_string())
    print("    (単位: 億ウォン、margins=True で「合計」の行・列が自動追加されます)")
    return pivot


def step2_weekday_pivot(df: pd.DataFrame) -> None:
    """[2] 曜日 x 店舗の平均売上ピボット — aggfunc を変えるだけで別の質問に答えます。"""
    print("\n[2] 曜日 x 店舗の「平均」売上ピボット (aggfunc='mean')")
    pivot = pd.pivot_table(df, index="weekday", columns="store",
                           values="revenue", aggfunc="mean")
    pivot = pivot.reindex(WEEKDAY_ORDER)  # 文字コード順 -> 月〜日の業務の順序に復元
    print((pivot / 1e4).round(0).astype(int).to_string())
    print("    (単位: 万ウォン — 土・日の行が平日より高い「週末効果」を確認してください)")


def step3_melt(df: pd.DataFrame) -> None:
    """[3] melt: wide のピボットを long フォーマットに戻します。"""
    print("\n[3] melt — たんす(wide)をハンガー(long)に戻す")
    wide = pd.pivot_table(df, index="store", columns="category",
                          values="revenue", aggfunc="sum")  # margins なしの版
    long = wide.reset_index().melt(id_vars="store",
                                   var_name="category", value_name="revenue")
    print(f"    wide: {wide.shape[0]}行 x {wide.shape[1]}列 (人が読みやすい)")
    print(f"    long: {long.shape[0]}行 x {long.shape[1]}列 (機械が扱いやすい)")
    print("    long の先頭3行:")
    print(long.head(3).to_string(index=False))
    print("    -> 分析・グラフ・結合は long、レポートの表は wide が定石です。")


def step4_rankings(df: pd.DataFrame) -> None:
    """[4] rank: 全社順位とカテゴリのグループ内順位。"""
    print("\n[4] 順位の分析 — rank と groupby+rank")
    store_total = df.groupby("store")["revenue"].sum()
    ranking = store_total.rank(ascending=False).astype(int).sort_values()
    print("    店舗別総売上の全社順位:")
    for store, r in ranking.items():
        print(f"      {r}位 {store} ({store_total[store] / 1e8:.1f}億)")

    # カテゴリのグループ内での店舗順位: 「コーヒー部門で渋谷店は何位か?」
    cat_store = df.groupby(["category", "store"], as_index=False)["revenue"].sum()
    cat_store["rank_in_category"] = (cat_store.groupby("category")["revenue"]
                                     .rank(ascending=False).astype(int))
    coffee = cat_store[cat_store["category"] == "コーヒー"].sort_values("rank_in_category")
    print("    「コーヒー」カテゴリのグループ内順位:")
    for _, row in coffee.iterrows():
        print(f"      {row['rank_in_category']}位 {row['store']} ({row['revenue'] / 1e8:.1f}億)")


def step5_monthly_flow(df: pd.DataFrame) -> None:
    """[5] 月別売上に pct_change (前月比) と cumsum (累計) を付けます。"""
    print("\n[5] 月別の流れ — pct_change と cumsum")
    monthly = (df.groupby(df["real_date"].dt.to_period("M"))["revenue"].sum())
    report = pd.DataFrame({
        "revenue_100m": (monthly / 1e8).round(2),                 # 月の売上(億)
        "mom_pct": (monthly.pct_change() * 100).round(1),         # 前月比 %
        "cum_100m": (monthly.cumsum() / 1e8).round(1),            # 累計売上(億)
    })
    print(report.to_string())
    print("    (mom_pct の最初の行の NaN は「比べる前月がない」という正直な印です)")


def step6_save_report(pivot: pd.DataFrame) -> None:
    """[6] 完成したピボットレポートを CSV で保存します。"""
    print("\n[6] レポートの保存")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = OUT_DIR / "pivot_report.csv"
    (pivot / 1e8).round(3).to_csv(out_path, encoding="utf-8-sig")  # Excel 互換のエンコーディング
    print(f"    保存完了: {out_path}")
    print("    Excel で開くと、手で作っていたピボット表がコード1回で再現されたことが分かります。")


def main() -> None:
    print("=" * 60)
    print("Level 10 — ピボット・リシェイプ・ウィンドウ演算")
    print("=" * 60)
    df = load_clean_sales()
    pivot = step1_store_category_pivot(df)
    step2_weekday_pivot(df)
    step3_melt(df)
    step4_rankings(df)
    step5_monthly_flow(df)
    step6_save_report(pivot)
    print("\n完了! 同じ pivot_table で aggfunc だけを変えながら、別の質問に答えてみてください。")


if __name__ == "__main__":
    main()
