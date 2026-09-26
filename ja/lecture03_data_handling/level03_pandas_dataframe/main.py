"""
Pandas の Series と DataFrame の構造を解剖する実習です。
カフェチェーンの90日分の売上 (hjh_data.sales_table) を DataFrame にして
shape / index / columns / dtypes / head / info / describe を順に確認し、
列を1本 Series として取り出し、派生列を作り、欠損の数まで診断します。
"""

import io
import pathlib
import sys

import pandas as pd

# 共用データモジュール (common/hjh_data.py) を読み込めるようにパスを追加します。
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 110)
    pd.set_option("display.max_columns", 10)

    # ------------------------------------------------------------------
    print("[1] 辞書のリスト -> DataFrame")
    rows = hjh_data.sales_table(n_days=90, seed=42)   # seed 固定: 常に同じデータ
    print(f"    元: Python の辞書のリスト、{len(rows)}個 (90日 x 5店舗 x 5カテゴリ)")
    df = pd.DataFrame(rows)
    print(f"    変換: pd.DataFrame(rows) の1行 -> 型 {type(df).__name__}")

    # ------------------------------------------------------------------
    print("\n[2] 表の3要素: 値 / 行の名札 (index) / 列の名前 (columns)")
    print(f"    shape   : {df.shape}  (行 {df.shape[0]}個, 列 {df.shape[1]}個)")
    print(f"    index   : {df.index}")
    print(f"    columns : {list(df.columns)}")

    # ------------------------------------------------------------------
    print("\n[3] 列ごとに1つずつ持つデータ型 (dtype)")
    print(df.dtypes.to_string())
    print("    -> revenue が float64 である理由: 欠損値 (NaN) が混ざると整数の列も実数になります。")

    # ------------------------------------------------------------------
    print("\n[4] head() — 先頭5行のプレビュー (Excel でスクロールの一番上を見るのと同じです)")
    print(df.head().to_string())

    # ------------------------------------------------------------------
    print("\n[5] info() — データの健康診断 (列ごとの欠損でない値の数 + dtype + メモリ)")
    buf = io.StringIO()                 # info() は戻り値がなく出力するだけなので、バッファで受けます。
    df.info(buf=buf)
    print(buf.getvalue())

    # ------------------------------------------------------------------
    print("[6] describe() — 数値列の要約統計")
    print(df[["ad_cost", "revenue"]].describe().round(1).to_string())
    print("    -> revenue の min がマイナス! 外れ値が混ざっているシグナルです (level06 で処理)。")

    # ------------------------------------------------------------------
    print("\n[7] 列を1本取り出すと Series — 名札 (index) 付きの1次元の値のまとまり")
    revenue = df["revenue"]
    print(f"    type(df['revenue']) = {type(revenue).__name__}")
    print(f"    長さ {len(revenue)}, dtype {revenue.dtype}")
    print(f"    平均 {revenue.mean():,.0f}ウォン / 最大 {revenue.max():,.0f}ウォン / 最小 {revenue.min():,.0f}ウォン")
    print("    先頭3個の値 (左の数字が index、右が値):")
    print(revenue.head(3).to_string())

    # ------------------------------------------------------------------
    print("\n[8] 新しい列を作る — 広告費に対する売上の比率 (roas)")
    df["roas"] = df["revenue"] / df["ad_cost"]        # Excel の「数式を入れてドラッグ」が1行
    print(df[["store", "category", "ad_cost", "revenue", "roas"]].head(3).round(2).to_string())
    print(f"    roas の平均: {df['roas'].mean():.2f} (広告費1ウォンあたりの売上ウォン)")

    # ------------------------------------------------------------------
    print("\n[9] value_counts() — カテゴリ列の構成比 (Excel の COUNTIF 繰り返しを1行で)")
    print("    店舗別の行数:")
    print(df["store"].value_counts().to_string())
    print("    曜日別の行数 (90日は7で割り切れないため、曜日ごとに日数が違います):")
    print(df["weekday"].value_counts().to_string())

    # ------------------------------------------------------------------
    print("\n[10] 欠損値の数の診断 — isna().sum()")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())
    ratio = df["revenue"].isna().mean() * 100
    print(f"    -> revenue の欠損比率 {ratio:.2f}%。処理戦略は level06 で学びます。")

    print("\nまとめ: DataFrame = index + columns + 値。列を1本取り出すと Series。")
    print("新しいデータを受け取ったら shape -> head -> info -> describe の順に挨拶しましょう。")


if __name__ == "__main__":
    main()
