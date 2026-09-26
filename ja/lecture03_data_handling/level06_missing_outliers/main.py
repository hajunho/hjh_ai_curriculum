"""
欠損値・外れ値の処理の実習。
カフェチェーンの売上データの汚染 (欠損、マイナス) を診断し、
ドメインルールと IQR で異常値を見つけた後、
dropna / 全体の中央値で代替 / グループ別の中央値で代替の3戦略を
統計量で比較して、どの戦略がいつ適切かを確認します。
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 画面のない環境でも図を保存できるように
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def diagnose(df: pd.DataFrame) -> None:
    """汚染の状態を診断して出力します。"""
    n = len(df)
    n_missing = int(df["revenue"].isna().sum())
    n_negative = int((df["revenue"] < 0).sum())  # NaN は比較から自動的に除外される
    print(f"  全体の行数        : {n:,}")
    print(f"  欠損 (NaN) の数   : {n_missing:,}件 ({n_missing / n:.2%})")
    print(f"  マイナス売上の数  : {n_negative:,}件 ({n_negative / n:.2%})")
    desc = df["revenue"].describe()
    print(f"  describe() の要約 : min={desc['min']:,.0f}  "
          f"median={desc['50%']:,.0f}  max={desc['max']:,.0f}")
    if desc["min"] < 0:
        print("  -> min がマイナス! 売上データとして辻褄の合わない値が混ざっています。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    print("[1] 汚染の診断 — 数えてみるまでは何も分かりません")
    rows = hjh_data.sales_table(n_days=365, seed=42)  # seed 固定: 常に同じデータ
    df = pd.DataFrame(rows)
    diagnose(df)

    # ------------------------------------------------------------------
    print("\n[2] ドメインルールの適用 — 「売上はマイナスになり得ない」")
    df_rule = df.copy()  # 原本は証拠物件として保存
    neg_mask = df_rule["revenue"] < 0
    print(f"  マイナス {int(neg_mask.sum())}件を削除せず、NaN (値不明) と印を付けます。")
    df_rule.loc[neg_mask, "revenue"] = np.nan
    print(f"  処理後の欠損の数 : {int(df_rule['revenue'].isna().sum()):,}件"
          f" (元の欠損 + マイナスだった値)")

    # ------------------------------------------------------------------
    print("\n[3] IQR 法 — 統計的に怪しい値を探す")
    rev = df_rule["revenue"].dropna()
    q1, q3 = rev.quantile(0.25), rev.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((rev < low) | (rev > high)).sum())
    print(f"  Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}")
    print(f"  正常範囲: [{low:,.0f}, {high:,.0f}]")
    print(f"  範囲外の外れ値候補: {n_out}件 ({n_out / len(rev):.2%})")
    print("  -> 週末・繁忙期の売上が混ざっている可能性があるので「候補」にすぎず、自動削除は禁物です。")

    # ------------------------------------------------------------------
    print("\n[4] 処理戦略の比較 — 同じデータ、違う数字")
    # (a) 欠損行の削除
    df_a = df_rule.dropna(subset=["revenue"])
    # (b) 全体の中央値で代替
    overall_median = df_rule["revenue"].median()
    df_b = df_rule.copy()
    df_b["revenue"] = df_b["revenue"].fillna(overall_median)
    # (c) 店舗・カテゴリ別の中央値で代替 (transform は行数を維持する)
    group_median = df_rule.groupby(["store", "category"])["revenue"].transform("median")
    df_c = df_rule.copy()
    df_c["revenue"] = df_c["revenue"].fillna(group_median)

    report = pd.DataFrame({
        "行数": [len(df_a), len(df_b), len(df_c)],
        "総売上(百万ウォン)": [df_a["revenue"].sum() / 1e6,
                       df_b["revenue"].sum() / 1e6,
                       df_c["revenue"].sum() / 1e6],
        "平均売上": [df_a["revenue"].mean(),
                   df_b["revenue"].mean(),
                   df_c["revenue"].mean()],
    }, index=["(a) dropna", "(b) 全体の中央値", "(c) グループ別の中央値"])
    print(report.round(0).to_string())
    print("  解釈: (a) は行が減り、総売上が最も小さく出ます。合計のレポートなら危険!")
    print("        (b) と (c) は行を保存しますが、(c) が店舗・品目の特性を反映してより精巧です。")

    # ------------------------------------------------------------------
    print("\n[5] 処理前後の要約 + 箱ひげ図の保存")
    before = df["revenue"].describe()
    after = df_c["revenue"].describe()
    summary = pd.DataFrame({"処理前": before, "処理後(c)": after}).round(0)
    print(summary.loc[["count", "mean", "min", "50%", "max"]].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].boxplot(df["revenue"].dropna())
    axes[0].set_title("before (with negatives)")
    axes[1].boxplot(df_c["revenue"])
    axes[1].set_title("after (rule + group median)")
    for ax in axes:
        ax.set_ylabel("revenue (KRW)")
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "boxplot_before_after.png")
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    print(f"  箱ひげ図の保存: {png_path}")
    print("\nまとめ: 診断 -> 原因の推定 -> ルールの適用 -> 戦略の比較。順序が品質を作ります。")


if __name__ == "__main__":
    main()
