"""
level01 — 平均・中央値・分散・標準偏差

カフェチェーンの売上データ (hjh_data) で代表値を計算・クロスチェックし、
極端な値 (大型の団体注文 1 件) が平均・中央値・標準偏差をそれぞれどれだけ
引きずるかを段階的に実験します。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_clean_sales() -> pd.DataFrame:
    """売上データを読み込み、欠損・負の汚染値を取り除きます。"""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.dropna(subset=["revenue"])          # 欠損を除去
    df = df[df["revenue"] > 0].copy()           # 負の汚染値を除去
    df["revenue"] = df["revenue"].astype(float)
    print(f"[1] データクレンジング: {before:,}行 -> {len(df):,}行 "
          f"(欠損/負の値 {before - len(df)}件を除去)")
    return df


def describe_revenue(df: pd.DataFrame) -> None:
    """pandas の計算値と numpy の手計算値をクロスチェックします。"""
    rev = df["revenue"].to_numpy()
    mean_np = rev.sum() / len(rev)              # 平均を手で
    var_np = ((rev - mean_np) ** 2).mean()      # 分散 (母分散) を手で
    std_np = var_np ** 0.5

    print()
    print("[2] 全売上 (1 件あたり) の代表値 — pandas vs numpy 手計算")
    print(f"    平均     : {df['revenue'].mean():>14,.0f} ウォン | 手計算 {mean_np:>14,.0f} ウォン")
    print(f"    中央値   : {df['revenue'].median():>14,.0f} ウォン")
    print(f"    分散     : {df['revenue'].var(ddof=0):>14,.0f} ウォン^2 | 手計算 {var_np:>14,.0f} ウォン^2")
    print(f"    標準偏差 : {df['revenue'].std(ddof=0):>14,.0f} ウォン | 手計算 {std_np:>14,.0f} ウォン")
    print("    -> 平均 > 中央値 : 売上分布の裾が大きい値の側に長いというサインです。")


def by_store(df: pd.DataFrame) -> None:
    """店舗ごとの平均・中央値・標準偏差・変動係数を比較します。"""
    g = df.groupby("store")["revenue"].agg(["mean", "median", "std"])
    g["cv"] = g["std"] / g["mean"]              # 変動係数
    print()
    print("[3] 店舗ごとの代表値 (変動係数 = 標準偏差/平均)")
    print(f"    {'店舗':<6} {'平均':>12} {'中央値':>12} {'標準偏差':>12} {'変動係数':>8}")
    for store, row in g.sort_values("mean", ascending=False).iterrows():
        print(f"    {store:<6} {row['mean']:>12,.0f} {row['median']:>12,.0f} "
              f"{row['std']:>12,.0f} {row['cv']:>8.2f}")


def outlier_experiment(df: pd.DataFrame) -> None:
    """極端な値 1 件が標本 30 件の統計をどれだけ動かすかを実験します。"""
    rng = np.random.default_rng(7)              # seed 固定
    gangnam = df[df["store"] == "渋谷店"]["revenue"].to_numpy()
    sample = rng.choice(gangnam, size=30, replace=False)

    def report(tag: str, values: np.ndarray) -> tuple[float, float, float]:
        m, md, sd = values.mean(), float(np.median(values)), values.std()
        print(f"    {tag:<24} 平均 {m:>12,.0f} | 中央値 {md:>12,.0f} | 標準偏差 {sd:>12,.0f}")
        return m, md, sd

    print()
    print("[4] 極端値の実験 — 渋谷店の売上 30 件の標本に大型の団体注文 1 件を追加")
    base = report("元の標本(30件)", sample)
    outlier = 500_000_000.0                     # 5 億ウォンの団体注文
    spiked = np.append(sample, outlier)
    after = report("極端値を追加(31件)", spiked)

    print()
    print("    値「1 つ」が生んだ変化:")
    print(f"      平均     : {after[0] - base[0]:>+14,.0f} ウォン  (大きく引きずられる)")
    print(f"      中央値   : {after[1] - base[1]:>+14,.0f} ウォン  (ほぼその場 = 頑健)")
    print(f"      標準偏差 : {after[2] - base[2]:>+14,.0f} ウォン  (2 乗計算のため爆発)")
    print("    => 年収・売上のように裾の長いデータの報告書には、中央値を併記しましょう。")


def main() -> None:
    df = load_clean_sales()
    describe_revenue(df)
    by_store(df)
    outlier_experiment(df)


if __name__ == "__main__":
    main()
