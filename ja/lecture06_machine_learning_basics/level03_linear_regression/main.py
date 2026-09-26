"""
level03 — 線形回帰: 広告費 -> 売上

カフェチェーンの売上データから「広告費 1万ウォンあたり売上がどれだけ増えたか」を
最小二乗法の直線で求めます。同じ答えを 2 つの方法で計算して比較します。
  (1) numpy の公式: a = Cov(x,y)/Var(x), b = mean(y) - a*mean(x)
  (2) sklearn LinearRegression
散布図 + 回帰直線を outputs/regression.png に保存します。
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")   # 画面のない環境でも図を保存できるように
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def load_daily_store_sales() -> pd.DataFrame:
    """[1] 元データは (日 x 店舗 x カテゴリ) 単位 -> (日 x 店舗) 単位に集計。
    欠損した revenue と負の値の汚染は、集計前に取り除きます (lecture03 の復習)。"""
    raw = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n_before = len(raw)
    clean = raw.dropna(subset=["revenue"])          # 欠損の除去
    clean = clean[clean["revenue"] > 0]             # 負の値(入力ミス)の除去
    print(f"    クレンジング: {n_before}行 -> {len(clean)}行 (欠損 {raw['revenue'].isna().sum()}件, "
          f"負の値 {(raw['revenue'].dropna() <= 0).sum()}件を除去)")
    # ad_cost は (日, 店舗) ごとに 1 つの値なので first、revenue はカテゴリの合計
    daily = (clean.groupby(["day_index", "store"], as_index=False)
                  .agg(ad_cost=("ad_cost", "first"), revenue=("revenue", "sum")))
    return daily


if __name__ == "__main__":
    np.random.seed(0)  # 再現性 (このレベルは乱数を使いませんが慣例として固定)

    # [1] データの準備 ------------------------------------------------------
    print("[1] データの準備 — カフェチェーン 365日 x 5店舗の売上")
    daily = load_daily_store_sales()
    x = daily["ad_cost"].to_numpy(dtype=float)      # 入力: 1日の広告費 (ウォン)
    y = daily["revenue"].to_numpy(dtype=float)      # 正解: 1日の売上 (ウォン)
    print(f"    分析単位: (日, 店舗) {len(daily)}件 / "
          f"広告費の範囲 {x.min():,.0f}〜{x.max():,.0f}ウォン\n")

    # [2] numpy の公式で最小二乗直線を求める -------------------------------
    slope_np = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)  # a = Cov/Var
    intercept_np = y.mean() - slope_np * x.mean()              # (x̄,ȳ) を通る
    print("[2] numpy の公式 (Cov/Var) で直接計算")
    print(f"    傾き a = {slope_np:.4f}   切片 b = {intercept_np:,.0f}")

    # [3] sklearn で同じ問題を解く ---------------------------------------
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)   # sklearn は 2 次元入力 (行=サンプル, 列=特徴量)
    slope_sk, intercept_sk = model.coef_[0], model.intercept_
    print("[3] sklearn LinearRegression")
    print(f"    傾き a = {slope_sk:.4f}   切片 b = {intercept_sk:,.0f}")
    same = np.isclose(slope_np, slope_sk) and np.isclose(intercept_np, intercept_sk)
    print(f"    2 つの方法の結果は一致? {same} — ライブラリは同じ公式の包み紙です。\n")

    # [4] ビジネス解釈 ----------------------------------------------------
    y_hat = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_hat)
    print("[4] ビジネス解釈")
    print(f"    広告費 1万ウォンあたり売上 +{slope_sk * 10_000:,.0f}ウォンの関連 (相関であって因果の証明ではない!)")
    print(f"    R^2 = {r2:.3f} -> 売上の変動の {r2:.1%} を広告費 1 つで説明")
    ad = 300_000
    pred = model.predict([[ad]])[0]
    print(f"    広告費 {ad:,}ウォンの日の予測売上: {pred:,.0f}ウォン")
    print("    (注意: 全データで学習して採点した楽観的な成績 -> level04)\n")

    # [5] 散布図 + 回帰直線の保存 ----------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x / 10_000, y / 10_000, s=8, alpha=0.3, label="daily data")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs / 10_000, (slope_sk * xs + intercept_sk) / 10_000,
            color="crimson", linewidth=2,
            label=f"y = {slope_sk:.2f}x + {intercept_sk/10_000:,.0f}")
    ax.set_xlabel("ad cost (10k KRW)")
    ax.set_ylabel("daily revenue (10k KRW)")
    ax.set_title("Ad cost vs daily revenue (least squares fit)")
    ax.legend()
    fig.tight_layout()
    png = OUT_DIR / "regression.png"
    fig.savefig(png, dpi=120)
    print(f"[5] 図を保存: {png}")
    print("    散布図のばらつき(ノイズ)の中で、直線が「平均的な傾向」を要約します。")
