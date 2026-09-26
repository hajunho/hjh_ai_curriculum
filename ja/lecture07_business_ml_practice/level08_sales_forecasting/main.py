"""
実践 1 — 売上・需要予測のフルパイプライン。
カフェチェーンの 1 年分の売上を整形・集計し、曜日/季節/ラグ/移動平均の特徴量を作って
最後の 4 週間を時間ベースのバックテストで評価し、「来週」の 7 日間を詳しく見ます。
ベースライン(昨日と同じ / 先週の同じ曜日)と比較し、予測 vs 実際の PNG を保存します。
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BACKTEST_DAYS = 28     # 評価区間: 最後の 4 週間 (7 日だけでは運が大きく混ざる)
FOCUS_DAYS = 7         # 表と図で詳しく見る「来週」
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 66)
    print(" 実践 1: 来週の売上予測パイプライン (sales_table)")
    print("=" * 66)

    # [1] 整形 + 日別集計 -------------------------------------------------
    raw = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n0 = len(raw)
    raw = raw.dropna(subset=["revenue"])
    raw = raw[raw["revenue"] > 0]
    daily = raw.groupby("day_index")["revenue"].sum().reset_index()
    print(f"\n[1] 整形 {n0} -> {len(raw)}行, 日別集計 {len(daily)}日")
    print(f"    1日平均売上 {daily['revenue'].mean():,.0f}ウォン")

    # [2] 時間特徴量を作る (予測時点で知り得る値だけ!) -------------------
    print("\n[2] 特徴量の生成: カレンダー + 季節 + 先週の情報 (7日先の予測なので lag>=7 のみ使用)")
    daily["weekday"] = daily["day_index"] % 7            # 0=月 ... 6=日
    daily["is_weekend"] = (daily["weekday"] >= 5).astype(int)
    daily["season_sin"] = np.sin(2 * np.pi * daily["day_index"] / 365)
    daily["season_cos"] = np.cos(2 * np.pi * daily["day_index"] / 365)
    daily["lag_7"] = daily["revenue"].shift(7)           # 先週の同じ曜日
    daily["ma7_prev"] = daily["revenue"].shift(7).rolling(7).mean()   # 先週基準の直近7日平均
    daily["ma28_prev"] = daily["revenue"].shift(7).rolling(28).mean() # 先週基準の直近4週平均
    for wd in range(7):                                  # 曜日ワンホット
        daily[f"wd_{WEEKDAYS[wd]}"] = (daily["weekday"] == wd).astype(int)
    daily = daily.dropna().reset_index(drop=True)
    features = (["is_weekend", "season_sin", "season_cos", "lag_7", "ma7_prev", "ma28_prev"]
                + [f"wd_{w}" for w in WEEKDAYS])
    print(f"    特徴量 {len(features)}個: 曜日ワンホット 7 + 週末/季節 3 + 先週の lag/移動平均 3")

    # [3] 時間ベースの分割 ----------------------------------------------------
    train = daily.iloc[:-BACKTEST_DAYS]
    test = daily.iloc[-BACKTEST_DAYS:]
    print(f"\n[3] 時間ベースの分割: 学習 {len(train)}日 (day~{train['day_index'].max()}) / "
          f"評価は最後の {BACKTEST_DAYS}日 (無作為分割は禁止!)")
    print("    (毎週予測を更新する運用を想定: 各日付の特徴量は7日前までの情報のみ使用)")

    # [4] ベースライン vs モデル ----------------------------------------------------
    print(f"\n[4] ベースラインとモデルの対決 — 直近 {BACKTEST_DAYS}日のバックテスト (MAE / MAPE)")
    base_naive = daily["revenue"].shift(1).iloc[-BACKTEST_DAYS:].to_numpy()  # ベースライン1: 昨日と同じ
    base_seasonal = test["lag_7"].to_numpy()             # ベースライン2: 先週の同じ曜日

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train[features], train["revenue"])
    pred = model.predict(test[features])

    for name, p in [("ベースライン1 昨日と同じ", base_naive),
                    ("ベースライン2 先週の同じ曜日", base_seasonal),
                    ("RandomForest モデル", pred)]:
        print(f"    {name:<22} MAE {mean_absolute_error(test['revenue'], p):>11,.0f}ウォン | "
              f"MAPE {mape(test['revenue'], p):5.2f}%")
    improve = (1 - mean_absolute_error(test["revenue"], pred)
               / mean_absolute_error(test["revenue"], base_seasonal)) * 100
    print(f"    => モデルは最強のベースライン比で MAE {improve:+.1f}% "
          f"({'改善' if improve > 0 else '悪化'}) — 報告は常にベースライン比で。")

    # [5] 来週の日別予測表 + PNG ----------------------------------------
    print(f"\n[5] 「来週」(最後の {FOCUS_DAYS}日) 予測 vs 実際")
    print(f"    {'day':>4} {'曜日':>4} {'実際':>12} {'予測':>12} {'誤差':>10}")
    focus = test.iloc[-FOCUS_DAYS:]
    focus_pred = pred[-FOCUS_DAYS:]
    for (_, row), p in zip(focus.iterrows(), focus_pred):
        wd = WEEKDAYS[int(row["weekday"])]
        err = p - row["revenue"]
        print(f"    {int(row['day_index']):>4} {wd:>4} {row['revenue']:>12,.0f} "
              f"{p:>12,.0f} {err:>+10,.0f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    recent = daily.iloc[-56:]                            # 直近 8 週間の流れ
    ax.plot(recent["day_index"], recent["revenue"], label="actual", color="#4477aa")
    ax.plot(test["day_index"], pred, "o--", label="model forecast", color="#cc6677",
            markersize=4)
    ax.plot(focus["day_index"], base_seasonal[-FOCUS_DAYS:], "s:",
            label="seasonal naive (last week)", color="#999933", markersize=4)
    ax.axvline(train["day_index"].max() + 0.5, color="gray", ls="--", lw=1)
    ax.text(train["day_index"].max() + 0.7, ax.get_ylim()[1] * 0.97, "backtest start",
            fontsize=8, va="top")
    ax.set_xlabel("day_index")
    ax.set_ylabel("daily revenue (KRW)")
    ax.set_title("Next-week sales forecast vs actual")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "forecast_vs_actual.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"\n    図の保存: {png}")

    # [6] モデルは何を見て予測したのか ---------------------------------------
    print("\n[6] 特徴量重要度 (上位 6 個)")
    imp = sorted(zip(features, model.feature_importances_), key=lambda t: -t[1])[:6]
    for name, v in imp:
        print(f"    {name:<12} {v:.3f} {'#' * int(v * 40)}")
    print("\n    教訓: 需要予測 = 店長の直感(曜日・季節・トレンド)を特徴量に翻訳して")
    print("          毎日自動で、一貫して適用する仕事です。")


if __name__ == "__main__":
    main()
