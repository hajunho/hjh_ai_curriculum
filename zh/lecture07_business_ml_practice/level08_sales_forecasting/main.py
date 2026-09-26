"""
实战 1 — 销售需求预测的完整流水线。
把咖啡连锁店一年的销售额清洗·汇总, 造出星期/季节/滞后/移动平均特征,
用最后 4 周做基于时间的回测评估, 再细看'下一周'的 7 天。
和基准线(和昨天一样 / 上周同一天)比较, 并保存预测 vs 实际的 PNG。
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
BACKTEST_DAYS = 28     # 评估区间: 最后 4 周 (只用 7 天的话运气成分太多)
FOCUS_DAYS = 7         # 在表和图里细看的'下一周'
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 66)
    print(" 实战 1: 下一周销售额预测流水线 (sales_table)")
    print("=" * 66)

    # [1] 清洗 + 按日汇总 ---------------------------------------------------
    raw = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n0 = len(raw)
    raw = raw.dropna(subset=["revenue"])
    raw = raw[raw["revenue"] > 0]
    daily = raw.groupby("day_index")["revenue"].sum().reset_index()
    print(f"\n[1] 清洗 {n0} -> {len(raw)}行, 按日汇总 {len(daily)}天")
    print(f"    日均销售额 {daily['revenue'].mean():,.0f}韩元")

    # [2] 造时间特征 (只用预测时点就能知道的值!) -------------------
    print("\n[2] 特征生成: 日历 + 季节 + 上周信息 (提前 7 天预测, 所以只用 lag>=7)")
    daily["weekday"] = daily["day_index"] % 7            # 0=周一 ... 6=周日
    daily["is_weekend"] = (daily["weekday"] >= 5).astype(int)
    daily["season_sin"] = np.sin(2 * np.pi * daily["day_index"] / 365)
    daily["season_cos"] = np.cos(2 * np.pi * daily["day_index"] / 365)
    daily["lag_7"] = daily["revenue"].shift(7)           # 上周的同一天
    daily["ma7_prev"] = daily["revenue"].shift(7).rolling(7).mean()   # 以上周为基准的近 7 天均值
    daily["ma28_prev"] = daily["revenue"].shift(7).rolling(28).mean() # 以上周为基准的近 4 周均值
    for wd in range(7):                                  # 星期独热
        daily[f"wd_{WEEKDAYS[wd]}"] = (daily["weekday"] == wd).astype(int)
    daily = daily.dropna().reset_index(drop=True)
    features = (["is_weekend", "season_sin", "season_cos", "lag_7", "ma7_prev", "ma28_prev"]
                + [f"wd_{w}" for w in WEEKDAYS])
    print(f"    特征 {len(features)}个: 星期独热 7 + 周末/季节 3 + 上周 lag/移动平均 3")

    # [3] 基于时间的划分 ----------------------------------------------------
    train = daily.iloc[:-BACKTEST_DAYS]
    test = daily.iloc[-BACKTEST_DAYS:]
    print(f"\n[3] 基于时间的划分: 训练 {len(train)}天 (day~{train['day_index'].max()}) / "
          f"评估最后 {BACKTEST_DAYS}天 (禁止随机划分!)")
    print("    (假设每周更新一次预测的运营方式: 每天的特征只用到 7 天前为止的信息)")

    # [4] 基准线 vs 模型 ----------------------------------------------------
    print(f"\n[4] 基准线与模型的对决 — 最近 {BACKTEST_DAYS}天回测 (MAE / MAPE)")
    base_naive = daily["revenue"].shift(1).iloc[-BACKTEST_DAYS:].to_numpy()  # 基准线1: 和昨天一样
    base_seasonal = test["lag_7"].to_numpy()             # 基准线2: 和上周同一天一样

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train[features], train["revenue"])
    pred = model.predict(test[features])

    for name, p in [("基准线1 和昨天一样", base_naive),
                    ("基准线2 和上周同一天一样", base_seasonal),
                    ("RandomForest 模型", pred)]:
        print(f"    {name:<22} MAE {mean_absolute_error(test['revenue'], p):>11,.0f}韩元 | "
              f"MAPE {mape(test['revenue'], p):5.2f}%")
    improve = (1 - mean_absolute_error(test["revenue"], pred)
               / mean_absolute_error(test["revenue"], base_seasonal)) * 100
    print(f"    => 模型相对最强基准线 MAE {improve:+.1f}% "
          f"({'改善' if improve > 0 else '变差'}) — 汇报永远要相对基准线来说。")

    # [5] 下一周的日别预测表 + PNG ----------------------------------------
    print(f"\n[5] '下一周'(最后 {FOCUS_DAYS}天) 预测 vs 实际")
    print(f"    {'day':>4} {'星期':>4} {'实际':>12} {'预测':>12} {'误差':>10}")
    focus = test.iloc[-FOCUS_DAYS:]
    focus_pred = pred[-FOCUS_DAYS:]
    for (_, row), p in zip(focus.iterrows(), focus_pred):
        wd = WEEKDAYS[int(row["weekday"])]
        err = p - row["revenue"]
        print(f"    {int(row['day_index']):>4} {wd:>4} {row['revenue']:>12,.0f} "
              f"{p:>12,.0f} {err:>+10,.0f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    recent = daily.iloc[-56:]                            # 最近 8 周的走势
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
    print(f"\n    图片已保存: {png}")

    # [6] 模型是看着什么做预测的 ---------------------------------------
    print("\n[6] 特征重要性 (前 6 名)")
    imp = sorted(zip(features, model.feature_importances_), key=lambda t: -t[1])[:6]
    for name, v in imp:
        print(f"    {name:<12} {v:.3f} {'#' * int(v * 40)}")
    print("\n    教训: 需求预测 = 把店长的直觉(星期·季节·趋势)翻译成特征,")
    print("          再让它每天自动、一致地跑起来。")


if __name__ == "__main__":
    main()
