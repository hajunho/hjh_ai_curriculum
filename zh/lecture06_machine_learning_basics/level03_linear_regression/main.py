"""
level03 — 线性回归: 广告费 -> 销售额

在咖啡连锁店销售数据上，用最小二乘直线求出
'广告费每多 1 万韩元，销售额涨了多少'。同一个答案用两种方法计算并比较。
  (1) numpy 公式: a = Cov(x,y)/Var(x), b = mean(y) - a*mean(x)
  (2) sklearn LinearRegression
散点图 + 回归线保存到 outputs/regression.png。
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")   # 在没有屏幕的环境里也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def load_daily_store_sales() -> pd.DataFrame:
    """[1] 原始数据是 (日 x 门店 x 品类) 粒度 -> 汇总成 (日 x 门店) 粒度。
    缺失的 revenue 和负数污染在汇总前剔除 (lecture03 复习)。"""
    raw = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n_before = len(raw)
    clean = raw.dropna(subset=["revenue"])          # 剔除缺失
    clean = clean[clean["revenue"] > 0]             # 剔除负数(录入错误)
    print(f"    清洗: {n_before} 行 -> {len(clean)} 行 (剔除缺失 {raw['revenue'].isna().sum()} 条、"
          f"负数 {(raw['revenue'].dropna() <= 0).sum()} 条)")
    # ad_cost 在每个 (日, 门店) 只有一个值所以取 first, revenue 是各品类合计
    daily = (clean.groupby(["day_index", "store"], as_index=False)
                  .agg(ad_cost=("ad_cost", "first"), revenue=("revenue", "sum")))
    return daily


if __name__ == "__main__":
    np.random.seed(0)  # 可复现 (本关不用随机数，但按惯例固定)

    # [1] 数据准备 ------------------------------------------------------
    print("[1] 数据准备 — 咖啡连锁 365 天 x 5 家门店的销售额")
    daily = load_daily_store_sales()
    x = daily["ad_cost"].to_numpy(dtype=float)      # 输入: 单日广告费 (韩元)
    y = daily["revenue"].to_numpy(dtype=float)      # 答案: 单日销售额 (韩元)
    print(f"    分析单位: (日, 门店) 共 {len(daily)} 条 / "
          f"广告费范围 {x.min():,.0f}~{x.max():,.0f} 韩元\n")

    # [2] 用 numpy 公式求最小二乘直线 -------------------------------
    slope_np = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)  # a = Cov/Var
    intercept_np = y.mean() - slope_np * x.mean()              # 经过 (x̄,ȳ)
    print("[2] 用 numpy 公式 (Cov/Var) 直接计算")
    print(f"    斜率 a = {slope_np:.4f}   截距 b = {intercept_np:,.0f}")

    # [3] 用 sklearn 解同一个问题 ---------------------------------------
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)   # sklearn 要二维输入 (行=样本, 列=特征)
    slope_sk, intercept_sk = model.coef_[0], model.intercept_
    print("[3] sklearn LinearRegression")
    print(f"    斜率 a = {slope_sk:.4f}   截距 b = {intercept_sk:,.0f}")
    same = np.isclose(slope_np, slope_sk) and np.isclose(intercept_np, intercept_sk)
    print(f"    两种方法结果一致? {same} — 库只是同一条公式的包装。\n")

    # [4] 业务解读 ----------------------------------------------------
    y_hat = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_hat)
    print("[4] 业务解读")
    print(f"    广告费每 1 万韩元与销售额 +{slope_sk * 10_000:,.0f} 韩元的关联 (是相关，不是因果证明!)")
    print(f"    R^2 = {r2:.3f} -> 销售额波动的 {r2:.1%} 可由广告费单独解释")
    ad = 300_000
    pred = model.predict([[ad]])[0]
    print(f"    广告费 {ad:,} 韩元那天的预测销售额: {pred:,.0f} 韩元")
    print("    (注意: 用全量数据既训练又打分，成绩偏乐观 -> level04)\n")

    # [5] 保存散点图 + 回归线 ----------------------------------------------
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
    print(f"[5] 图片已保存: {png}")
    print("    在散点的杂乱(噪声)中，直线总结出了'平均趋势'。")
