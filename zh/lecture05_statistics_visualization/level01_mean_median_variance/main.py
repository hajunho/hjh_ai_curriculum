"""
level01 — 平均数、中位数、方差与标准差

用连锁咖啡店销售数据 (hjh_data) 计算并交叉验证各代表值，
再逐步实验一个极端值 (1 笔大型团体订单) 分别能把
平均数、中位数、标准差拖走多远。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_clean_sales() -> pd.DataFrame:
    """载入销售数据，去除缺失与负数污染值。"""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.dropna(subset=["revenue"])          # 去除缺失
    df = df[df["revenue"] > 0].copy()           # 去除负数污染值
    df["revenue"] = df["revenue"].astype(float)
    print(f"[1] 数据清洗: {before:,}行 -> {len(df):,}行 "
          f"(去除缺失/负数 {before - len(df)}条)")
    return df


def describe_revenue(df: pd.DataFrame) -> None:
    """交叉验证 pandas 的计算值与 numpy 手工计算值。"""
    rev = df["revenue"].to_numpy()
    mean_np = rev.sum() / len(rev)              # 手工算平均数
    var_np = ((rev - mean_np) ** 2).mean()      # 手工算方差(总体方差)
    std_np = var_np ** 0.5

    print()
    print("[2] 全部销售额(每笔)的代表值 — pandas vs numpy 手工计算")
    print(f"    平均数   : {df['revenue'].mean():>14,.0f} 韩元 | 手工计算 {mean_np:>14,.0f} 韩元")
    print(f"    中位数   : {df['revenue'].median():>14,.0f} 韩元")
    print(f"    方差     : {df['revenue'].var(ddof=0):>14,.0f} 韩元^2 | 手工计算 {var_np:>14,.0f} 韩元^2")
    print(f"    标准差   : {df['revenue'].std(ddof=0):>14,.0f} 韩元 | 手工计算 {std_np:>14,.0f} 韩元")
    print("    -> 平均数 > 中位数 : 销售额分布的尾巴拖向大值一侧的信号。")


def by_store(df: pd.DataFrame) -> None:
    """比较各门店的平均数、中位数、标准差与变异系数。"""
    g = df.groupby("store")["revenue"].agg(["mean", "median", "std"])
    g["cv"] = g["std"] / g["mean"]              # 变异系数
    print()
    print("[3] 各门店代表值 (变异系数 = 标准差/平均数)")
    print(f"    {'门店':<6} {'平均数':>12} {'中位数':>12} {'标准差':>12} {'变异系数':>8}")
    for store, row in g.sort_values("mean", ascending=False).iterrows():
        print(f"    {store:<6} {row['mean']:>12,.0f} {row['median']:>12,.0f} "
              f"{row['std']:>12,.0f} {row['cv']:>8.2f}")


def outlier_experiment(df: pd.DataFrame) -> None:
    """实验一个极端值能把 30 条样本的统计量移动多少。"""
    rng = np.random.default_rng(7)              # 固定 seed
    gangnam = df[df["store"] == "朝阳店"]["revenue"].to_numpy()
    sample = rng.choice(gangnam, size=30, replace=False)

    def report(tag: str, values: np.ndarray) -> tuple[float, float, float]:
        m, md, sd = values.mean(), float(np.median(values)), values.std()
        print(f"    {tag:<24} 平均数 {m:>12,.0f} | 中位数 {md:>12,.0f} | 标准差 {sd:>12,.0f}")
        return m, md, sd

    print()
    print("[4] 极端值实验 — 向朝阳店销售额 30 条样本中加入 1 笔大型团体订单")
    base = report("原始样本(30条)", sample)
    outlier = 500_000_000.0                     # 5 亿韩元的团体订单
    spiked = np.append(sample, outlier)
    after = report("加入极端值(31条)", spiked)

    print()
    print("    '一个'值造成的变化:")
    print(f"      平均数   : {after[0] - base[0]:>+14,.0f} 韩元  (被狠狠拖走)")
    print(f"      中位数   : {after[1] - base[1]:>+14,.0f} 韩元  (几乎原地不动 = 稳健)")
    print(f"      标准差   : {after[2] - base[2]:>+14,.0f} 韩元  (平方计算导致爆炸)")
    print("    => 年薪、销售额这类长尾数据的报告里，请同时写上中位数。")


def main() -> None:
    df = load_clean_sales()
    describe_revenue(df)
    by_store(df)
    outlier_experiment(df)


if __name__ == "__main__":
    main()
