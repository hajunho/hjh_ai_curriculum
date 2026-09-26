"""
解剖 Pandas Series 与 DataFrame 结构的实战。
把咖啡连锁店 90 天的销售数据 (hjh_data.sales_table) 做成 DataFrame，
依次确认 shape / index / columns / dtypes / head / info / describe，
再把一列抽成 Series、造一个派生列，最后诊断缺失个数。
"""

import io
import pathlib
import sys

import pandas as pd

# 添加路径，以便导入公用数据模块 (common/hjh_data.py)。
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 110)
    pd.set_option("display.max_columns", 10)

    # ------------------------------------------------------------------
    print("[1] 字典列表 -> DataFrame")
    rows = hjh_data.sales_table(n_days=90, seed=42)   # 固定 seed: 数据永远一样
    print(f"    原始: Python 字典列表，{len(rows)} 个 (90天 x 5门店 x 5品类)")
    df = pd.DataFrame(rows)
    print(f"    转换: pd.DataFrame(rows) 一行 -> 类型 {type(df).__name__}")

    # ------------------------------------------------------------------
    print("\n[2] 表的三要素: 值 / 行标签(index) / 列名(columns)")
    print(f"    shape   : {df.shape}  (行 {df.shape[0]} 个，列 {df.shape[1]} 个)")
    print(f"    index   : {df.index}")
    print(f"    columns : {list(df.columns)}")

    # ------------------------------------------------------------------
    print("\n[3] 每列各自持有一个数据类型(dtype)")
    print(df.dtypes.to_string())
    print("    -> revenue 之所以是 float64: 一旦混进缺失值(NaN)，整数列也会变成浮点。")

    # ------------------------------------------------------------------
    print("\n[4] head() — 预览前 5 行 (相当于在 Excel 里把滚动条拉到最上面)")
    print(df.head().to_string())

    # ------------------------------------------------------------------
    print("\n[5] info() — 数据体检 (各列的非缺失值个数 + dtype + 内存)")
    buf = io.StringIO()                 # info() 没有返回值、只会打印，所以用缓冲区接住。
    df.info(buf=buf)
    print(buf.getvalue())

    # ------------------------------------------------------------------
    print("[6] describe() — 数值列的摘要统计")
    print(df[["ad_cost", "revenue"]].describe().round(1).to_string())
    print("    -> revenue 的 min 是负数! 这是混进了异常值的信号 (level06 处理)。")

    # ------------------------------------------------------------------
    print("\n[7] 抽出一列就是 Series — 带标签(index)的一维值集合")
    revenue = df["revenue"]
    print(f"    type(df['revenue']) = {type(revenue).__name__}")
    print(f"    长度 {len(revenue)}，dtype {revenue.dtype}")
    print(f"    平均 {revenue.mean():,.0f} 韩元 / 最大 {revenue.max():,.0f} 韩元 / 最小 {revenue.min():,.0f} 韩元")
    print("    前 3 个值 (左边的数字是 index，右边是值):")
    print(revenue.head(3).to_string())

    # ------------------------------------------------------------------
    print("\n[8] 造一个新列 — 广告费产出销售额比(roas)")
    df["roas"] = df["revenue"] / df["ad_cost"]        # Excel 的"输公式再往下拖"变成一行
    print(df[["store", "category", "ad_cost", "revenue", "roas"]].head(3).round(2).to_string())
    print(f"    roas 平均: {df['roas'].mean():.2f} (每 1 韩元广告费带来的销售额)")

    # ------------------------------------------------------------------
    print("\n[9] value_counts() — 类别列的构成比 (Excel 里重复 COUNTIF 的活儿一行搞定)")
    print("    各门店行数:")
    print(df["store"].value_counts().to_string())
    print("    各星期行数 (90 天除不尽 7，所以每个星期的天数不一样):")
    print(df["weekday"].value_counts().to_string())

    # ------------------------------------------------------------------
    print("\n[10] 缺失值个数诊断 — isna().sum()")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())
    ratio = df["revenue"].isna().mean() * 100
    print(f"    -> revenue 缺失比例 {ratio:.2f}%。处理策略在 level06 学。")

    print("\n小结: DataFrame = index + columns + 值。抽出一列就是 Series。")
    print("拿到新数据，就按 shape -> head -> info -> describe 的顺序打招呼。")


if __name__ == "__main__":
    main()
