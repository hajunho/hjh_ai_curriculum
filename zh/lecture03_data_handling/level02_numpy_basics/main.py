"""
NumPy 数组基础 — 亲手测量 Python 循环 vs 向量化的速度差，
再把销售数据变成 ndarray，掌握 shape/dtype、聚合、广播和布尔掩码。
核心信息: 把运算的单位从"一个数字"换成"一堆数字"，
代码会变短，速度会快上几十倍。
"""

import pathlib
import sys
import time

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

N_SPEED = 1_000_000     # 速度实验要用的数字个数


def python_sum_of_squares(values: list[float]) -> float:
    """Python 循环方式 — 一个一个拿起来求平方再累加。"""
    total = 0.0
    for v in values:
        total += v * v
    return total


def main() -> None:
    rng = np.random.default_rng(42)          # 固定 seed → 随机数永远一样

    print(f"[1] 速度对决 — {N_SPEED:,} 个数字的平方和")
    data_arr = rng.random(N_SPEED)           # 100 万个 0~1 之间的实数
    data_list = data_arr.tolist()            # 同样的值也准备一份 Python 列表

    t0 = time.perf_counter()
    loop_result = python_sum_of_squares(data_list)
    loop_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_result = float((data_arr * data_arr).sum())   # 向量化: 没有循环
    vec_sec = time.perf_counter() - t0

    print(f"    Python 循环: {loop_sec * 1000:8.1f} ms")
    print(f"    NumPy 向量化: {vec_sec * 1000:8.1f} ms")
    print(f"    → 大约快 {loop_sec / vec_sec:.0f} 倍。"
          f"(两个答案之差 {abs(loop_result - vec_result):.6f} → 是同一个计算)")
    print()

    print("[2] 把销售数据变成数组 — 确认 shape 和 dtype")
    rows = hjh_data.sales_table(n_days=90, seed=42)
    # 排除缺失 (None)，只把销售额抽出来做成数组
    rev = np.array([r["revenue"] for r in rows if r["revenue"] is not None])
    weekend_mask_src = [r["weekday"] in ("周六", "周日")
                        for r in rows if r["revenue"] is not None]
    is_weekend = np.array(weekend_mask_src)
    n_missing = len(rows) - len(rev)
    print(f"    原始 {len(rows)} 行中排除缺失 {n_missing} 条 → 数组 {len(rev)} 个")
    print(f"    rev.shape = {rev.shape}, rev.dtype = {rev.dtype}")
    print()

    print("[3] 聚合一行搞定 — 不用循环就能出统计量")
    print(f"    合计: {rev.sum():>16,} 韩元")
    print(f"    平均: {rev.mean():>16,.0f} 韩元")
    print(f"    标准差: {rev.std():>12,.0f} 韩元")
    print(f"    最大: {rev.max():>16,} 韩元 / 最小: {rev.min():,} 韩元 (含负数污染)")
    print()

    print("[4] 广播 — 增值税 10% 用\"一个章\"盖完")
    with_vat = rev * 1.1                       # 一个数字被扩展到整个数组
    rounded = np.round(with_vat, -3)           # 千元位四舍五入也是向量操作
    print(f"    税前平均: {rev.mean():>14,.0f} 韩元")
    print(f"    税后平均: {with_vat.mean():>14,.0f} 韩元 (= 税前 × 1.1)")
    print(f"    千元四舍五入示例: {rev[:3]} → {rounded[:3].astype(int)}")
    print()

    print("[5] 布尔掩码 — 用条件挑出来")
    weekend_avg = rev[is_weekend].mean()       # 只选 True 位置上的值
    weekday_avg = rev[~is_weekend].mean()      # ~ 就是把 True/False 翻过来
    print(f"    周末平均: {weekend_avg:>14,.0f} 韩元 ({int(is_weekend.sum())} 条)")
    print(f"    工作日平均: {weekday_avg:>14,.0f} 韩元 ({int((~is_weekend).sum())} 条)")
    print(f"    → 周末是工作日的 {weekend_avg / weekday_avg:.2f} 倍")
    negative = rev < 0
    print(f"    负数污染条数: {int(negative.sum())} 条 "
          f"(数 True 的个数 = 掩码的 sum)")
    print(f"    排除负数后的平均: {rev[~negative].mean():,.0f} 韩元")
    print()
    print("    下一关我们学给这个数组穿上\"列名\"的 DataFrame。")


if __name__ == "__main__":
    main()
