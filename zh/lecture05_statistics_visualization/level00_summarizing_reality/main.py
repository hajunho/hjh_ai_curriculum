"""
level00 — 用数字概括现实意味着什么

生成 4 组平均数都恰好是 100 的销售数据 (A~D)，
验证仅凭概括值 (平均数) 完全区分不出截然不同的现实。
再用文本直方图亲手画出每组数据的"形状"。
"""

import random
import statistics


def make_datasets() -> dict[str, list[float]]:
    """设计成平均数都为 100 (万韩元) 的日销售额数据 4 组。"""
    rng = random.Random(500)  # 固定 seed 以保证可复现
    n = 200

    # A. 稳定型: 均匀聚集在 100 附近 (靠回头客的店)
    a = [rng.gauss(100, 8) for _ in range(n)]

    # B. 双群型: 工作日客人 (60 档) 和周末客人 (140 档) 各占一半
    b = [rng.gauss(60, 7) for _ in range(n // 2)] + \
        [rng.gauss(140, 7) for _ in range(n // 2)]

    # C. 极端值型: 平时在 80 档，几单大合同把平均数拉了上去
    c = [rng.gauss(80, 6) for _ in range(n - 4)] + [1200, 1150, 900, 850]

    # D. 碰运气型: 在 40~160 之间大范围散开
    d = [rng.uniform(40, 160) for _ in range(n)]

    data = {"A(稳定型)": a, "B(双群型)": b, "C(极端值型)": c, "D(碰运气型)": d}
    # 校正细微的随机误差，把平均数精确调到 100。
    for name, values in data.items():
        shift = 100 - statistics.mean(values)
        data[name] = [v + shift for v in values]
    return data


def summarize(values: list[float]) -> dict[str, float]:
    """计算最基本的几个概括值。"""
    return {
        "平均": statistics.mean(values),
        "中位数": statistics.median(values),
        "最小": min(values),
        "最大": max(values),
    }


def ascii_hist(values: list[float], lo: float, hi: float,
               n_bins: int = 12, width: int = 40) -> None:
    """划分区间 (bin) 计数、用星号绘制的文本直方图。"""
    counts = [0] * n_bins
    step = (hi - lo) / n_bins
    for v in values:
        idx = int((v - lo) / step)
        idx = max(0, min(n_bins - 1, idx))  # 超出范围的值归入两端
        counts[idx] += 1
    peak = max(counts) or 1
    for i, cnt in enumerate(counts):
        left = lo + i * step
        bar = "*" * round(cnt / peak * width)
        print(f"  {left:7.1f} ~ {left + step:7.1f} | {bar} ({cnt})")


def main() -> None:
    data = make_datasets()

    print("[1] 只看概括值(平均数)，四家店一模一样。")
    for name, values in data.items():
        print(f"    {name:<10} 平均日销售额 = {statistics.mean(values):7.1f} 万韩元")

    print()
    print("[2] 配上其他概括值，差异开始显现。")
    header = f"    {'数据':<10} {'平均':>8} {'中位数':>8} {'最小':>8} {'最大':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name, values in data.items():
        s = summarize(values)
        print(f"    {name:<10} {s['平均']:>8.1f} {s['中位数']:>8.1f} "
              f"{s['最小']:>8.1f} {s['最大']:>8.1f}")
    print("    -> C 的平均数(100)和中位数(约 80)严重错位。")
    print("       这是少数极端值把平均数拉高的信号。")

    print()
    print("[3] 亲眼看形状 — 文本直方图 (0~200 区间)")
    for name, values in data.items():
        print(f"  <{name}>")
        ascii_hist(values, lo=0, hi=200)
        print()

    print("[4] 概括值回答不了的问题")
    print("    Q. 哪家店该增加周末夜班人手？        -> B (双群型)")
    print("    Q. 哪家店该担心对大合同的依赖度？    -> C (极端值型)")
    print("    Q. 哪家店需要防备现金流波动？        -> D (碰运气型)")
    print("    四家店的'平均数'都是 100，但正确的决策各不相同。")
    print("    => 概括只是起点，离散程度、形状、极端值必须逐一确认。")


if __name__ == "__main__":
    main()
