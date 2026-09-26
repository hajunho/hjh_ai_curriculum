"""
level00 — 数字で現実を要約するということ

平均がまったく同じ 100 の売上データ 4 種類 (A〜D) を作り、
要約値 (平均) だけでは全然違う現実を区別できないことを確かめます。
テキストヒストグラムで、各データの「形」を自分の手で描いてみます。
"""

import random
import statistics


def make_datasets() -> dict[str, list[float]]:
    """平均がすべて 100 (万ウォン) になるよう設計した日次売上データ 4 種類。"""
    rng = random.Random(500)  # 再現性のため seed を固定
    n = 200

    # A. 安定型: 100 の周りに均等に集まる (常連中心の店)
    a = [rng.gauss(100, 8) for _ in range(n)]

    # B. 二つの山型: 平日客 (60 前後) と週末客 (140 前後) が半々
    b = [rng.gauss(60, 7) for _ in range(n // 2)] + \
        [rng.gauss(140, 7) for _ in range(n // 2)]

    # C. 極端値型: 普段は 80 前後、大型契約数件が平均を押し上げる
    c = [rng.gauss(80, 6) for _ in range(n - 4)] + [1200, 1150, 900, 850]

    # D. 当たり外れ型: 40〜160 の間に広く散らばる
    d = [rng.uniform(40, 160) for _ in range(n)]

    data = {"A(安定型)": a, "B(二つの山型)": b, "C(極端値型)": c, "D(当たり外れ型)": d}
    # わずかな乱数誤差を補正して、平均を正確に 100 に合わせます。
    for name, values in data.items():
        shift = 100 - statistics.mean(values)
        data[name] = [v + shift for v in values]
    return data


def summarize(values: list[float]) -> dict[str, float]:
    """最も基本的な要約値を計算します。"""
    return {
        "平均": statistics.mean(values),
        "中央値": statistics.median(values),
        "最小": min(values),
        "最大": max(values),
    }


def ascii_hist(values: list[float], lo: float, hi: float,
               n_bins: int = 12, width: int = 40) -> None:
    """区間 (bin) に分けて頻度を数え、アスタリスクで描くテキストヒストグラム。"""
    counts = [0] * n_bins
    step = (hi - lo) / n_bins
    for v in values:
        idx = int((v - lo) / step)
        idx = max(0, min(n_bins - 1, idx))  # 範囲外の値は両端の枠に
        counts[idx] += 1
    peak = max(counts) or 1
    for i, cnt in enumerate(counts):
        left = lo + i * step
        bar = "*" * round(cnt / peak * width)
        print(f"  {left:7.1f} ~ {left + step:7.1f} | {bar} ({cnt})")


def main() -> None:
    data = make_datasets()

    print("[1] 要約 (平均) だけを見ると、4 つの店は同じに見えます。")
    for name, values in data.items():
        print(f"    {name:<10} 平均日次売上 = {statistics.mean(values):7.1f} 万ウォン")

    print()
    print("[2] ほかの要約値を添えると、違いが見え始めます。")
    header = f"    {'データ':<10} {'平均':>8} {'中央値':>8} {'最小':>8} {'最大':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name, values in data.items():
        s = summarize(values)
        print(f"    {name:<10} {s['平均']:>8.1f} {s['中央値']:>8.1f} "
              f"{s['最小']:>8.1f} {s['最大']:>8.1f}")
    print("    -> C は平均 (100) と中央値 (約 80) が大きく食い違います。")
    print("       極端な値が数件、平均を押し上げたというサインです。")

    print()
    print("[3] 形を直接見ます — テキストヒストグラム (0〜200 の区間)")
    for name, values in data.items():
        print(f"  <{name}>")
        ascii_hist(values, lo=0, hi=200)
        print()

    print("[4] 要約が答えられない質問たち")
    print("    Q. 週末の夜間人員を増やすべき店は?        -> B (二つの山型)")
    print("    Q. 大型契約への依存を心配すべき店は?      -> C (極端値型)")
    print("    Q. キャッシュフロー変動への備えが要る店は? -> D (当たり外れ型)")
    print("    4 つの店の「平均」はどれも 100 で同じですが、正しい決定はすべて異なります。")
    print("    => 要約は出発点にすぎず、散らばり・形・極端な値を必ず確認する必要があります。")


if __name__ == "__main__":
    main()
