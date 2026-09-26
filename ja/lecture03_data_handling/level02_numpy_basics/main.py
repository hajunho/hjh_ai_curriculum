"""
NumPy 配列の基礎 — Python の繰り返し vs ベクトル化の速度を実測し、
売上データを ndarray に変えて shape/dtype・集計・ブロードキャスト・ブールマスクを身につけます。
核心のメッセージ: 「数値1個」ではなく「数値のまとまり」を演算の単位にすれば、
コードは短くなり、速度は数十倍速くなります。
"""

import pathlib
import sys
import time

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

N_SPEED = 1_000_000     # 速度実験に使う数値の個数


def python_sum_of_squares(values: list[float]) -> float:
    """Python の繰り返し方式 — 1個ずつ取り出して2乗し、累積します。"""
    total = 0.0
    for v in values:
        total += v * v
    return total


def main() -> None:
    rng = np.random.default_rng(42)          # seed 固定 → 常に同じ乱数

    print(f"[1] 速度対決 — {N_SPEED:,}個の数値の2乗和")
    data_arr = rng.random(N_SPEED)           # 0〜1 の実数100万個
    data_list = data_arr.tolist()            # 同じ値を Python のリストとしても用意

    t0 = time.perf_counter()
    loop_result = python_sum_of_squares(data_list)
    loop_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_result = float((data_arr * data_arr).sum())   # ベクトル化: 繰り返しなし
    vec_sec = time.perf_counter() - t0

    print(f"    Python の繰り返し: {loop_sec * 1000:8.1f} ms")
    print(f"    NumPy ベクトル化 : {vec_sec * 1000:8.1f} ms")
    print(f"    → 約 {loop_sec / vec_sec:.0f}倍速いです。"
          f"(2つの答えの差 {abs(loop_result - vec_result):.6f} → 同じ計算です)")
    print()

    print("[2] 売上データを配列に — shape と dtype の確認")
    rows = hjh_data.sales_table(n_days=90, seed=42)
    # 欠損 (None) を除いて売上だけを抜き出し、配列にします
    rev = np.array([r["revenue"] for r in rows if r["revenue"] is not None])
    weekend_mask_src = [r["weekday"] in ("土", "日")
                        for r in rows if r["revenue"] is not None]
    is_weekend = np.array(weekend_mask_src)
    n_missing = len(rows) - len(rev)
    print(f"    元の {len(rows)}行のうち欠損 {n_missing}件を除外 → 配列 {len(rev)}個")
    print(f"    rev.shape = {rev.shape}, rev.dtype = {rev.dtype}")
    print()

    print("[3] 集計1行 — 繰り返しなしで統計を出す")
    print(f"    合計: {rev.sum():>16,}ウォン")
    print(f"    平均: {rev.mean():>16,.0f}ウォン")
    print(f"    標準偏差: {rev.std():>12,.0f}ウォン")
    print(f"    最大: {rev.max():>16,}ウォン / 最小: {rev.min():,}ウォン (マイナスの汚染を含む)")
    print()

    print("[4] ブロードキャスト — 付加価値税10%を「判子一発」で")
    with_vat = rev * 1.1                       # 数値1個が配列全体に拡張される
    rounded = np.round(with_vat, -3)           # 千ウォン単位の四捨五入もベクトルで
    print(f"    税抜き平均: {rev.mean():>14,.0f}ウォン")
    print(f"    税込み平均: {with_vat.mean():>14,.0f}ウォン (= 税抜き × 1.1)")
    print(f"    千ウォン四捨五入の例: {rev[:3]} → {rounded[:3].astype(int)}")
    print()

    print("[5] ブールマスク — 条件で選び出す")
    weekend_avg = rev[is_weekend].mean()       # True の位置の値だけを選択
    weekday_avg = rev[~is_weekend].mean()      # ~ は True/False の反転
    print(f"    週末の平均: {weekend_avg:>14,.0f}ウォン ({int(is_weekend.sum())}件)")
    print(f"    平日の平均: {weekday_avg:>14,.0f}ウォン ({int((~is_weekend).sum())}件)")
    print(f"    → 週末は平日の {weekend_avg / weekday_avg:.2f}倍")
    negative = rev < 0
    print(f"    マイナスの汚染の件数: {int(negative.sum())}件 "
          f"(True の個数を数える = マスクの sum)")
    print(f"    マイナス除外の平均: {rev[~negative].mean():,.0f}ウォン")
    print()
    print("    次のレベルでは、この配列に「列の名前」を着せた DataFrame を学びます。")


if __name__ == "__main__":
    main()
