"""内包表記・ジェネレーター・ラムダ・デコレーター — Python らしい表現の入門。

[1] 売上データの変換をループ vs 内包表記で比較し、
[2] 取引100万件をリスト (倉庫) vs ジェネレーター (蛇口) で処理して
    メモリ使用量を実測比較します。
[3] ラムダでソートの基準を渡し、[4] デコレーターの判子を味わいます。
"""

import random
import sys
import time


# =============================================================
# [4] 用のデコレーター: 実行時間を測る「決裁の判子」
# =============================================================
def stopwatch(func):
    """関数を受け取り、「時間計測の機能が上乗せされた関数」を返す。"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)          # 元の関数はそのまま実行
        elapsed = time.perf_counter() - start
        print(f"    (stopwatch: {func.__name__} 実行 {elapsed * 1000:.1f}ms)")
        return result
    return wrapper


@stopwatch                       # = sum_with_list = stopwatch(sum_with_list)
def sum_with_list(n):
    """取引 n 件をリストで全部作ってから合算 (倉庫方式)。"""
    rows = [i % 1000 * 100 for i in range(n)]   # n 件をメモリに丸ごと積載
    return sum(rows), sys.getsizeof(rows)


@stopwatch
def sum_with_generator(n):
    """取引 n 件をジェネレーターで流しながら合算 (蛇口方式)。"""
    stream = (i % 1000 * 100 for i in range(n))  # まだ何も作られていない
    return sum(stream), sys.getsizeof(stream)


def transaction_stream(n):
    """yield のデモ: 取引を1件ずつ送り出すジェネレーター関数。"""
    for i in range(n):
        yield {"id": i, "amount": (i * 37) % 900 * 1000}   # 1件渡して止まって待つ


def main():
    print("=" * 56)
    print(" 内包表記・ジェネレーター・ラムダ・デコレーター")
    print("=" * 56)

    random.seed(42)
    # 店舗別の取引データ (辞書のリスト)
    rows = [
        {"store": store, "amount": random.randint(3, 80) * 10000}
        for store in ["渋谷", "新宿", "池袋", "品川", "上野", "横浜", "大阪", "福岡"]
    ]

    # ---------------------------------------------------------
    # [1] ループ vs 内包表記: 同じ変換、2つの表記
    # ---------------------------------------------------------
    print("\n[1] ループ vs 内包表記 — 10万ウォン以上の取引に付加価値税を反映")

    # ループ方式 (4行)
    loop_result = []
    for r in rows:
        if r["amount"] >= 100000:
            loop_result.append(int(r["amount"] * 1.1))

    # 内包表記方式 (1行): 「10万以上の r それぞれの金額 x1.1 のリスト」
    comp_result = [int(r["amount"] * 1.1) for r in rows if r["amount"] >= 100000]

    print(f"  ループ方式 (4行)    : {loop_result}")
    print(f"  内包表記 (1行)      : {comp_result}")
    print(f"  2つの結果は同じ?    : {loop_result == comp_result}")

    # 辞書内包表記: 店舗 -> 金額の表を作る
    by_store = {r["store"]: r["amount"] for r in rows}
    print(f"  辞書内包表記        : {by_store}")

    # ---------------------------------------------------------
    # [2] ジェネレーター: 大容量ストリームのメモリ感覚
    # ---------------------------------------------------------
    print("\n[2] リスト (倉庫) vs ジェネレーター (蛇口) — 取引100万件の合算")

    N = 1_000_000
    total_l, mem_l = sum_with_list(N)
    print(f"  リスト方式          合計 {total_l:>13,} / メモリ {mem_l:>10,} bytes (~{mem_l / 1e6:.1f}MB)")
    total_g, mem_g = sum_with_generator(N)
    print(f"  ジェネレーター方式  合計 {total_g:>13,} / メモリ {mem_g:>10,} bytes")
    print(f"  -> 同じ答え、メモリの差は約 {mem_l // mem_g:,}倍。データが大きいほど蛇口!")

    # yield の動きを目で: 先頭3件だけ取り出してみる
    tap = transaction_stream(N)                  # まだ1件も作られていない
    print("  yield のデモ (必要な分だけ取り出す):")
    for _ in range(3):
        print(f"    next() -> {next(tap)}")
    print("    ... 残りは作られてもいないのでコスト 0")

    # 一度消費したジェネレーターは再利用不可
    small = (x for x in range(3))
    print(f"  1回目の消費: {list(small)} / 2回目の消費: {list(small)}  <- 空の結果 (一度使えば終わり)")

    # ---------------------------------------------------------
    # [3] ラムダ: ソート・最大の「基準」を付箋で渡す
    # ---------------------------------------------------------
    print("\n[3] ラムダ — key 引数に基準を渡す")

    top3 = sorted(rows, key=lambda r: r["amount"], reverse=True)[:3]
    print("  売上上位3店舗:")
    for rank, r in enumerate(top3, start=1):
        print(f"    {rank}位 {r['store']} {r['amount']:,}ウォン")

    best = max(rows, key=lambda r: r["amount"])
    worst = min(rows, key=lambda r: r["amount"])
    print(f"  最高 {best['store']} {best['amount']:,}ウォン / 最低 {worst['store']} {worst['amount']:,}ウォン")

    # ジェネレーター式 + ラムダなしでもできる集計: sum(式 for ...)
    big_total = sum(r["amount"] for r in rows if r["amount"] >= 300000)
    print(f"  30万ウォン以上の取引の合計 (ジェネレーター式): {big_total:,}ウォン (中間リストなし)")

    # ---------------------------------------------------------
    # [4] デコレーターのまとめ
    # ---------------------------------------------------------
    print("\n[4] デコレーター — 実は [2] ですでに判子が押されていました")
    print("  sum_with_list / sum_with_generator の上の @stopwatch がデコレーター。")
    print("  関数本体は1行も直さずに「実行時間の出力」機能が上乗せされました。")
    print("  @判子の表記 = 関数を包む関数。ログや権限確認などの共通手順に使います。")

    print("\n[終] 内包表記は「一文として読めるときだけ」、ジェネレーターは「データが大きいとき」、")
    print("     ラムダは「基準を渡す付箋」、デコレーターは「読めれば十分」。")


if __name__ == "__main__":
    main()
