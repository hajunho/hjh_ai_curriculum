"""
Lecture 01 / Level 11 — リモートサーバー・クラウド・GPU 環境

GPU が AI に必須である理由「並列性」を、CPU コア数個で直接体感します。
分割しやすい計算の仕事(区間ごとの素数の個数数え)を
(a) ひとりで(単一プロセス) vs (b) 分担して(マルチプロセス) 処理して時間を比較し、
この原理が働き手数千人の GPU とクラウド費用にどうつながるかを見ます。
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

# 仕事の設定 (「やってみよう」で変えてみてください)
RANGE_END = 400_000   # 2 〜 この数字までの間で素数を数えます
N_CHUNKS = 8          # 仕事をいくつの断片に分けるか
N_WORKERS = 4         # 同時に働く働き手(プロセス)の数


def count_primes(bounds):
    """区間 [start, end) の素数の個数を数えます — 分割しやすい単純計算の仕事。"""
    start, end = bounds
    count = 0
    for n in range(max(start, 2), end):
        is_prime = True
        divisor = 2
        while divisor * divisor <= n:     # 平方根まで割ってみれば十分
            if n % divisor == 0:
                is_prime = False
                break
            divisor += 1
        if is_prime:
            count += 1
    return count


def make_chunks(end, n_chunks):
    """[1] 仕事の準備: 全区間を n 個の断片に分けます (常に同じ仕事 — 再現性)。"""
    size = end // n_chunks
    return [(i * size, end if i == n_chunks - 1 else (i + 1) * size) for i in range(n_chunks)]


def solo_run(chunks):
    """[2] ひとりで働く: 1つのプロセスが断片を順番に処理。"""
    started = time.perf_counter()
    total = sum(count_primes(chunk) for chunk in chunks)
    return total, time.perf_counter() - started


def team_run(chunks, n_workers):
    """[3] 分担して働く: 複数のプロセスが断片を同時に処理。"""
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(count_primes, chunks))  # 核心の1行!
    return sum(results), time.perf_counter() - started


def main():
    print("=" * 60)
    print("並列性の体験 — ひとり vs 分担、そして GPU への道")
    print("=" * 60)

    cores = os.cpu_count()
    print(f"\n[1] 仕事の準備: 2〜{RANGE_END:,} の区間の素数数えを {N_CHUNKS}個の断片に分割")
    print(f"    このコンピュータの CPU コア(専門家の席)の数: {cores}個 / 今日の働き手の数: {N_WORKERS}人")
    chunks = make_chunks(RANGE_END, N_CHUNKS)

    print("\n[2] ひとりで働く — 単一プロセスが断片を順番に処理")
    solo_total, solo_time = solo_run(chunks)
    print(f"    見つけた素数: {solo_total:,}個 / かかった時間: {solo_time:.2f}秒")

    print(f"\n[3] 分担して働く — {N_WORKERS}人の働き手(プロセス)が同時に処理")
    team_total, team_time = team_run(chunks, N_WORKERS)
    print(f"    見つけた素数: {team_total:,}個 / かかった時間: {team_time:.2f}秒")
    print(f"    結果の一致確認: {'同じ — 仕事を分割しても答えは同じです' if solo_total == team_total else '違う?!'}")

    # [4] 成績表: 倍速と、理論どおりに出ない理由
    speedup = solo_time / team_time if team_time > 0 else float("inf")
    print("\n[4] 成績表")
    print(f"    倍速: {speedup:.2f}倍 (働き手 {N_WORKERS}人を投入)")
    print(f"    理論上の最大 {N_WORKERS}倍に届かない理由:")
    print("      - 働き手の招集コスト: プロセスを作って仕事を配るのにも時間がかかる")
    print("      - 順次区間: 結果の取りまとめのように分割できない仕事が残る (アムダールの法則)")
    print("      - 席の限界: コア数より働き手が多くても、同時に座る席が無い")

    # [5] GPU とクラウドへの拡張
    print("\n[5] この原理の先に GPU があります")
    print("    - 今日: 働き手4人(CPU コア) — AI の学習: 働き手数千人(GPU コア)")
    print("    - AI の学習 = 依存関係の無い掛け算・足し算の海 -> 分割しやすい仕事の極み")
    print("    - 接続は ssh(保安電話線)、ファイル転送は scp(宅配便)、サーバーはほぼ Linux")
    print("\n    クラウド見積もりの感覚 (架空の例、金額は韓国ウォン=KRW):")
    hourly_big, hourly_small = 30_000, 1_000
    plan_a = hourly_big * 48
    plan_b = hourly_small * 3 * 2 + hourly_big * 48
    print(f"      計画 A: 大型 GPU(1時間 {hourly_big:,}ウォン) いきなり48時間 = {plan_a:,}ウォン")
    print(f"              (設定ミス時は再実行 -> 最大 {plan_a * 2:,}ウォン)")
    print(f"      計画 B: 小型(1時間 {hourly_small:,}ウォン) 事前実験3時間x2回の後に本学習")
    print(f"              = {plan_b:,}ウォン — ミスを6千ウォンの実験で先につかまえます")
    print("      教訓: 実験は安く、本学習だけ高く。そしてインスタンスは消すこと!")

    print("\nまとめ: 並列化の判断基準は「仕事を分割できるか」です。")
    print("        分割できる仕事 + 働き手数千人(GPU) = AI 時代の計算力です。")


if __name__ == "__main__":
    main()
