"""
Lecture 01 / Level 00 — コンピュータとは何をする機械なのか

コンピュータの仕事の進め方を「入力 -> 計算 -> 出力」の流れで体験します。
CPU(実務担当者)、メモリ(デスク)、ストレージ(書類倉庫)の役割分担を、
カフェチェーン本社の売上集計業務になぞらえて確認します。
標準ライブラリのみを使用し、インターネット接続は不要です。
"""

import json
import tempfile
import time
from pathlib import Path

# 繰り返し計算の回数(「やってみよう」の課題1でこの値を変えてみてください)
LOOP_COUNT = 1_000_000


def receive_orders():
    """[入力] 受付窓口: 店舗別の注文データが届きます。"""
    # (店舗名, 注文件数, 総売上額[ウォン]) — コード内で作った架空データです。
    orders = [
        ("渋谷店", 182, 1_512_000),
        ("新宿店", 141, 1_098_000),
        ("大阪店", 210, 1_745_000),
        ("福岡店", 95, 702_000),
    ]
    return orders


def save_to_storage(orders, path):
    """[ストレージ] 書類倉庫に保管: ファイルに保存すれば電源が切れても残ります。"""
    rows = [{"store": s, "count": c, "revenue": r} for s, c, r in orders]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.stat().st_size  # 倉庫に保管された書類のサイズ(バイト)


def load_from_storage(path):
    """[メモリ] 倉庫の書類をデスクの上へ: ファイルを読んでメモリに載せます。"""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(row["store"], row["count"], row["revenue"]) for row in rows]


def process_orders(orders):
    """[計算] CPU 担当者: 入力を受け取り、計算して結果を返します。(IPO モデル)"""
    total_revenue = sum(revenue for _, _, revenue in orders)
    total_count = sum(count for _, count, _ in orders)
    avg_per_order = total_revenue / total_count
    best_store = max(orders, key=lambda row: row[2])
    return {
        "total_revenue": total_revenue,
        "total_count": total_count,
        "avg_per_order": avg_per_order,
        "best_store": best_store[0],
        "best_revenue": best_store[2],
    }


def measure_cpu_speed():
    """CPU が単純な計算をどれほど速く繰り返せるか、時間を測ってみます。"""
    started = time.perf_counter()
    acc = 0
    for i in range(LOOP_COUNT):
        acc += i  # ごく小さな足し算をひたすら繰り返す
    elapsed = time.perf_counter() - started
    return elapsed, acc


def show_binary(text):
    """文字がコンピュータの内部で 0 と 1 でどう表現されるかを見せます。"""
    for ch in text:
        code = ord(ch)  # 文字ごとに決められた番号(Unicode)
        print(f"    文字 '{ch}' -> 番号 {code} -> 二進法 {code:08b}")


def main():
    print("=" * 60)
    print("株式会社コンピュータ 業務フロー体験 — 入力 -> 計算 -> 出力")
    print("=" * 60)

    # [1] 入力: 受付窓口に注文データが届きます。
    orders = receive_orders()
    print(f"\n[1] 入力(Input): 受付窓口に {len(orders)} 店舗の注文が到着")
    for store, count, revenue in orders:
        print(f"    - {store}: {count}件, {revenue:,}ウォン")

    # [2] ストレージ <-> メモリ: 倉庫に保管してから、再びデスクの上へ取り出します。
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "orders_archive.json"
        size = save_to_storage(orders, archive)
        print(f"\n[2] ストレージ(倉庫): '{archive.name}' ファイルとして保管 ({size}バイト)")
        print("    - ファイルに保存した内容は電源を切っても残ります (不揮発性)")
        orders_on_desk = load_from_storage(archive)
        print(f"    - 倉庫から取り出してデスク(メモリ)に展開: {len(orders_on_desk)}件を復元")
        print("    - メモリ上のデータはプログラムが終わると消えます (揮発性)")

    # [3] 計算: CPU 担当者が合計と平均を出します。
    report = process_orders(orders_on_desk)
    elapsed, _ = measure_cpu_speed()
    print(f"\n[3] 計算(Process): CPU 担当者の決裁スピード体験")
    print(f"    - 単純な足し算を {LOOP_COUNT:,}回 繰り返すのにかかった時間: {elapsed:.3f}秒")
    print(f"    - 1秒あたり約 {LOOP_COUNT / elapsed:,.0f}回 — 人間には真似すらできない速さです")

    # [4] 出力: 人が読みやすいレポートとして発送します。
    print(f"\n[4] 出力(Output): 売上集計レポート")
    print(f"    - 総売上            : {report['total_revenue']:,}ウォン")
    print(f"    - 総注文件数        : {report['total_count']:,}件")
    print(f"    - 注文あたり平均売上: {report['avg_per_order']:,.0f}ウォン")
    print(f"    - 売上トップの店舗  : {report['best_store']} ({report['best_revenue']:,}ウォン)")

    # [5] 二進法の味見: すべての情報は結局 0 と 1 です。
    print(f"\n[5] 二進法の味見: 'AI' という文字の実際の保存形")
    show_binary("AI")

    print("\nまとめ: どんなプログラムも「入力 -> 計算 -> 出力」の3つに分けて眺めましょう。")
    print("        CPU=担当者、メモリ=デスク(高速・揮発)、ストレージ=倉庫(低速・永続)です。")


if __name__ == "__main__":
    main()
