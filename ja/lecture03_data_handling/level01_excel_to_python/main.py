"""
Excel から Python へ — 売上表を CSV に保存/再読み込みした後、
Excel でやっていた SUM・SUMIF・オートフィルタ・並べ替えを標準ライブラリのコードで再現します。
欠損・マイナスの汚染をスキップした件数を自ら報告させることで、
「検証可能な手順」としてのコードと、再現性の価値を体験します。
"""

import csv
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

BASE_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"


def load_clean_rows(csv_path: str) -> tuple[list[dict], int, int]:
    """CSV を読み、revenue を数値に復元します。
    欠損 (空文字列)・マイナスの行はスキップし、その数を一緒に返します。"""
    clean, n_missing, n_negative = [], 0, 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw = r["revenue"]
            if raw == "" or raw == "None":          # 欠損 → スキップ
                n_missing += 1
                continue
            revenue = int(raw)
            if revenue < 0:                          # マイナスの汚染 → スキップ
                n_negative += 1
                continue
            r["revenue"] = revenue                   # 型の復元 (str → int)
            r["ad_cost"] = int(r["ad_cost"])
            clean.append(r)
    return clean, n_missing, n_negative


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] 売上表60日分を CSV に保存 (Excel でも開ける形式)")
    rows = hjh_data.sales_table(n_days=60, seed=42)
    csv_path = hjh_data.to_csv(rows, str(OUT_DIR / "sales.csv"))
    print(f"    保存完了: {csv_path} ({len(rows)}行)")
    print()

    print("[2] CSV の読み直し — すべての値が「文字列」として入ってきます")
    with open(csv_path, newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    print(f"    先頭の行: {first}")
    print(f"    revenue の型: {type(first['revenue']).__name__} ← 数値ではありません!")
    print()

    print("[3] 型の復元 + 汚染行のスキップ (捨てたものは必ず報告)")
    clean, n_missing, n_negative = load_clean_rows(csv_path)
    print(f"    使用 {len(clean)}行 / 欠損スキップ {n_missing}件 / マイナススキップ {n_negative}件")
    print()

    print("[4] SUM — Excel の =SUM(G:G) に相当")
    total = sum(r["revenue"] for r in clean)
    print(f"    60日全体の売上合計: {total:,}ウォン")
    print()

    print("[5] SUMIF — 店舗別の合計 (辞書への累積)")
    by_store: dict[str, int] = {}
    for r in clean:
        by_store[r["store"]] = by_store.get(r["store"], 0) + r["revenue"]
    for store, subtotal in sorted(by_store.items(), key=lambda kv: kv[1]):
        print(f"    {store}: {subtotal:>15,}ウォン")
    print()

    print("[6] オートフィルタ — 「渋谷店 & 週末」の行だけを選んで平日と比較")
    gangnam = [r for r in clean if r["store"] == "渋谷店"]
    weekend = [r for r in gangnam if r["weekday"] in ("土", "日")]
    weekday_rows = [r for r in gangnam if r["weekday"] not in ("土", "日")]
    avg_weekend = sum(r["revenue"] for r in weekend) / len(weekend)
    avg_weekday = sum(r["revenue"] for r in weekday_rows) / len(weekday_rows)
    print(f"    渋谷店 週末の平均: {avg_weekend:>12,.0f}ウォン ({len(weekend)}行)")
    print(f"    渋谷店 平日の平均: {avg_weekday:>12,.0f}ウォン ({len(weekday_rows)}行)")
    print(f"    → 週末は平日の {avg_weekend / avg_weekday:.2f}倍です。")
    print()

    print("[7] 並べ替え — 売上上位5 (Excel の降順ソート)")
    top5 = sorted(clean, key=lambda r: r["revenue"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        print(f"    {i}位 | {r['date']} {r['weekday']} | {r['store']} "
              f"{r['category']} | {r['revenue']:,}ウォン")
    print()

    print("[8] 再現性 — このスクリプトの本当の価値")
    print("    いま見たすべての数字は「固定 seed + 記録された手順」から生まれました。")
    print("    明日もう一度実行しても、他の人が実行しても、完全に同じ結果になります。")
    print("    Excel のクリックは記憶に残るだけですが、コードは文書として残ります。")


if __name__ == "__main__":
    main()
