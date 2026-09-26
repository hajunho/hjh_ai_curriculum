"""ファイルの読み書き — 取引明細 CSV の生成 -> 集計 -> サマリーレポート保存。

with open / csv モジュール / encoding="utf-8" を実習します。
[1] 取引明細の CSV を作り [2] DictReader で読んで集計した後
[3] 人が読むサマリーレポートのテキストを保存します。
[4] 追記 ("a") モードのログとエンコーディングの文字化け再現まで確認します。
成果物はこのファイルがあるフォルダの outputs/ 以下に保存されます。
"""

import csv
import os
import random

# このファイルがあるフォルダ基準で outputs/ のパスを作る (実行場所に関係なく動作)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")


def create_transactions_csv(path):
    """[1] 取引明細20件の CSV ファイルを生成する (システムがくれたファイルの役割)。"""
    random.seed(7)                     # 再現性: 常に同じデータ
    categories = ["事務用品", "食費", "交通費", "ソフトウェア"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "category", "amount"])          # ヘッダー
        for day in range(1, 21):                                 # 9月1〜20日
            category = random.choice(categories)
            amount = random.randint(5, 300) * 1000               # 5千〜30万ウォン
            writer.writerow([f"2026-09-{day:02d}", category, amount])


def summarize_transactions(path):
    """[2] CSV を読んで (総額, 件数, カテゴリ別合計, 最大の取引) を返す。"""
    total = 0
    count = 0
    by_category = {}                   # カテゴリ -> 合計の引き出し
    biggest = ("", "", 0)              # (日付, カテゴリ, 金額)

    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):              # 各行が辞書としてやって来る
            amount = int(row["amount"])            # CSV の値はすべて文字列 -> 変換必須!
            total += amount
            count += 1
            by_category[row["category"]] = by_category.get(row["category"], 0) + amount
            if amount > biggest[2]:
                biggest = (row["date"], row["category"], amount)

    return total, count, by_category, biggest


def write_report(path, total, count, by_category, biggest):
    """[3] 集計結果を人が読むレポートテキストとして保存する。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== 9月 取引明細サマリーレポート ===\n")
        f.write(f"総支出     : {total:,}ウォン ({count}件)\n")
        f.write(f"1件あたり  : {total // count:,}ウォン\n")
        f.write("カテゴリ別支出:\n")
        for category, amount in sorted(by_category.items(), key=lambda kv: -kv[1]):
            share = amount / total
            f.write(f"  - {category:6s}: {amount:>9,}ウォン ({share:.1%})\n")
        f.write(f"最大の取引 : {biggest[0]} {biggest[1]} {biggest[2]:,}ウォン\n")


def main():
    print("=" * 56)
    print(" ファイル入出力 — 取引明細 CSV とサマリーレポート")
    print("=" * 56)

    os.makedirs(OUT_DIR, exist_ok=True)            # outputs/ がなければ作成

    # ---------------------------------------------------------
    # [1] 取引明細 CSV の生成 ("w" 書き込みモード)
    # ---------------------------------------------------------
    print("\n[1] 取引明細 CSV の生成")
    csv_path = os.path.join(OUT_DIR, "transactions.csv")
    create_transactions_csv(csv_path)
    print(f"  保存完了 -> {csv_path}")

    with open(csv_path, encoding="utf-8") as f:    # 先頭3行だけプレビュー
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  プレビュー {i}: {line.strip()}")   # strip で行末の \n を除去

    # ---------------------------------------------------------
    # [2] CSV を読んで集計 (DictReader)
    # ---------------------------------------------------------
    print("\n[2] CSV の読み込みと集計")
    total, count, by_category, biggest = summarize_transactions(csv_path)
    print(f"  総支出 {total:,}ウォン / {count}件")
    for category, amount in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"  {category:6s}: {amount:>9,}ウォン")
    print(f"  最大の取引: {biggest[0]} {biggest[1]} {biggest[2]:,}ウォン")

    # ---------------------------------------------------------
    # [3] サマリーレポートを保存して読み直す
    # ---------------------------------------------------------
    print("\n[3] サマリーレポートの保存")
    report_path = os.path.join(OUT_DIR, "daily_report.txt")
    write_report(report_path, total, count, by_category, biggest)
    print(f"  保存完了 -> {report_path}")

    with open(report_path, encoding="utf-8") as f:
        for line in f:                              # 1行ずつ読む (メモリ節約パターン)
            print(f"  | {line.rstrip()}")

    # ---------------------------------------------------------
    # [4] "a" 追記モードのログ + 文字化けの再現
    # ---------------------------------------------------------
    print("\n[4] 追記モードとエンコーディング")

    log_path = os.path.join(OUT_DIR, "run_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:      # "a": 既存の内容の後に書き足す
        f.write(f"レポート生成完了: 合計 {total:,}ウォン / {count}件\n")
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  run_log.txt 累計 {len(lines)}行 (再実行するたびに増える — 'a' モード)")

    # utf-8 で書いた日本語を latin-1 規格で読むと? (違うコンセントにプラグを挿す)
    sample_path = os.path.join(OUT_DIR, "encoding_sample.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("月間売上報告")
    with open(sample_path, encoding="utf-8") as f:
        ok_text = f.read()
    with open(sample_path, encoding="latin-1") as f:      # わざと間違ったエンコーディング
        broken_text = f.read()
    print(f"  utf-8 で読む   : {ok_text!r}")
    print(f"  latin-1 で読む : {broken_text!r}  <- 文字化け!")
    print("  -> 読み書きともに encoding='utf-8' の明示が、日本語データの生存ルール。")

    print("\n[終] プログラムが終わっても outputs/ のファイルは残ります。自分で開いてみましょう。")


if __name__ == "__main__":
    main()
