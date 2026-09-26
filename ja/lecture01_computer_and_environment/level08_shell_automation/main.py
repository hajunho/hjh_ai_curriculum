"""
Lecture 01 / Level 08 — シェルスクリプトで繰り返し作業を自動化する

散らかった共有フォルダ(一時フォルダに再現)を自動で整理する実習です。
自動化の標準パターン「収集 -> 判断 -> (ドライラン) -> 実行 -> 報告」に沿って、
正規表現でファイル名を解析して店舗・年月別のフォルダへ一括移動し、
ファイル名を統一されたルールに変えます。
"""

import random
import re
import tempfile
from pathlib import Path

# 散らかった共有フォルダを再現するファイルたち — 命名ルールがばらばら!
MESSY_FILES = [
    "売上報告_渋谷_2024-03.csv", "売上報告_渋谷_2024-04.csv",
    "売上報告_新宿_2024-03.csv", "sales_fukuoka_2024-03.csv",
    "sales_fukuoka_2024-04.csv", "売上報告_大阪_2024-04.csv",
    "sales_osaka_2024-03.csv", "売上報告_新宿_2024-04.csv",
    "メモ.txt", "ランチ投票.txt", "発表資料.pptx", "昔のバックアップ.zip",
    "売上報告_渋谷_2024-05.csv", "sales_fukuoka_2024-05.csv",
]

# ファイル名の解析ルール: 「店舗」と「年-月」を抜き出す正規表現2種類
PATTERNS = [
    re.compile(r"売上報告_(?P<store>.+)_(?P<ym>\d{4}-\d{2})\.csv"),
    re.compile(r"sales_(?P<store>.+)_(?P<ym>\d{4}-\d{2})\.csv"),
]
# 英語の店舗名 -> 日本語に統一
STORE_KO = {"fukuoka": "福岡", "osaka": "大阪"}


def create_mess(folder: Path):
    """[1] 散らかりの生成: ばらばらなファイルを seed 固定の乱数の内容で作ります。"""
    rng = random.Random(42)  # seed 固定 — 何回実行しても同じ内容
    for name in MESSY_FILES:
        revenue = rng.randint(500, 2000)
        (folder / name).write_text(f"revenue,{revenue}000\n", encoding="utf-8")


def plan_moves(folder: Path):
    """[2] 収集・判断: ファイルごとに「どこへ、どんな名前で」送るか、計画だけを立てます。

    実行と分離しておけば、ドライラン(予行演習)がタダで手に入ります。"""
    plans = []  # (元の Path, 目的地の Path, 分類の理由)
    for path in sorted(folder.glob("*")):        # 収集: 対象の一覧
        if path.is_dir():
            continue
        for pattern in PATTERNS:                 # 判断: ルールの適用
            matched = pattern.match(path.name)
            if matched:
                store = STORE_KO.get(matched["store"], matched["store"])
                ym = matched["ym"]
                # 目的地: 店舗フォルダ/店舗_年-月.csv と名前まで統一
                dest = folder / store / f"{store}_{ym}.csv"
                plans.append((path, dest, f"売上ファイル -> {store}/{ym}"))
                break
        else:  # どのルールにも合わなければ、人が見る要確認の箱へ
            dest = folder / "_要確認" / path.name
            plans.append((path, dest, "ルール不一致 -> 要確認の箱"))
    return plans


def execute_moves(plans):
    """[4] 実行: 計画表どおりフォルダを作り、一括移動します。"""
    moved = 0
    for src, dest, _ in plans:
        dest.parent.mkdir(parents=True, exist_ok=True)  # シェルの mkdir -p
        if dest.exists():                                # 上書き事故の防止!
            dest = dest.with_name(dest.stem + "_重複" + dest.suffix)
        src.rename(dest)                                 # シェルの mv
        moved += 1
    return moved


def draw_tree(folder: Path):
    """整理結果をツリーで出力します。"""
    for p in sorted(folder.rglob("*")):
        depth = len(p.relative_to(folder).parts) - 1
        tag = "/" if p.is_dir() else ""
        print("      " + "  " * depth + f"- {p.name}{tag}")


def main():
    print("=" * 60)
    print("ファイル整理の自動化 — 収集 -> 判断 -> ドライラン -> 実行 -> 報告")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        shared = Path(tmp) / "shared_folder"
        shared.mkdir()

        create_mess(shared)
        print(f"\n[1] 散らかりの再現: 命名ルールがばらばらのファイル {len(MESSY_FILES)}個を生成")
        print("    " + ", ".join(MESSY_FILES[:6]) + " ...")

        plans = plan_moves(shared)
        print(f"\n[2] 収集・判断: 正規表現でファイル名を解析して計画 {len(plans)}件を立案")

        # [3] ドライラン: 実行前に計画表だけを目で検証 — 自動化の第一の安全装置
        print("\n[3] ドライラン(予行演習) — まだ何も移していません")
        for src, dest, reason in plans:
            print(f"    {src.name:<28} -> {dest.relative_to(shared)}  ({reason})")

        moved = execute_moves(plans)
        print(f"\n[4] 実際の実行: {moved}件の移動・名前統一が完了")

        print("\n[5] 結果報告 — 整理後のフォルダ構造:")
        draw_tree(shared)
        review = sum(1 for _, dest, _ in plans if "_要確認" in str(dest))
        print(f"\n    サマリー: 自動分類 {moved - review}件 / 人の確認が必要 {review}件")
        print("    手作業ならファイルあたり30秒 x 14個 = 約7分、")
        print("    スクリプトは0.1秒 — しかも毎週やらせてもタダです。")

    print("\nまとめ: 計画(plan)と実行(execute)を分ければ、ドライランがタダで手に入ります。")
    print("        一括作業は必ず「計画表の検証 -> 実行」の順で!")


if __name__ == "__main__":
    main()
