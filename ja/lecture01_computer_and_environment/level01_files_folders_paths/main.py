"""
Lecture 01 / Level 01 — ファイル・フォルダ・パスの概念

pathlib で架空の会社「あさひ商事」の文書フォルダツリーを作って探検します。
絶対パス/相対パスの違い、拡張子別のファイル集計、パスの分解(parent/stem/suffix)を、
一時フォルダの中で安全に実習します。終了時に一時フォルダは自動削除されます。
"""

import os
import tempfile
from collections import Counter
from pathlib import Path


def build_company_tree(root: Path):
    """[ツリー生成] 部署フォルダとサンプル文書ファイルを作ります。"""
    # フォルダ構造: (相対パス) — mkdir(parents=True) で途中のフォルダごと一度に作成
    folders = [
        "総務部",
        "営業部/契約書",
        "営業部/報告書",
        "開発部/コード",
    ]
    for name in folders:
        (root / name).mkdir(parents=True, exist_ok=True)

    # ファイル: (相対パス, 内容) — 拡張子は「文書種別のハンコ」です。
    files = [
        ("総務部/備品申請書.txt", "ボールペン10本、A4用紙2箱"),
        ("総務部/駐車規定.txt", "地下2階は来客専用です。"),
        ("営業部/契約書/A社_契約書.txt", "甲: A社 / 乙: あさひ商事"),
        ("営業部/契約書/B社_契約書.txt", "甲: B社 / 乙: あさひ商事"),
        ("営業部/報告書/3月_実績.csv", "store,revenue\n渋谷,1512000\n新宿,1098000"),
        ("営業部/報告書/4月_実績.csv", "store,revenue\n渋谷,1620000\n新宿,1150000"),
        ("開発部/コード/hello.py", "print('hello')"),
    ]
    for rel_path, content in files:
        (root / rel_path).write_text(content, encoding="utf-8")
    return len(folders), len(files)


def draw_tree(folder: Path, indent: int = 0):
    """[ツリー描画] 再帰呼び出しでフォルダ構造をインデント図として出力します。"""
    marker = "📁" if indent == 0 else "└─"
    print("    " + "   " * indent + f"{marker} {folder.name}/")
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            draw_tree(child, indent + 1)  # フォルダなら自分自身をもう一度呼ぶ (再帰)
        else:
            print("    " + "   " * (indent + 1) + f"└─ {child.name}")


def main():
    print("=" * 60)
    print("ファイル・フォルダ・パス実習 — あさひ商事の文書庫探検")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "asahi_corp"
        root.mkdir()

        # [1] フォルダツリー生成
        n_folders, n_files = build_company_tree(root)
        print(f"\n[1] ツリー生成: フォルダ {n_folders}個(+途中のフォルダ)、ファイル {n_files}個を作りました")
        print(f"    場所(一時フォルダ): {root}")

        # [2] ツリー描画 — ターミナルの tree コマンドを Python で真似る
        print("\n[2] 文書庫の全体構造:")
        draw_tree(root)

        # [3] 絶対パス vs 相対パス
        target = root / "営業部" / "報告書" / "3月_実績.csv"
        print("\n[3] 同じファイルを指す2種類の住所:")
        print(f"    絶対パス: {target.resolve()}")
        print(f"    相対パス(会社の正面玄関基準): {target.relative_to(root)}")
        # '..' はひとつ上の階(親フォルダ)を意味します。
        sibling = target.parent.parent / "契約書" / "A社_契約書.txt"
        print(f"    報告書フォルダから '../契約書/A社_契約書.txt' へ行くと:")
        print(f"      -> {sibling.relative_to(root)} (存在する? {sibling.exists()})")
        print(f"    カレントワーキングディレクトリ(CWD): {os.getcwd()}")
        print("      -> 相対パスは常にこの CWD(または明示した基準)から解釈されます")

        # [4] 拡張子別のファイル集計 — rglob は下位フォルダまで全部探します
        counts = Counter(p.suffix for p in root.rglob("*") if p.is_file())
        print("\n[4] 拡張子(文書種別のハンコ)別の集計:")
        for ext, count in sorted(counts.items()):
            print(f"    {ext:<6} {count}個")

        # [5] パスの解剖 — パスひとつを部位ごとに分解
        print("\n[5] パスの解剖: 営業部/報告書/3月_実績.csv")
        print(f"    parent (入っているフォルダ)  : {target.parent.name}/")
        print(f"    name   (ファイルのフルネーム): {target.name}")
        print(f"    stem   (拡張子を除いた名前)  : {target.stem}")
        print(f"    suffix (拡張子)              : {target.suffix}")

    print("\n[6] 実習終了: 一時フォルダは自動削除されました — コンピュータに痕跡はありません")
    print("まとめ: 絶対パスは正面玄関からのフル住所、相対パスは現在位置基準の略式住所です。")


if __name__ == "__main__":
    main()
