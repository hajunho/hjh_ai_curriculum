"""
Lecture 01 / Level 02 — ターミナルの第一歩

Python で作った「ミニシェルシミュレータ」で pwd/ls/cd/mkdir/cp/mv/rm コマンドの
動作原理を、安全な一時フォルダの中で体験します。
シェルとは結局「現在位置(cwd)を記憶しながらファイル作業を代行してくれる通訳者」
であることを、コードの構造で確認します。
"""

import shutil
import tempfile
from pathlib import Path


class MiniShell:
    """本物のシェルの核心動作(現在位置の管理 + ファイルコマンド)を真似たクラス。"""

    def __init__(self, root: Path):
        # resolve(): シンボリックリンクまで解決した絶対パス (Mac の /var -> /private/var 対策)
        self.root = root.resolve()   # 練習場の最上位 (この外へは出られない)
        self.cwd = self.root      # カレントワーキングディレクトリ — シェルの核心状態!

    def run(self, line: str) -> str:
        """'cp a.txt backup' のような1行を解釈し、該当メソッドを呼び出します。"""
        parts = line.split()
        command, args = parts[0], parts[1:]
        handler = getattr(self, f"cmd_{command}", None)  # cmd_ls, cmd_cd ...
        if handler is None:
            return f"(エラー) '{command}': コマンドが見つかりません"
        return handler(*args)

    # ---- 各コマンドの実装: 結局は pathlib の関数呼び出しです ----
    def cmd_pwd(self):
        rel = self.cwd.relative_to(self.root)
        return "/" + str(rel) if str(rel) != "." else "/ (練習場の最上位)"

    def cmd_ls(self):
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in self.cwd.iterdir())
        return "  ".join(names) if names else "(空です)"

    def cmd_cd(self, name):
        target = (self.cwd / name).resolve()
        if not target.is_dir():
            return f"(エラー) '{name}': そのようなフォルダはありません"
        self.cwd = target  # 「移動」とは単に現在位置の変数を書き換えること!
        return f"移動完了 -> {self.cmd_pwd()}"

    def cmd_mkdir(self, name):
        (self.cwd / name).mkdir()
        return f"フォルダ '{name}' を作成"

    def cmd_cp(self, src, dst):
        dst_path = self.cwd / dst
        if dst_path.is_dir():
            dst_path = dst_path / src  # フォルダへコピーすると同じ名前で入る
        shutil.copy(self.cwd / src, dst_path)
        return f"'{src}' -> '{dst}' コピー (元のファイルはそのまま)"

    def cmd_mv(self, src, dst):
        dst_path = self.cwd / dst
        if dst_path.is_dir():
            dst_path = dst_path / src
        (self.cwd / src).rename(dst_path)
        return f"'{src}' -> '{dst}' 移動 (同じフォルダ内なら名前の変更)"

    def cmd_rm(self, name):
        (self.cwd / name).unlink()
        return f"'{name}' を削除 — ごみ箱を経由せず即座に消えました!"


# 練習シナリオ: (コマンド, このコマンドがやることの解説)
SCENARIO = [
    ("pwd", "いま自分がどこに立っているかを確認 — ターミナル作業の第一の習慣"),
    ("ls", "この部屋(フォルダ)に何があるかを確認"),
    ("mkdir backup", "バックアップ用のキャビネット(フォルダ)を新設"),
    ("cp 週次報告.txt backup", "報告書のコピーを backup フォルダに保管"),
    ("cd backup", "backup フォルダの中へ移動"),
    ("ls", "コピーがちゃんと入ったか確認"),
    ("mv 週次報告.txt 週次報告_バックアップ.txt", "同じフォルダ内の mv = 名前の変更"),
    ("cd ..", "'..' = ひとつ上の階(親フォルダ)へ"),
    ("rm 臨時メモ.txt", "不要になったメモを削除 (即時・永久!)"),
    ("ls", "整理が終わった部屋の最終状態を確認"),
]


def main():
    print("=" * 60)
    print("ターミナルの第一歩 — ミニシェルシミュレータ")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        office = Path(tmp) / "practice_office"
        office.mkdir()

        # [1] 練習場の準備: サンプルファイルを置いておきます (一時フォルダなので失敗しても安全)
        for name, content in [
            ("週次報告.txt", "今週の売上サマリー"),
            ("議事録.txt", "9月の定例会議"),
            ("臨時メモ.txt", "ランチのメニュー候補"),
        ]:
            (office / name).write_text(content, encoding="utf-8")
        print(f"\n[1] 練習場の準備完了: 一時フォルダにサンプルファイルを3個作成")
        print("    (皆さんのコンピュータの実際のファイルには一切触れません)")

        # [2] シナリオ実行: コマンド1行ずつ「入力 -> 実行 -> 解説」
        shell = MiniShell(office)
        print("\n[2] コマンドシナリオの実行 ($ マークが「入力したコマンド」です)")
        for step, (line, note) in enumerate(SCENARIO, start=1):
            print(f"\n  ({step}) $ {line}")
            print(f"      -> {shell.run(line)}")
            print(f"      解説: {note}")

        # [3] 最終状態とまとめ
        n_ops = len(SCENARIO)
        print("\n[3] 最終フォルダ構造:")
        for p in sorted(office.rglob("*")):
            depth = len(p.relative_to(office).parts) - 1
            tag = "/" if p.is_dir() else ""
            print("      " + "  " * depth + f"- {p.name}{tag}")
        print(f"\n    コマンド {n_ops}行 で片付きました。マウスならウィンドウを開いてドラッグして名前をクリックして…")
        print("    しかもこの {n}行 は、保存しておけば明日も、後任者も、そのまま再実行できます。".format(n=n_ops))

    print("\nまとめ: シェル = 「現在位置」を記憶しながらファイル作業を代行する通訳者。")
    print("        さあ本物のターミナルを開いて、pwd、ls、cd から自分で打ってみましょう!")


if __name__ == "__main__":
    main()
