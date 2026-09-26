"""
Lecture 01 / Level 06 — Git: バージョン管理の考え方

一時フォルダに本物の Git リポジトリを作って init -> add -> commit -> diff ->
log -> branch の流れを自動実習します。コミット=決裁のハンコ、ブランチ=並行宇宙
というたとえを、実際のコマンド出力で確認します。
git がインストールされていない場合は、インストール案内の後に概念シミュレーションで代替します。
"""

import shutil
import subprocess
import tempfile
from pathlib import Path


def run_git(args, cwd, note=""):
    """git コマンドを実行し、「$ コマンド -> 出力」の形で見せます。"""
    print(f"\n  $ git {' '.join(args)}")
    if note:
        print(f"    解説: {note}")
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines()[:12]:  # 長すぎる場合は先頭だけ
        print(f"    | {line}")
    return output


def real_git_demo():
    """git がインストールされている場合: 一時フォルダで実際のコマンドによる全体の流れの実習。"""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "cafe_project"
        repo.mkdir()
        price_file = repo / "価格表.txt"

        # [1] 書類棚の設置: init + 実習用の身元登録(このリポジトリ内でのみ有効)
        print("\n[1] 書類棚の設置 — git init")
        run_git(["init", "-b", "main"], repo, "このフォルダの履歴管理を開始(.git 書類棚を作成)")
        run_git(["config", "user.name", "実習生"], repo, "ハンコに刻む名前(実習用、このリポジトリ限定)")
        run_git(["config", "user.email", "student@example.com"], repo, "ハンコに刻むメール")

        # [2] 最初の決裁: ファイル作成 -> add(決裁板) -> commit(ハンコ)
        print("\n[2] 最初の決裁 — add と commit")
        price_file.write_text("アメリカーノ 4000ウォン\nラテ 4500ウォン\n", encoding="utf-8")
        print(f"    (ファイル作成: {price_file.name})")
        run_git(["status", "--short"], repo, "?? = まだ書類棚が知らない新しいファイル")
        run_git(["add", "価格表.txt"], repo, "決裁板(ステージング)に載せる")
        run_git(["commit", "-m", "価格表の草案を作成"], repo, "ハンコをドン! この瞬間が永久保存される")

        # [3] 修正と diff: 何が変わったかを確認してから2つ目のコミット
        print("\n[3] 修正と diff — 変わった箇所の確認")
        price_file.write_text("アメリカーノ 4200ウォン\nラテ 4500ウォン\n", encoding="utf-8")
        print("    (アメリカーノの価格を 4000 -> 4200 に修正)")
        run_git(["diff"], repo, "- が古い内容、+ が新しい内容")
        run_git(["add", "."], repo)
        run_git(["commit", "-m", "アメリカーノの値上げ(4000->4200)"], repo, "メッセージに「なぜ」を書くのがコツ")

        # [4] 履歴の照会: ハンコが積み重なった決裁記録
        print("\n[4] 履歴の照会 — git log")
        run_git(["log", "--oneline"], repo, "先頭の短いコードがコミットの通し番号(ハッシュ)")

        # [5] ブランチ: 並行宇宙で実験してから本編に復帰
        print("\n[5] ブランチ — 並行宇宙の実験")
        run_git(["switch", "-c", "experiment"], repo, "experiment 宇宙を作って移動")
        price_file.write_text("アメリカーノ 9900ウォン (実験!)\nラテ 4500ウォン\n", encoding="utf-8")
        run_git(["add", "."], repo)
        run_git(["commit", "-m", "実験: プレミアム価格ポリシー"], repo)
        print(f"    experiment 宇宙の価格表: {price_file.read_text(encoding='utf-8').splitlines()[0]}")
        run_git(["switch", "main"], repo, "本編(main)に復帰 — タイムトラベル!")
        print(f"    main 宇宙の価格表      : {price_file.read_text(encoding='utf-8').splitlines()[0]}")
        print("    -> 同じファイルなのに、ブランチを切り替えたら内容が元に戻りました。")
        print("       実験が失敗したら experiment 宇宙だけ捨てれば、本編は無事です。")


def fallback_simulation():
    """git が無い場合: インストール案内 + コミット概念のシミュレーション。"""
    print("\n[案内] git コマンドが見つかりませんでした。")
    print("  インストール: Mac は 'xcode-select --install' または 'brew install git'、")
    print("        Windows は git-scm.com の Git for Windows のインストールをおすすめします。")
    print("  インストール後にこのスクリプトを再実行すると、実際のコマンドで実習できます。")
    print("\n[概念シミュレーション] コミット = フォルダ状態のスナップショット + 決裁のハンコ")
    history = [
        ("a1f9c02", "価格表の草案を作成", {"価格表.txt": "アメリカーノ 4000ウォン"}),
        ("b7e3d11", "アメリカーノの値上げ(4000->4200)", {"価格表.txt": "アメリカーノ 4200ウォン"}),
    ]
    for i, (commit_hash, message, snapshot) in enumerate(history, start=1):
        print(f"\n  コミット {i}: [{commit_hash}] \"{message}\"")
        for fname, content in snapshot.items():
            print(f"    保管されたスナップショット: {fname} -> '{content}'")
    print("\n  どのハンコの時点にもフォルダを戻せるのが Git の力です。")


def main():
    print("=" * 60)
    print("Git 実習 — 決裁のハンコ(コミット)と並行宇宙(ブランチ)")
    print("=" * 60)
    if shutil.which("git"):
        print("\n(git を発見 — 一時フォルダに本物のリポジトリを作って実習します。")
        print(" 皆さんの他のフォルダ・リポジトリには一切触れません。)")
        real_git_demo()
    else:
        fallback_simulation()

    print("\nまとめ: 小さく、こまめにコミットして、メッセージには「なぜ」を書きましょう。")
    print("        コミットしておいたものは、どんな失敗をしても取り戻せます。")


if __name__ == "__main__":
    main()
