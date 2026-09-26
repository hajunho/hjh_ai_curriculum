"""
Lecture 01 / Level 07 — GitHub: リモートリポジトリと共同作業

インターネットなしでリモート共同作業を実習します。一時フォルダに bare リポジトリ
(本社の文書庫)を作って「偽物の GitHub」に見立て、開発者 A・B が
clone -> push -> pull で書類をやり取りする共同作業シナリオを自動実行します。
git が無ければ、インストール案内の後に概念シミュレーションで代替します。
"""

import shutil
import subprocess
import tempfile
from pathlib import Path


def run(args, cwd, actor="", note=""):
    """git コマンドを実行し、「誰が($ の前の名前)何をしたのか」を出力します。"""
    label = f"[{actor}] " if actor else ""
    print(f"\n  {label}$ git {' '.join(args)}")
    if note:
        print(f"      解説: {note}")
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines()[:8]:
        print(f"      | {line}")
    return output


def setup_identity(repo, name):
    """実習用のコミット身元登録 (このリポジトリ内でのみ有効)。"""
    run(["config", "user.name", name], repo)
    run(["config", "user.email", f"{name}@example.com"], repo)


def list_files(repo, actor):
    """支社の書類棚(作業フォルダ)のファイル一覧を見せます。"""
    files = sorted(p.name for p in Path(repo).iterdir() if p.is_file())
    print(f"      [{actor}] 支社フォルダの内容: {files if files else '(空です)'}")


def real_remote_demo():
    """bare リモート + 2つの支社(A・B)で push/pull 共同作業の流れを実習。"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        hq = base / "headquarters.git"   # 本社の中央文書庫 (bare)
        dev_a = base / "dev_a"           # 開発者 A の支社
        dev_b = base / "dev_b"           # 開発者 B の支社

        # [1] 本社の設立: bare リポジトリ = 作業デスクなしで履歴だけ保管 (GitHub サーバーの実際の形)
        print("\n[1] 本社の設立 — bare リモートリポジトリを作る")
        run(["init", "--bare", "-b", "main", str(hq)], base,
            note="--bare: 作業フォルダの無い中央文書庫。これが私たちの「偽物の GitHub」")

        # [2] 支社の開設: A と B がそれぞれ clone
        print("\n[2] 支社の開設 — 開発者 A・B が clone")
        run(["clone", str(hq), str(dev_a)], base, "A", "本社の文書庫を丸ごと複製(まだ空です)")
        run(["clone", str(hq), str(dev_b)], base, "B", "B も自分の支社を開設")
        setup_identity(dev_a, "dev-a")
        setup_identity(dev_b, "dev-b")

        # [3] A の push: ファイル作成 -> コミット -> 本社へ送り上げる
        print("\n[3] A の作業と push")
        (dev_a / "menu.txt").write_text("アメリカーノ 4000ウォン\n", encoding="utf-8")
        run(["add", "menu.txt"], dev_a, "A")
        run(["commit", "-m", "メニュー表の草案"], dev_a, "A", "支社の決裁(コミット)はまだ A のコンピュータの中だけに存在")
        run(["push", "origin", "main"], dev_a, "A", "push = 決裁書類を本社の文書庫へ送付")

        # [4] B の pull: 本社の最新書類を受け取る — 共同作業の核心の瞬間
        print("\n[4] B の pull — A の作業が B に届く")
        list_files(dev_b, "B(pull 前)")
        run(["pull", "origin", "main"], dev_b, "B", "pull = 本社の新しい書類を受け取って合流させる")
        list_files(dev_b, "B(pull 後)")
        print("      -> A が作った menu.txt が B の支社に現れました!")

        # [5] やり取り: B が追加 -> push、A が pull で最新化
        print("\n[5] 逆方向 — B が追加して A が受け取る")
        menu_b = dev_b / "menu.txt"
        menu_b.write_text(menu_b.read_text(encoding="utf-8") + "ラテ 4500ウォン\n", encoding="utf-8")
        run(["add", "."], dev_b, "B")
        run(["commit", "-m", "ラテを追加"], dev_b, "B")
        run(["push", "origin", "main"], dev_b, "B")
        run(["pull", "origin", "main"], dev_a, "A", "pull が先、push は後 — 共同作業のマナー")
        print(f"      [A] 最終メニュー表: {(dev_a / 'menu.txt').read_text(encoding='utf-8').splitlines()}")
        print("      -> 両方の支社が同じ最新状態になりました。")

        # [6] 実際の GitHub なら: PR の流れの解説
        print("\n[6] 実際の GitHub の共同作業なら、この流れに「レビュー」が挟まります:")
        for step in [
            "1) ブランチを作って作業してから push",
            "2) GitHub の Web で Pull Request(稟議書)を作成",
            "3) 同僚がレビュー — コメント・修正依頼が行き交う",
            "4) 承認されたら main に merge (本編に合流)",
            "5) チーム全員が git pull で最新化",
        ]:
            print(f"      {step}")


def fallback_simulation():
    """git が無い場合: 概念だけテキストでシミュレーション。"""
    print("\n[案内] git コマンドが見つかりませんでした。")
    print("  Mac: 'xcode-select --install' / Windows: git-scm.com からインストール後に再実行してください。")
    print("\n[概念シミュレーション] リモート共同作業の一日:")
    events = [
        ("本社", "中央文書庫を開設 (bare リポジトリ)"),
        ("A", "clone — 文書庫全体の複製で支社を開設"),
        ("A", "コミット後 push — 決裁書類を本社へ送付"),
        ("B", "pull — 本社の新しい書類を受け取り支社を最新化"),
        ("B", "追加作業後 push、A は再び pull"),
    ]
    for actor, action in events:
        print(f"  [{actor:^4}] {action}")
    print("\n  アドレスがインターネットの URL(GitHub)でもローカルフォルダでも、原理は同じです。")


def main():
    print("=" * 60)
    print("リモートリポジトリ共同作業の実習 — 本社(リモート)と2つの支社(A・B)")
    print("=" * 60)
    if shutil.which("git"):
        real_remote_demo()
    else:
        fallback_simulation()

    print("\nまとめ: clone で始めて、push で上げて、pull で受け取ります。")
    print("        pull が先・小さくこまめに push が、コンフリクトを減らす共同作業のマナーです。")


if __name__ == "__main__":
    main()
