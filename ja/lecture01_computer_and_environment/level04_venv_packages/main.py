"""
Lecture 01 / Level 04 — 仮想環境とパッケージ管理

いまこのコードを実行している Python 環境を診断します。
sys.prefix と sys.base_prefix の比較で仮想環境かどうかを判定し、
importlib.metadata でインストール済みパッケージ(道具箱の中身)を並べ、
pip freeze が作る requirements.txt の正体を見せます。
"""

import importlib.metadata
import importlib.util
import sys


def check_which_toolbox():
    """[1] どの道具箱か: 仮想環境の中かシステム Python かを判定。"""
    prefix = sys.prefix            # いまの Python が道具を探す空間
    base = sys.base_prefix         # 元の(システム)Python の空間
    in_venv = prefix != base       # 2つが違えば = 専用の道具箱を着けている
    print("[1] どの道具箱を着けているか")
    print(f"    sys.prefix      (いま使う空間): {prefix}")
    print(f"    sys.base_prefix (元の空間)    : {base}")
    if in_venv:
        print("    判定: 仮想環境の中です — プロジェクト専用の道具箱を着用中")
    else:
        print("    判定: システム Python です — 会社共用の工具箱を使用中")
        print("    (リポジトリの .venv/bin/python で実行すると判定が変わります)")
    print(f"    実行ファイル: {sys.executable}")
    return in_venv


def list_installed_packages(limit=15):
    """[2] 道具箱の中身: インストール済みパッケージの名前とバージョンを並べる。"""
    dists = sorted(
        ((d.metadata["Name"] or "?", d.version) for d in importlib.metadata.distributions()),
        key=lambda pair: pair[0].lower(),
    )
    print(f"\n[2] この環境にインストールされたパッケージ: 計 {len(dists)}個 (pip list と同じ情報)")
    for name, version in dists[:limit]:
        print(f"    - {name:<28} {version}")
    if len(dists) > limit:
        print(f"    ... ほか {len(dists) - limit}個")
    return dists


def check_standard_tools():
    """[3] 基本の工具(標準ライブラリ)の確認: インストールなしですでにある道具たち。"""
    basics = ["json", "csv", "datetime", "sqlite3", "pathlib", "random"]
    print("\n[3] 基本の工具の確認 — 標準ライブラリはインストールなしですぐ使えます")
    for name in basics:
        found = importlib.util.find_spec(name) is not None
        print(f"    {'[OK]' if found else '[なし]'} {name}")
    print("    -> このカリキュラムの lecture01〜02 は、すべて基本の工具だけで進みます")


def preview_requirements(dists, limit=8):
    """[4] requirements.txt のプレビュー: pip freeze が作る物品リスト。"""
    print("\n[4] requirements.txt のプレビュー — 「名前==バージョン」形式の物品リスト")
    if not dists:
        print("    (インストール済みパッケージが無いため、リストは空です)")
        return
    for name, version in dists[:limit]:
        print(f"    {name}=={version}")
    if len(dists) > limit:
        print(f"    ... ほか {len(dists) - limit}行")
    print("    -> このファイルさえ渡せば、同僚は 'pip install -r requirements.txt' で")
    print("       同じ構成の道具箱を自分のコンピュータにそのまま再現できます")


def main():
    print("=" * 60)
    print("仮想環境・パッケージ診断 — 私はいまどの道具箱を着けているか")
    print("=" * 60 + "\n")

    in_venv = check_which_toolbox()
    dists = list_installed_packages()
    check_standard_tools()
    preview_requirements(dists)

    # [5] まとめ
    print("\n[5] 診断サマリー")
    print(f"    仮想環境かどうか    : {'はい (.venv 系)' if in_venv else 'いいえ (システム Python)'}")
    print(f"    インストール済み    : {len(dists)}個")
    print("\nまとめ: プロジェクトごとに専用の道具箱(venv)を作り、")
    print("        pip freeze のリスト(requirements.txt)で構成を共有しましょう。")
    print("        「自分のパソコンでは動くんですけど」トラブルの半分がここで予防できます。")


if __name__ == "__main__":
    main()
