"""
Lecture 01 / Level 03 — Python のインストールと初めての実行

自分のコンピュータの Python 環境を自分で診断する「健康診断」スクリプトです。
バージョン・実行ファイルの場所・OS・PATH を確認し、
REPL(対話型)とスクリプト実行の違いを出力で比較します。
"""

import os
import platform
import sys

# このカリキュラムが要求する最小 Python バージョン (「やってみよう」の課題2で変えてみてください)
REQUIRED = (3, 10)


def check_python_identity():
    """[1] 通訳者の身元確認: バージョンと実行ファイルの場所。"""
    version = sys.version_info  # (major, minor, micro, ...) 数値として比較できる
    ok = version >= REQUIRED
    print("[1] Python の身元確認")
    print(f"    バージョン        : {platform.python_version()}")
    print(f"    実行ファイルの場所: {sys.executable}")
    print(f"    要求バージョン    : {REQUIRED[0]}.{REQUIRED[1]} 以上")
    print(f"    判定              : {'合格 — カリキュラムを進められます' if ok else '不足 — アップグレードが必要です'}")
    return ok


def check_workplace():
    """[2] 勤務地の確認: OS とカレントワーキングディレクトリ。"""
    os_name = platform.system()  # 'Darwin'(Mac) / 'Windows' / 'Linux'
    friendly = {"Darwin": "macOS(Mac)", "Windows": "Windows", "Linux": "Linux"}.get(os_name, os_name)
    print("\n[2] 勤務環境の確認")
    print(f"    OS                : {friendly} ({platform.release()})")
    print(f"    プロセッサ        : {platform.machine()}")
    print(f"    現在の作業フォルダ: {os.getcwd()}")
    print("    -> 相対パスはすべてこのフォルダを基準に解釈されます (level01 の復習)")
    return True


def check_path():
    """[3] PATH をざっと見る: シェルがコマンドを探すフォルダの一覧。"""
    raw = os.environ.get("PATH", "")
    entries = [e for e in raw.split(os.pathsep) if e]
    print("\n[3] PATH — シェルが 'python3' のようなコマンドを探し回るフォルダの一覧")
    print(f"    登録フォルダ数    : {len(entries)}個 (先頭から順に探索)")
    for i, entry in enumerate(entries[:5], start=1):
        print(f"    第{i}順位: {entry}")
    if len(entries) > 5:
        print(f"    ... ほか {len(entries) - 5}個")
    print("    -> 'command not found' = この一覧のどこにもその名前が無いという意味")
    return len(entries) > 0


def compare_repl_vs_script():
    """[4] REPL(その場の会話) vs スクリプト(文書での指示) の比較。"""
    print("\n[4] 2つの実行方式の比較")
    print("    (a) REPL — ターミナルに python3 とだけ打つと現れる '>>>' でのその場の会話:")
    print("        >>> 120 * 12")
    print(f"        {120 * 12}")
    print("        >>> '報告書' + '_最終.xlsx'")
    print(f"        '{'報告書' + '_最終.xlsx'}'")
    print("    (b) スクリプト — まさにいま! この出力全体が、main.py という文書を")
    print("        python3 に丸ごと渡して、上から順に実行した結果です。")
    print("    -> 実験は REPL、正式な業務はスクリプト(保管・共有・再実行が可能)")
    return True


def main():
    print("=" * 60)
    print("Python 環境の自己診断 — 私の通訳者は準備できているか")
    print("=" * 60 + "\n")

    results = {
        "Python バージョン": check_python_identity(),
        "OS の確認": check_workplace(),
        "PATH の設定": check_path(),
        "実行方式の理解": compare_repl_vs_script(),
    }

    # [5] 総合判定表
    print("\n[5] 総合診断サマリー")
    for item, ok in results.items():
        mark = "[OK] " if ok else "[注意]"
        print(f"    {mark} {item}")
    if all(results.values()):
        print("\n    おめでとうございます! このコンピュータはカリキュラムを進める準備ができています。")
    else:
        print("\n    [注意] の項目を解決してから、もう一度実行してみてください。")

    print("\nまとめ: 環境がおかしいときは、いつも sys.executable(どの Python か)と")
    print("        python3 --version(何バージョンか)から確認する習慣をつけましょう。")


if __name__ == "__main__":
    main()
