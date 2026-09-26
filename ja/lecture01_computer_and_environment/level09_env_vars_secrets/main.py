"""
Lecture 01 / Level 09 — 環境変数とシークレットキーの安全な扱い方

秘密をコードの外で扱う3つの道具を実習します。
1) os.environ による環境変数の読み取り、2) 秘密の値のマスキング出力、
3) .env ファイルのパーサー(dotenv の原理)、4) ハードコーディングされた秘密を
捕まえるミニスキャナー。
すべてのキーの値は実習用の偽物で、インターネットには接続しません。
"""

import os
import re

# ハードコーディングされた秘密の検知ルール: (説明, 正規表現) — 実際のセキュリティツールの縮小版
SECRET_PATTERNS = [
    ("API キーの形 (sk-...)", re.compile(r"sk-[A-Za-z0-9]{8,}")),
    ("password= のハードコーディング", re.compile(r"password\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)),
    ("secret/token のハードコーディング", re.compile(r"(secret|token)\s*=\s*['\"][^'\"]{6,}['\"]", re.IGNORECASE)),
]

# [4]で検査する例のコード — 悪い例(秘密がコードの中)と良い例(名前だけ参照)
BAD_CODE = '''
# bad_app.py — こう書いてはいけません!
api_key = "sk-demo1234567890abcdef"
password = "corp!2024"
db = connect("10.0.0.7", password=password)
'''
GOOD_CODE = '''
# good_app.py — 秘密は名前だけで参照します
import os
api_key = os.environ["LLM_API_KEY"]      # 値は環境変数(ロッカー)に
password = os.environ.get("DB_PASSWORD")  # コードには名札だけ
'''


def mask(value, show=4):
    """秘密の値を 'sk-d****5678' の形に隠します。確認はしても露出は防ぐ。"""
    if value is None:
        return "(未設定)"
    if len(value) <= show * 2:
        return "*" * len(value)
    return value[:show] + "*" * 4 + value[-show:]


def parse_dotenv(text):
    """[3] .env パーサー: 「名前=値」形式のテキストを辞書に。dotenv という道具の原理。"""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):   # 空行とコメントは飛ばす
            continue
        name, _, value = line.partition("=")
        result[name.strip()] = value.strip().strip('"').strip("'")
    return result


def scan_for_secrets(code, filename):
    """[4] ミニスキャナー: コードの中から「秘密の形」をした文字列を探し出します。"""
    findings = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append((filename, lineno, label, line.strip()))
    return findings


def main():
    print("=" * 60)
    print("環境変数とシークレットキー — 秘密はコードの外に、コードには名前だけ")
    print("=" * 60)

    # [1] 環境変数の読み取り: 実習用の変数をこのプロセスに設定して読んでみます
    os.environ["DEMO_LLM_API_KEY"] = "sk-demo1122334455667788"  # 実習用の偽の値
    print("\n[1] 環境変数の読み取り — OS が手渡す「名前=値」のメモ紙")
    print(f"    os.environ['DEMO_LLM_API_KEY'] -> (読み取り成功、値は下でマスキング)")
    print(f"    .get() での安全な読み取り: 無い変数 -> {os.environ.get('NO_SUCH_VAR')}")
    print(f"    すでに使っていた環境変数 PATH の長さ: {len(os.environ.get('PATH', ''))}文字")
    practice = os.environ.get("PRACTICE_KEY")
    print(f"    PRACTICE_KEY: {mask(practice)}  (「やってみよう」の課題1で export してみてください)")

    # [2] マスキング出力: 「設定されているか」だけを見せて値は隠します
    print("\n[2] マスキング出力 — ログや画面に秘密を丸ごと出力しない")
    secret = os.environ["DEMO_LLM_API_KEY"]
    print(f"    元の値をそのまま出力: (絶対禁止!)")
    print(f"    マスキングして出力  : {mask(secret)}")

    # [3] .env パーサー: ファイルの内容を環境変数に載せる原理
    print("\n[3] .env パターン — プロジェクトフォルダの秘密保管ファイル (Git には絶対禁止)")
    dotenv_text = '# 実習用の .env の内容\nDB_PASSWORD="s3cret!pw"\nSLACK_TOKEN=xoxb-demo-9988\n'
    loaded = parse_dotenv(dotenv_text)
    for name, value in loaded.items():
        os.environ[name] = value           # 環境変数へ昇格 (dotenv がやっていること)
        print(f"    {name:<14} = {mask(value)}  -> os.environ に載せた")
    print("    -> コードはもう os.environ['DB_PASSWORD'] と名前を呼ぶだけで済みます")

    # [4] ハードコーディングスキャナー: 悪いコードでだけ警報が鳴るべきです
    print("\n[4] ハードコーディング検知ミニスキャナー — コミット前の自動検査の原理")
    for filename, code in [("bad_app.py", BAD_CODE), ("good_app.py", GOOD_CODE)]:
        findings = scan_for_secrets(code, filename)
        if findings:
            print(f"    {filename}: 警報 {len(findings)}件!")
            for fname, lineno, label, line in findings:
                print(f"      - {lineno}行目 [{label}] {line}")
        else:
            print(f"    {filename}: 合格 — ハードコーディングされた秘密なし")

    # [5] 掟のチェックリスト
    print("\n[5] 三重の防衛線チェックリスト")
    print("    [予防] 秘密は .env/環境変数だけに、.gitignore に .env を登録")
    print("    [検知] コミット前にスキャナーで「秘密の形」を検査")
    print("    [対応] 流出したらコミット削除ではなくキーの廃棄・再発行")

    print("\nまとめ: 玄関のメモ紙(コード)に金庫の番号(秘密)を書かないでください。")
    print("        マニュアルには「鍵はロッカーに」という名札だけを残すのです。")


if __name__ == "__main__":
    main()
