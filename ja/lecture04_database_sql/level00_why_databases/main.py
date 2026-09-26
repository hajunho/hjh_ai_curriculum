"""
Lecture 04 · Level 00 — データベースはなぜ必要なのか
「Excel ファイルの使い回し」が起こす三つの事故 (同時更新の消失・整合性の汚染・
権限の不在) を Python シミュレーションで再現し、同じシナリオをデータベース
(sqlite3) 方式でやり直して、何が変わるのかを対比します。SQL の文法は次の
レベルから学ぶので、ここでは出力結果の「違い」だけに注目すれば大丈夫です。
"""

import copy
import pathlib
import sqlite3
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data  # 共用データジェネレーター (実習用ショップ DB を作ってくれます)


def show_rows(title, rows):
    """辞書のリストを簡単な表として出力します。"""
    print(f"  {title}")
    for r in rows:
        print("   ", " | ".join(f"{k}={v}" for k, v in r.items()))


def excel_style_disaster():
    """[1] 同時更新: 二人が同じファイルのコピーを直して順番に保存すると?"""
    print("[1] Excel 方式 — 同時更新の事故 (更新の消失)")
    shared_file = [  # 共有フォルダの「顧客リスト.xlsx」だと想像してください
        {"顧客番号": 1, "氏名": "佐藤翔太", "グレード": "SILVER"},
        {"顧客番号": 2, "氏名": "鈴木陽菜", "グレード": "BASIC"},
    ]
    # 二人がそれぞれファイルを「ダウンロード」(コピーを作成)
    kim_copy = copy.deepcopy(shared_file)
    park_copy = copy.deepcopy(shared_file)

    kim_copy[0]["グレード"] = "VIP"      # 高橋さん: 1番の顧客を VIP に昇格
    park_copy[1]["グレード"] = "GOLD"    # 田中課長: 2番の顧客を GOLD に昇格

    shared_file = kim_copy    # 高橋さんが先に保存 (ファイルを丸ごと上書き)
    shared_file = park_copy   # 田中課長が後から保存 → 高橋さんの作業が消えた!

    show_rows("最終的に保存されたファイル:", shared_file)
    print("  → 1番の顧客がまだ SILVER のまま。高橋さんの昇格作業が『エラーもなく』蒸発しました。\n")


def integrity_disaster():
    """[2] 整合性: 同じ取引先が表記ゆれで二重に登録された帳簿。"""
    print("[2] Excel 方式 — 整合性の汚染 (同じ取引先、違う表記)")
    dirty_ledger = [
        {"取引先": "(株)ひかり", "金額": 300},
        {"取引先": "ひかり株式会社", "金額": 200},   # 実は上と同じ会社
        {"取引先": "みらい商事", "金額": 150},
    ]
    totals = {}
    for row in dirty_ledger:
        totals[row["取引先"]] = totals.get(row["取引先"], 0) + row["金額"]
    for name, amount in totals.items():
        print(f"    {name}: {amount}万ウォン")
    print("  → ひかりは本当は 500万ウォンの取引先なのに 300/200 に分裂し、レポートが狂います。\n")


def permission_disaster():
    """[3] 権限: ファイルを送った瞬間、機密の列まで丸ごと渡ってしまいます。"""
    print("[3] Excel 方式 — 権限の不在 (ファイル送付 = 全部流出)")
    hr_file = [
        {"氏名": "高橋美咲", "部署": "営業", "年俸": 9000},
        {"氏名": "伊藤大輝", "部署": "開発", "年俸": 7200},
    ]
    print("  「部署の状況だけ参考にしてください」とファイルを添付すると、受け取った人の画面には:")
    show_rows("送られたファイルの内容:", hr_file)
    print("  → 年俸の列を外す方法は『ファイルを作り直すこと』だけ。原本のコントロールが不可能です。\n")


def database_way():
    """[4] DB 方式: 一か所の原本 + 受付窓口。同じシナリオをやり直します。"""
    print("[4] データベース方式 — 同じシナリオ、違う結果")
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))   # 実習用ショップ DB を生成 (インストール不要)
    print(f"  実習 DB を生成: {db_path.name} (customers など 5 テーブル)")

    # 二つの接続 = 二人のユーザー。原本は一つで、それぞれ「リクエスト」を送るだけです。
    kim = sqlite3.connect(db_path)
    park = sqlite3.connect(db_path)

    kim.execute("UPDATE customers SET grade='VIP' WHERE customer_id=1")
    kim.commit()      # 高橋さん: 1番の顧客を昇格するようリクエスト → 窓口が原本に記録
    park.execute("UPDATE customers SET grade='GOLD' WHERE customer_id=2")
    park.commit()     # 田中課長: 2番の顧客を昇格するようリクエスト → こちらも原本に記録

    cur = kim.execute(
        "SELECT customer_id, name, grade FROM customers WHERE customer_id IN (1, 2)")
    rows = [{"顧客番号": r[0], "氏名": r[1], "グレード": r[2]} for r in cur.fetchall()]
    show_rows("原本テーブルの現在の状態:", rows)
    print("  → 二人の更新が両方とも残りました。原本が一つなので上書き事故が起きません。")

    # 整合性: ルール (制約条件) に反するデータは、保存そのものが拒否されます。
    try:
        kim.execute("INSERT INTO customers (customer_id, name) VALUES (1, '幽霊顧客')")
    except sqlite3.IntegrityError as e:
        print(f"  重複した顧客番号を保存しようとすると → DB が拒否: {e}")

    # 権限: 機密の列を除いた「閲覧専用の窓 (ビュー)」だけを開放できます。
    kim.execute("CREATE VIEW IF NOT EXISTS emp_public AS "
                "SELECT name, dept FROM employees")   # salary 列はそもそも除外
    cur = kim.execute("SELECT * FROM emp_public LIMIT 2")
    rows = [{"氏名": r[0], "部署": r[1]} for r in cur.fetchall()]
    show_rows("共有用の窓口 (ビュー) から見える内容:", rows)
    print("  → 年俸の列は窓口に存在しないので、流出そのものが不可能です。\n")

    kim.close()
    park.close()


def main():
    print("=" * 62)
    print(" Excel 使い回しの三つの地獄 vs データベース")
    print("=" * 62 + "\n")
    excel_style_disaster()
    integrity_disaster()
    permission_disaster()
    database_way()
    print("[5] まとめ")
    print("  Excel 方式: コピーが何個もある → 事故が起きても『エラーすら』出ない。")
    print("  DB 方式   : 原本一つ + 受付窓口 → 順序の保証、ルール検査、権限管理。")
    print("  次のレベルから、この窓口に話しかける言語 SQL を学びます。")


if __name__ == "__main__":
    main()
