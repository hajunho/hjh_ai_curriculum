"""チーム共用ユーティリティモジュール。

複数のスクリプトで繰り返し使う手順 (通貨の書式化、付加価値税、営業日計算) を
1つのファイルにまとめた「業務マニュアルバインダー」です。
main.py から `import utils` で呼び込んで使います。
"""

import datetime

VAT_RATE = 0.1          # モジュールレベルの定数: import した側から utils.VAT_RATE でアクセス可能

# import された瞬間、モジュールのコードは上から下へ「1回」実行される。
# この print は、その事実を目で確認するための教育用の表示である。
print("  (utils モジュールがロードされました — この行は import 時にちょうど1回実行)")


def format_krw(amount):
    """金額を「1,234,567ウォン」形式の韓国ウォン (KRW) 文字列に書式化する。"""
    return f"{amount:,}ウォン"


def calc_vat(amount):
    """付加価値税 (10%) をウォン単位の整数で計算する。"""
    return int(amount * VAT_RATE)


def add_business_days(start_date, days):
    """週末 (土・日) を飛ばして、営業日基準で days 日後の日付を返す。"""
    current = start_date
    remaining = days
    while remaining > 0:
        current += datetime.timedelta(days=1)
        if current.weekday() < 5:        # 0=月 ... 4=金 / 5=土, 6=日は除外
            remaining -= 1
    return current


if __name__ == "__main__":
    # このブロックは `python3 utils.py` で「直接実行」したときだけ動く。
    # main.py が import するときは実行されない — 定義と実行の分離。
    print("[utils 自己デモ] 直接実行したときだけ見える出力です")
    print(f"  format_krw(1234567) = {format_krw(1234567)}")
    print(f"  calc_vat(50000)     = {calc_vat(50000)}")
    demo_day = datetime.date(2026, 9, 25)   # 金曜日
    print(f"  {demo_day}(金) + 営業日3日 = {add_business_days(demo_day, 3)}")
