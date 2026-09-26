"""モジュールとパッケージ — 自分のモジュール (utils.py) の import と標準ライブラリツアー。

同じフォルダの utils.py (共用関数モジュール) を import して使う方法と、
Python 標準の「専門部署」4つ — datetime / math / random / collections —
の代表機能を、実務の例題で見て回ります。
"""

import datetime
import math
import random
from collections import Counter, defaultdict

import utils                     # 同じフォルダの utils.py — ファイル1つがモジュール1つ


def main():
    print("=" * 56)
    print(" モジュールとパッケージ — import と標準ライブラリツアー")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 自分のモジュールの import: utils.py の共用関数を使う
    # ---------------------------------------------------------
    print("\n[1] 自分のモジュール utils を使う (モジュール.関数 = 部署.担当者)")

    monthly_sales = [1200000, 980000, 1450000, 730000, 2100000]
    total = sum(monthly_sales)
    vat = utils.calc_vat(total)

    print(f"  5店舗の売上合計     : {utils.format_krw(total)}")
    print(f"  付加価値税 (utils.calc_vat): {utils.format_krw(vat)}")
    print(f"  モジュール定数へのアクセス : utils.VAT_RATE = {utils.VAT_RATE}")
    print("  -> utils.py の自己デモは実行されませんでした (__name__ による区別のおかげ)。")

    # ---------------------------------------------------------
    # [2] datetime 部署: 日付計算
    # ---------------------------------------------------------
    print("\n[2] datetime — 日付・時刻の計算")

    base_day = datetime.date(2026, 9, 26)          # 基準日を固定 (出力の再現性)
    deadline = datetime.date(2026, 12, 31)
    d_day = (deadline - base_day).days             # 日付同士を引くと期間が出る

    print(f"  基準日          : {base_day} ({['月','火','水','木','金','土','日'][base_day.weekday()]}曜日)")
    print(f"  年末の締切まで  : あと{d_day}日")
    print(f"  書式化          : {base_day.strftime('%Y年 %m月 %d日')}")
    ship_day = utils.add_business_days(base_day, 3)
    print(f"  営業日3日後の納期 (週末を除く、utils の関数): {ship_day}")

    # ---------------------------------------------------------
    # [3] math 部署: 切り上げが必要な瞬間
    # ---------------------------------------------------------
    print("\n[3] math — 配車計画に切り上げ (ceil) が必要な理由")

    people = 17
    van_capacity = 5
    exact = people / van_capacity
    vans = math.ceil(exact)                        # 3.4台は存在しない。4台が必要。
    print(f"  17人 / 5人乗り = {exact}  ->  math.ceil() = {vans}台を配車")
    print(f"  math.floor(3.4) = {math.floor(3.4)} / math.sqrt(2) = {math.sqrt(2):.4f}"
          f" / math.pi = {math.pi:.4f}")

    # ---------------------------------------------------------
    # [4] random 部署: 抽選とシャッフル (seed 固定で再現性を確保)
    # ---------------------------------------------------------
    print("\n[4] random — 景品の抽選と当番の順番 (seed=42 固定)")

    random.seed(42)                                # seed 固定: 実行するたびに同じ結果
    staff = ["田中主任", "佐藤課長", "鈴木係長", "高橋部長", "伊藤社員", "渡辺係長"]

    winners = random.sample(staff, 2)              # 重複なしで2人を抽選
    print(f"  景品の当選者2人 : {winners}")

    rotation = staff.copy()
    random.shuffle(rotation)                       # 元を保存するため copy してからシャッフル
    print(f"  当直の順番      : {rotation}")
    print(f"  サイコロ1回     : {random.randint(1, 6)}")

    # ---------------------------------------------------------
    # [5] collections 部署: Counter と defaultdict
    # ---------------------------------------------------------
    print("\n[5] collections — 集計の専門ツール")

    orders = ["アメリカーノ", "ラテ", "アメリカーノ", "ティー", "ラテ", "アメリカーノ", "ティー", "アメリカーノ"]
    counter = Counter(orders)                      # 頻度集計が1行
    print(f"  注文頻度 Counter : {dict(counter)}")
    print(f"  最多注文の1位    : {counter.most_common(1)[0][0]} ({counter.most_common(1)[0][1]}件)")

    branch_sales = [("渋谷店", 120), ("新宿店", 80), ("渋谷店", 200), ("大阪店", 150), ("新宿店", 90)]
    by_branch = defaultdict(int)                   # 存在しないキーは自動的に0から始まる
    for branch, amount in branch_sales:
        by_branch[branch] += amount                # get(キー, 0) なしで直接累積
    print(f"  店舗別売上 defaultdict: {dict(by_branch)}")

    print("\n[終] 必要な機能はまず標準ライブラリから探し、")
    print("     チーム共用の手順は utils のようなモジュールにまとめて再利用しましょう。")


if __name__ == "__main__":
    main()
