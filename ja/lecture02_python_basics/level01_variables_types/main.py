"""変数とデータ型 — カフェ売上計算の例題。

int/float/str/bool の4つの基本データ型を作ってみて、
変数で売上を計算し、データ型変換のミスを再現し、
f-string の書式で人が読みやすいレポートを出力します。
"""


def main():
    print("=" * 52)
    print(" 変数とデータ型 — カフェの1日の売上計算")
    print("=" * 52)

    # ---------------------------------------------------------
    # [1] 変数を作る: 名札付きの箱に値を入れる
    # ---------------------------------------------------------
    print("\n[1] 変数の作成とデータ型の確認")

    menu_name = "アメリカーノ"   # str  : 文字列
    unit_price = 4500            # int  : 整数 (ウォン単位の価格)
    cups_sold = 120              # int  : 販売した杯数
    vat_rate = 0.1               # float: 付加価値税率 10%
    is_open = True               # bool : 今日は営業中か

    print(f"  menu_name  = {menu_name!r:14} -> {type(menu_name).__name__}")
    print(f"  unit_price = {unit_price!r:14} -> {type(unit_price).__name__}")
    print(f"  cups_sold  = {cups_sold!r:14} -> {type(cups_sold).__name__}")
    print(f"  vat_rate   = {vat_rate!r:14} -> {type(vat_rate).__name__}")
    print(f"  is_open    = {is_open!r:14} -> {type(is_open).__name__}")

    # ---------------------------------------------------------
    # [2] 計算: '=' は「右側の値を左側の箱に入れよ」
    # ---------------------------------------------------------
    print("\n[2] 売上計算 (順次実行)")

    revenue = unit_price * cups_sold        # 売上 = 単価 x 数量
    vat = int(revenue * vat_rate)           # 付加価値税 (ウォン単位の整数に)
    total_with_vat = revenue + vat          # 税込み合計

    print(f"  売上            = {unit_price} x {cups_sold} = {revenue}ウォン")
    print(f"  付加価値税(10%) = {vat}ウォン")
    print(f"  合計            = {total_with_vat}ウォン")

    # count = count + 1 の形: 今の値に足して入れ直す
    cups_sold = cups_sold + 5               # 閉店直前に5杯追加販売
    print(f"  追加販売後 cups_sold = {cups_sold} (元の値に5を足して入れ直した)")

    # ---------------------------------------------------------
    # [3] データ型の変換: 実務事故ワースト1「文字の数字」
    # ---------------------------------------------------------
    print("\n[3] データ型の変換 — 文字列の数字の落とし穴")

    typed_price = "4500"        # CSV/入力から読んだ値はこのように文字列で入ってくる
    typed_qty = "2"

    wrong = typed_price * 2                  # 文字列 * 2 = 連結!
    print(f"  '4500' * 2          = {wrong!r}  <- 9000 ではなく文字列の繰り返し")

    right = int(typed_price) * int(typed_qty)  # 変換してから計算
    print(f"  int('4500')*int('2') = {right}   <- 変換すれば正しく計算できる")

    # ---------------------------------------------------------
    # [4] f-string の書式でレポートを作る
    # ---------------------------------------------------------
    print("\n[4] f-string 売上レポート")

    target = 600000                              # 今日の目標売上
    achieve_rate = total_with_vat / target       # 達成率

    print(f"  メニュー   : {menu_name}")
    print(f"  総売上     : {total_with_vat:,}ウォン (税込)")     # 3桁区切りのカンマ
    print(f"  目標       : {target:,}ウォン")
    print(f"  達成率     : {achieve_rate:.1%}")                  # パーセント小数1桁
    print(f"  1杯あたり  : {unit_price:,.0f}ウォン")             # 小数なしカンマ書式

    # ---------------------------------------------------------
    # [5] float の微小な誤差
    # ---------------------------------------------------------
    print("\n[5] float の誤差を確認")

    result = 0.1 + 0.2
    print(f"  0.1 + 0.2            = {result}  <- 正確に 0.3 ではない (2進法保存の限界)")
    print(f"  round(0.1 + 0.2, 2)  = {round(result, 2)}  <- 四捨五入で解決")
    print("  金額計算はできるだけ整数 (ウォン単位) で行うのが安全です。")

    print("\n[終] 変数 = 名札付きの箱、データ型 = 箱の中身の種類。")


if __name__ == "__main__":
    main()
