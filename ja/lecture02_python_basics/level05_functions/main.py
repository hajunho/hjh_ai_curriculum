"""関数 — 税金・割引計算のリファクタリング前後比較。

同じ決済計算 (ランク割引 + 付加価値税) を
[1] コピー&ペースト方式 (リファクタリング前、わざと仕込んだ不一致バグ入り) と
[2] 関数方式 (リファクタリング後) で実行し、結果を比較します。
デフォルト値引数、キーワード引数、複数の戻り値、スコープも合わせて実演します。
"""

VAT_RATE = 0.1          # 付加価値税率 10%


# =============================================================
# リファクタリング「後」: ロジックを関数1か所にだけ置く
# =============================================================
def get_discount_rate(grade):
    """顧客ランクに応じた割引率を返す。"""
    if grade == "VIP":
        return 0.10
    elif grade == "GOLD":
        return 0.05
    else:
        return 0.0


def calc_vat(amount):
    """付加価値税 (10%) をウォン単位の整数で計算する。税率変更時はここの1行だけ修正。"""
    return int(amount * VAT_RATE)


def calc_payment(amount, grade="一般"):
    """割引と付加価値税を適用した最終決済額を計算する。

    戻り値: (割引額, 付加価値税, 最終決済額) のタプル
    """
    discount = int(amount * get_discount_rate(grade))   # 関数が関数へ委任
    base = amount - discount
    vat = calc_vat(base)
    return discount, vat, base + vat


def main():
    print("=" * 56)
    print(" 関数 — 税金・割引計算のリファクタリング前後")
    print("=" * 56)

    # 注文データ: (顧客名, ランク, 注文額)
    orders = [
        ("田中主任", "VIP", 200000),
        ("佐藤課長", "GOLD", 150000),
        ("鈴木係長", "一般", 80000),
    ]

    # ---------------------------------------------------------
    # [1] リファクタリング前: 同じロジックを注文ごとにコピー&ペースト
    # ---------------------------------------------------------
    print("\n[1] リファクタリング前 — コピー&ペースト方式")

    before_results = []

    # --- 注文 1 (コピー 1)
    name1, grade1, amount1 = orders[0]
    if grade1 == "VIP":
        rate1 = 0.10
    elif grade1 == "GOLD":
        rate1 = 0.05
    else:
        rate1 = 0.0
    discount1 = int(amount1 * rate1)
    vat1 = int((amount1 - discount1) * 0.1)
    pay1 = amount1 - discount1 + vat1
    before_results.append(pay1)
    print(f"  {name1}: 決済額 {pay1:,}ウォン")

    # --- 注文 2 (コピー 2 — 税率にタイプミスが! 0.1 が 0.01 に)
    name2, grade2, amount2 = orders[1]
    if grade2 == "VIP":
        rate2 = 0.10
    elif grade2 == "GOLD":
        rate2 = 0.05
    else:
        rate2 = 0.0
    discount2 = int(amount2 * rate2)
    vat2 = int((amount2 - discount2) * 0.01)   # <- コピー中に生まれたタイプミス (バグ!)
    pay2 = amount2 - discount2 + vat2
    before_results.append(pay2)
    print(f"  {name2}: 決済額 {pay2:,}ウォン   <- どこかおかしいが気づきにくい")

    # --- 注文 3 (コピー 3)
    name3, grade3, amount3 = orders[2]
    if grade3 == "VIP":
        rate3 = 0.10
    elif grade3 == "GOLD":
        rate3 = 0.05
    else:
        rate3 = 0.0
    discount3 = int(amount3 * rate3)
    vat3 = int((amount3 - discount3) * 0.1)
    pay3 = amount3 - discount3 + vat3
    before_results.append(pay3)
    print(f"  {name3}: 決済額 {pay3:,}ウォン")
    print("  -> 同じロジックが3セット。税率が変わったら3か所 (実務なら数十か所) を直す羽目に。")

    # ---------------------------------------------------------
    # [2] リファクタリング後: 関数呼び出し3回
    # ---------------------------------------------------------
    print("\n[2] リファクタリング後 — 関数方式")

    after_results = []
    for name, grade, amount in orders:
        discount, vat, pay = calc_payment(amount, grade)    # 委任の1行
        after_results.append(pay)
        print(f"  {name}: 注文 {amount:,}ウォン - 割引 {discount:,}ウォン + 付加価値税 {vat:,}ウォン"
              f" = 決済額 {pay:,}ウォン")
    print("  -> ロジックは calc_payment の1か所だけ。税率変更も calc_vat の1行修正で終わり。")

    # ---------------------------------------------------------
    # [3] 前後の結果比較: コピー方式の不一致バグを発見
    # ---------------------------------------------------------
    print("\n[3] 前後の結果比較")
    for (name, _, _), b, a in zip(orders, before_results, after_results):
        mark = "一致" if b == a else f"不一致! 差額 {abs(b - a):,}ウォン (コピー版の税率タイプミス)"
        print(f"  {name}: 前 {b:,}ウォン / 後 {a:,}ウォン -> {mark}")

    # ---------------------------------------------------------
    # [4] 関数文法のまとめ
    # ---------------------------------------------------------
    print("\n[4] 関数文法のミニまとめ")

    # デフォルト値引数: ランクを渡さなければ「一般」として処理
    _, _, pay_default = calc_payment(100000)
    print(f"  calc_payment(100000)            -> {pay_default:,}ウォン (grade のデフォルト値 '一般')")

    # キーワード引数: 順序の代わりに名前で明確に
    _, _, pay_kw = calc_payment(amount=100000, grade="VIP")
    print(f"  calc_payment(amount=..., grade='VIP') -> {pay_kw:,}ウォン (キーワード引数)")

    # 複数の戻り値: タプルのアンパックで受け取る
    d, v, p = calc_payment(50000, "GOLD")
    print(f"  複数戻り値のアンパック -> 割引 {d:,} / 付加価値税 {v:,} / 決済 {p:,}")

    # スコープ: 関数の中のローカル変数は外から見えない
    def inner_demo():
        local_memo = "関数の机の上のメモ"       # ローカル変数
        return len(local_memo)

    inner_demo()
    try:
        print(local_memo)                         # 外からアクセスを試みる
    except NameError as e:
        print(f"  ローカル変数への外部アクセス -> NameError: {e}")
        print("  -> 関数の中の変数は、関数が終わると消えます (「各自の机」の原則)。")

    print("\n[終] 2回コピーしそうになった瞬間が、関数に切り出すタイミングです (DRY 原則)。")


if __name__ == "__main__":
    main()
