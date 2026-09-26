"""型ヒント・テスト・コード品質 — 契約書と品質検査のデモ。

税金・割引の計算関数に型ヒント (契約書) と docstring (説明書) を付け、
[3] 超ミニ型検査器で mypy の原理を体験した後、
[4] assert ベースの自作テストランナーで正常・境界・エラーケースを検証します。
わざと仕込んだ境界値バグをテストが捕まえる過程も再現します。
"""

VAT_RATE: float = 0.1        # マジックナンバーの代わりに名前付き定数 (モジュールレベルのヒント)


# =============================================================
# 型ヒント + docstring の付いた「業務用」関数たち
# =============================================================
def calc_vat(amount: int, rate: float = VAT_RATE) -> int:
    """付加価値税をウォン単位の整数で計算する。

    Args:
        amount: 税抜き金額 (ウォン)。0以上。
        rate: 税率。デフォルト 10%。
    Returns:
        付加価値税 (ウォン、整数切り捨て)。
    """
    return int(amount * rate)


def net_price(price: int, discount_rate: float) -> int:
    """割引適用価格を計算する。

    Args:
        price: 定価 (ウォン)。0以上。
        discount_rate: 割引率。0.0 〜 1.0。
    Returns:
        割引適用価格 (ウォン、整数切り捨て)。
    Raises:
        ValueError: 割引率が 0.0〜1.0 の範囲を外れた場合。
    """
    if not 0.0 <= discount_rate <= 1.0:
        raise ValueError(f"割引率は 0.0〜1.0 でなければなりません: {discount_rate}")
    return int(price * (1 - discount_rate))


def grade_of(purchase_total: int) -> str:
    """年間購入額で顧客ランクを判定する。100万以上 GOLD、500万以上 VIP。

    (バグ修正版: 境界値「ちょうど100万ウォン」も GOLD に含まれるよう >= を使用)
    """
    if purchase_total >= 5000000:
        return "VIP"
    elif purchase_total >= 1000000:      # 修正: > ではなく >= (境界を含む)
        return "GOLD"
    return "一般"


def grade_of_buggy(purchase_total: int) -> str:
    """バグのある旧バージョン: 境界値「ちょうど100万ウォン」が一般に漏れる > の比較。"""
    if purchase_total >= 5000000:
        return "VIP"
    elif purchase_total > 1000000:       # バグ: ちょうど 1,000,000 が脱落する
        return "GOLD"
    return "一般"


# =============================================================
# [3] 超ミニ型検査器: mypy の原理体験 (シグネチャ vs 実際の値の突き合わせ)
# =============================================================
def tiny_type_check(func, *args) -> list[str]:
    """関数の型ヒントと実際の引数の型を突き合わせ、違反リストを返す。"""
    hints = {k: v for k, v in func.__annotations__.items() if k != "return"}
    problems: list[str] = []
    for (name, expected), value in zip(hints.items(), args):
        # int の場所に bool が来るのも捕まえるため type() を直接比較 (概念デモ用)
        if expected in (int, float, str, bool) and type(value) is not expected:
            problems.append(
                f"{func.__name__}(): 引数 '{name}' は {expected.__name__} の契約なのに "
                f"{type(value).__name__} の値 {value!r} が入ってきた"
            )
    return problems


# =============================================================
# [4] ミニテストランナー: assert で正常・境界・エラーの3点セットを検証
# =============================================================
def run_tests(target, label: str) -> tuple[int, int]:
    """ランク判定関数 target をチェックリストで検査し (合格, 不合格) を返す。"""
    tests = [
        # (説明, 入力, 期待値) — 期待値は実装を写さず、規定から手で求める
        ("正常: 30万ウォンは一般",          300000,   "一般"),
        ("正常: 200万ウォンは GOLD",       2000000,  "GOLD"),
        ("正常: 700万ウォンは VIP",        7000000,  "VIP"),
        ("境界: 0ウォンは一般",            0,        "一般"),
        ("境界: ちょうど100万ウォンは GOLD", 1000000, "GOLD"),   # バグ版が引っかかる地点
        ("境界: ちょうど500万ウォンは VIP",  5000000, "VIP"),
        ("境界: 99万9999ウォンは一般",     999999,   "一般"),
    ]
    passed = failed = 0
    print(f"  --- {label} チェックリスト ---")
    for desc, given, expected in tests:
        try:
            actual = target(given)
            assert actual == expected, f"期待 {expected!r}, 実際 {actual!r}"
            passed += 1
            print(f"    PASS {desc}")
        except AssertionError as e:
            failed += 1
            print(f"    FAIL {desc} -> {e}")

    # エラーケース: 不正な入力が「意図した例外」を出すか (net_price で実演)
    for desc, bad_rate in [("エラー: 割引率 1.5 は ValueError", 1.5),
                           ("エラー: 割引率 -0.1 は ValueError", -0.1)]:
        try:
            net_price(10000, bad_rate)
            failed += 1
            print(f"    FAIL {desc} -> 例外が出ない (静かな誤答が最悪)")
        except ValueError:
            passed += 1
            print(f"    PASS {desc}")

    print(f"  ==> {passed} passed, {failed} failed")
    return passed, failed


def main() -> None:
    print("=" * 56)
    print(" 型ヒント・テスト・コード品質")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 契約書を読む: 型ヒントと __annotations__
    # ---------------------------------------------------------
    print("\n[1] 関数のシグネチャ = 契約書")
    print("  def calc_vat(amount: int, rate: float = 0.1) -> int")
    print("  def net_price(price: int, discount_rate: float) -> int")
    print(f"  Python が覚えている契約: calc_vat.__annotations__ = {calc_vat.__annotations__}")
    print(f"  契約どおりに使う: calc_vat(50000) = {calc_vat(50000)} / "
          f"net_price(20000, 0.3) = {net_price(20000, 0.3)}")

    # ---------------------------------------------------------
    # [2] ヒントは実行時に強制されない
    # ---------------------------------------------------------
    print("\n[2] 契約違反なのに実行できてしまう?!")
    sloppy = calc_vat("500", 2)           # int/float の契約の場所に str/int... それでも動く
    print(f"  calc_vat('500', 2) = {sloppy!r}")
    print("  <- '500' * 2 = '500500' になり int('500500') が成立して、静かに誤答を出します。")
    sneaky = calc_vat(True)               # bool も int 扱いなのでそのまま通過
    print(f"  calc_vat(True)     = {sneaky!r}  <- bool が int の場所に入っても素通り")
    print("  -> Python は実行時にヒントを強制しません。だから「検査器」が必要なのです。")

    # ---------------------------------------------------------
    # [3] 超ミニ型検査器 (mypy の原理)
    # ---------------------------------------------------------
    print("\n[3] 超ミニ型検査器 — 実行前に契約違反を捕まえる")
    for args in [(50000,), ("500500",), (50000, "10%")]:
        problems = tiny_type_check(calc_vat, *args)
        shown = ", ".join(repr(a) for a in args)
        if problems:
            for p in problems:
                print(f"  calc_vat({shown}) -> 違反! {p}")
        else:
            print(f"  calc_vat({shown}) -> 契約遵守")
    print("  -> mypy はこの突き合わせを「実行せずに」コード全体に行う専門の検査器です。")

    # ---------------------------------------------------------
    # [4] ミニテストランナー: バグ版 vs 修正版
    # ---------------------------------------------------------
    print("\n[4] テストランナー — 境界値バグを捕まえる過程")
    print("  ランク規定: 100万ウォン「以上」GOLD、500万ウォン「以上」VIP")

    buggy_pass, buggy_fail = run_tests(grade_of_buggy, "バグ版 grade_of_buggy (>)")
    print()
    fixed_pass, fixed_fail = run_tests(grade_of, "修正版 grade_of (>=)")

    print(f"\n  バグ版: {buggy_fail}件不合格 -> 「ちょうど100万ウォン」の顧客が一般に降格されるバグ")
    print(f"  修正版: {fixed_fail}件不合格 -> すべて合格。これで安心してリファクタリングできます")
    assert fixed_fail == 0, "修正版は必ずすべて合格しなければなりません"

    # ---------------------------------------------------------
    # [5] docstring: help() が見せる取扱説明書
    # ---------------------------------------------------------
    print("\n[5] docstring — help(net_price) の要約")
    doc = net_price.__doc__ or ""
    for line in doc.strip().splitlines()[:5]:
        print(f"  | {line.strip()}")
    print("  -> 型ヒント = 規格、docstring = 単位・範囲・例外まで収める説明書。")

    print("\n[終] 「動くコード」から「信頼できるコード」へ。")
    print("     契約 (ヒント) + チェックリスト (テスト) + 説明書 (docstring) = 業務用コードの3点セット。")


if __name__ == "__main__":
    main()
