"""例外処理とデバッグ — わざと壊した売上データで前後比較。

汚れたデータ (文字の金額、空の値、隠れた空白、列名の欠落) を
[2] 保護なしで処理して死ぬ様子と [3] try/except で生き残る様子を比較します。
代表的な例外5種の観察、print デバッグ (!r)、raise による早期警報まで実習します。
"""


def validate_amount(amount):
    """[5] 用の検証関数: マイナスの金額は契約違反として即座に raise。"""
    if amount < 0:
        raise ValueError(f"金額は負の値にできません: {amount}")
    return amount


def main():
    print("=" * 56)
    print(" 例外処理とデバッグ — 壊れたデータから生き残る")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 代表的な例外5種をわざと起こして「最後の行」を読んでみる
    # ---------------------------------------------------------
    print("\n[1] 代表的な例外の観察 (種類: 原因メッセージ)")

    demos = [
        ("int('協議中')", lambda: int("協議中")),
        ("'100' + 5", lambda: "100" + 5),
        ("{'a':1}['金額']", lambda: {"a": 1}["金額"]),
        ("[][0]", lambda: [][0]),
        ("10 / 0", lambda: 10 / 0),
    ]
    for code_text, run in demos:
        try:
            run()
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as e:
            # traceback の最後の行に当たる情報: 種類の名前 + 原因
            print(f"  {code_text:18s} -> {type(e).__name__}: {e}")
    print("  -> エラーメッセージは下から上へ。最後の行が「種類: 原因」です。")

    # 汚れた月間売上データ (実務ファイルでよくある事故のパターンを集めたもの)
    dirty_rows = [
        {"store": "渋谷店", "amount": "1200000"},
        {"store": "新宿店", "amount": "980000"},
        {"store": "池袋店", "amount": "協議中"},      # 文字の金額 -> ValueError
        {"store": "品川店", "amount": "1450000"},
        {"store": "上野店", "amount": ""},            # 空の値 -> ValueError
        {"store": "横浜店", "amount": " 730000 "},    # 隠れた空白 (int はできるが [4] で観察)
        {"store": "川崎店"},                           # amount キーそのものがない -> KeyError
        {"store": "大阪店", "amount": "2100000"},
        {"store": "名古屋店", "amount": "880000"},
        {"store": "福岡店", "amount": "1010000"},
    ]

    # ---------------------------------------------------------
    # [2] 保護なしの処理: 3番目の行で即死する
    # ---------------------------------------------------------
    print("\n[2] 例外処理なしで集計すると?")
    try:
        total = 0
        for i, row in enumerate(dirty_rows):
            total += int(row["amount"])           # 保護なし!
        print(f"  合計: {total}")                  # ここまでたどり着けない
    except (ValueError, KeyError) as e:
        print(f"  {i}番目の行 ({dirty_rows[i]['store']}) で死亡 -> {type(e).__name__}: {e}")
        print(f"  -> そこまでの部分合計 {total:,}ウォンも捨てられ、残りの行も処理できない。")

    # ---------------------------------------------------------
    # [3] 保護された処理: 不良は記録して飛ばし、最後まで進む
    # ---------------------------------------------------------
    print("\n[3] try/except で保護された集計")

    total = 0
    ok_count = 0
    failures = []                                  # (店舗, 理由) の記録
    for row in dirty_rows:
        try:
            amount = int(row["amount"].strip())    # 事故が起こりうる最小区間だけ try の中に
        except KeyError:
            failures.append((row["store"], "amount 列なし"))
            continue
        except ValueError as e:
            failures.append((row["store"], f"金額の形式エラー ({row['amount']!r})"))
            continue
        total += amount
        ok_count += 1

    print(f"  成功 {ok_count}件 / 失敗 {len(failures)}件 / 合計 {total:,}ウォン")
    print("  失敗の内訳 (黙って pass せず、必ず記録):")
    for store, reason in failures:
        print(f"    - {store}: {reason}")
    print("  -> 同じデータなのに [2] は死に、[3] は運用レポートまで出します。")

    # ---------------------------------------------------------
    # [4] print デバッグ: エラーなしで値だけがおかしいとき
    # ---------------------------------------------------------
    print("\n[4] print デバッグ — 隠れた空白を捕まえる")

    raw = " 730000 "
    print(f"  print(raw)      -> {raw}    (見た目は普通)")
    print(f"  print(f'{{raw!r}}') -> {raw!r}  <- !r で出すと空白が見える!")
    print(f"  int(raw.strip()) = {int(raw.strip()):,}  (strip してから変換すれば安全)")
    print("  コツ: 変数名と一緒に、!r で、疑わしい区間にだけ出し、解決したら消す。")

    # ---------------------------------------------------------
    # [5] raise: 不正な値は早く、大きく知らせる
    # ---------------------------------------------------------
    print("\n[5] raise — マイナスの金額の早期警報")
    try:
        validate_amount(150000)
        print("  validate_amount(150000)  -> 通過")
        validate_amount(-50000)
        print("  この行は実行されません")
    except ValueError as e:
        print(f"  validate_amount(-50000) -> ValueError: {e}")
        print("  -> 異常値を黙って通すより、早い段階で拒否するほうが安全です。")

    # ---------------------------------------------------------
    # finally のデモ: 事故の有無に関係ない後始末
    # ---------------------------------------------------------
    print("\n[6] finally — 何があってもやる後始末")
    try:
        risky = int("不良")
    except ValueError:
        print("  except: 不良値への対処完了")
    finally:
        print("  finally: (エラーが起きても) ファイルを閉じる・接続を片付けるといった戸締まりは実行されます")

    print("\n[終] 実務のデータは必ず壊れています。")
    print("     不良の1件は記録して飛ばすが、業務全体は止めないこと — それが例外処理です。")


if __name__ == "__main__":
    main()
