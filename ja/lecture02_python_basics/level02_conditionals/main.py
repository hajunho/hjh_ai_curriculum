"""条件分岐 (if/elif/else) — 出張経費の自動承認判定機。

社内の経費規定を if/elif/else のはしごに置き換え、自動判定をしてみます。
比較演算子 (==, <=, ...) と論理演算子 (and/or/not) の True/False を確認し、
条件の「順序」が規定の優先順位であることを実験で証明します。
"""


def judge_expense(amount, has_receipt):
    """経費1件を社内規定に従って判定する。

    規定 (上から順に、最初に該当した条項を一つだけ適用):
      1) 5万ウォン以下              -> 自動承認
      2) 15万ウォン以下 + 領収書    -> 課長専決
      3) 50万ウォン以下             -> 部長決裁
      4) それ以外                   -> 差し戻し(説明要求)
    """
    if amount <= 50000:                       # 条項 1
        return "自動承認"
    elif amount <= 150000 and has_receipt:    # 条項 2 (and: 両方とも真であること)
        return "課長専決"
    elif amount <= 500000:                    # 条項 3
        return "部長決裁"
    else:                                     # 条項 4: 上のすべてに該当なし
        return "差し戻し(説明要求)"


def judge_wrong_order(amount, has_receipt):
    """わざと「広い条件を先に」検査する間違った規定。順序の重要性の比較用。"""
    if amount <= 500000:                      # 広い条件が一番上に来ると...
        return "部長決裁"
    elif amount <= 150000 and has_receipt:    # この条項は永遠に実行されない!
        return "課長専決"
    elif amount <= 50000:                     # この条項も同じ
        return "自動承認"
    else:
        return "差し戻し(説明要求)"


def main():
    print("=" * 56)
    print(" 条件分岐 — 出張経費の自動承認判定機")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 単一判定: 経費1件を規定に通してみる
    # ---------------------------------------------------------
    print("\n[1] 単一判定")
    amount = 120000          # 申請金額 (ウォン)
    has_receipt = True       # 領収書の提出有無

    print(f"  申請金額 {amount:,}ウォン / 領収書 {'あり' if has_receipt else 'なし'}")
    print(f"  -> 判定: {judge_expense(amount, has_receipt)}")
    print("  (5万超なので条項1は不成立 -> 15万以下+領収書ありなので条項2で確定、以降の条項は見ない)")

    # ---------------------------------------------------------
    # [2] 比較・論理演算子の実験室
    # ---------------------------------------------------------
    print("\n[2] 比較・論理演算子の True/False")
    print(f"  amount <= 150000        -> {amount <= 150000}")
    print(f"  amount == 120000        -> {amount == 120000}")
    print(f"  amount != 120000        -> {amount != 120000}")
    print(f"  amount <= 150000 and has_receipt -> {amount <= 150000 and has_receipt}")
    print(f"  amount <= 50000 or has_receipt   -> {amount <= 50000 or has_receipt}")
    print(f"  not has_receipt         -> {not has_receipt}")
    print(f"  50000 < amount <= 150000 (範囲比較) -> {50000 < amount <= 150000}")
    grade = "VIP"
    print(f"  grade in ('VIP','VVIP') -> {grade in ('VIP', 'VVIP')}")

    # ---------------------------------------------------------
    # [3] 条件の「順序」の重要性: 同じ条項、違う順序
    # ---------------------------------------------------------
    print("\n[3] 順序を入れ替えると規定が壊れる")
    test_amount = 30000      # 本来は「自動承認」になるはずの金額
    ok = judge_expense(test_amount, True)
    bad = judge_wrong_order(test_amount, True)
    print(f"  3万ウォンの経費、正しい順序 (狭い条件が先) -> {ok}")
    print(f"  3万ウォンの経費、間違った順序 (広い条件が先) -> {bad}")
    print("  -> 広い条件が上にあると、下の厳しい条項は永遠に実行されません。")

    # ---------------------------------------------------------
    # [4] 一括判定: 申請6件を同じ規定で処理 (繰り返しの先取り)
    # ---------------------------------------------------------
    print("\n[4] 今週の経費申請6件を一括判定")

    # (申請者, 金額, 領収書の有無)
    requests = [
        ("田中主任", 32000, True),
        ("佐藤課長", 120000, True),
        ("鈴木係長", 120000, False),   # 同じ金額でも領収書なし -> 結果が変わる
        ("高橋部長", 480000, True),
        ("伊藤社員", 750000, True),
        ("渡辺係長", 50000, False),
    ]

    approved = 0     # 自動承認の件数
    escalated = 0    # 決裁が必要な件数
    rejected = 0     # 差し戻しの件数

    for name, amt, receipt in requests:
        decision = judge_expense(amt, receipt)
        print(f"  {name} | {amt:>8,}ウォン | 領収書 {'O' if receipt else 'X'} -> {decision}")
        if decision == "自動承認":
            approved += 1
        elif decision == "差し戻し(説明要求)":
            rejected += 1
        else:
            escalated += 1

    print(f"\n  集計: 自動承認 {approved}件 / 決裁必要 {escalated}件 / 差し戻し {rejected}件")
    print("\n[終] 規定文書の条項 = if/elif/else の枝。条項の順序 = 条件の順序。")


if __name__ == "__main__":
    main()
