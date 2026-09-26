"""繰り返し (for/while/range/break/continue) — 顧客100人の購入データ集計。

同じ集計を「手計算方式」(1行ずつ全部書く) と「繰り返し方式」で比較して、
自動化のメリットを確認します。累積変数のパターン、繰り返し+条件分岐の組み合わせ、
while シミュレーション、break/continue、enumerate まで一気に身につけます。
"""

import random


def main():
    print("=" * 56)
    print(" 繰り返し — 顧客100人の購入データ集計")
    print("=" * 56)

    # 再現性のため seed を固定: 誰が実行しても同じデータになる
    random.seed(42)

    # 顧客100人の今月の購入額 (ウォン)。一部は0ウォン (休眠顧客)。
    purchases = []
    for i in range(100):
        if random.random() < 0.15:          # 15% は休眠顧客
            purchases.append(0)
        else:
            purchases.append(random.randint(10, 600) * 1000)  # 1万〜60万ウォン

    # ---------------------------------------------------------
    # [1] 手計算方式: 繰り返しがなかったら?
    # ---------------------------------------------------------
    print("\n[1] 繰り返しがなかったら (3人分だけ真似してみる)")
    total_by_hand = purchases[0] + purchases[1] + purchases[2]
    print(f"  total = purchases[0] + purchases[1] + purchases[2]  # = {total_by_hand:,}ウォン")
    print("  ... 100人ならこの足し算を100項書くか、100行並べる必要があります。")
    print("  顧客が101人になった瞬間、コードも直さなければなりません。")

    # ---------------------------------------------------------
    # [2] for の繰り返し: 4行で100人を集計
    # ---------------------------------------------------------
    print("\n[2] for の繰り返しで集計")

    total = 0                       # 累積変数は繰り返しの「前」で初期化
    best_amount = 0                 # 最高購入額
    for amount in purchases:        # 100人分を1件ずつ取り出して
        total += amount             # 毎周累積
        if amount > best_amount:    # 最高記録を更新
            best_amount = amount

    average = total / len(purchases)
    print(f"  顧客数       : {len(purchases)}人")
    print(f"  購入総額     : {total:,}ウォン")
    print(f"  1人あたり平均: {average:,.0f}ウォン")
    print(f"  最高購入額   : {best_amount:,}ウォン")
    print("  -> 顧客が1万人になっても、上のコードは一文字も変わりません。")

    # ---------------------------------------------------------
    # [3] 繰り返し + 条件分岐: 優良顧客の選別 (continue 活用)
    # ---------------------------------------------------------
    print("\n[3] 優良顧客の選別 (if/continue の組み合わせ)")

    VIP_THRESHOLD = 300000          # 優良顧客の基準: 30万ウォン以上
    vip_count = 0
    dormant_count = 0
    for amount in purchases:
        if amount == 0:             # 休眠顧客はスキップして次の人へ
            dormant_count += 1
            continue
        if amount >= VIP_THRESHOLD:
            vip_count += 1

    print(f"  優良顧客 (>= {VIP_THRESHOLD:,}ウォン) : {vip_count}人")
    print(f"  休眠顧客 (0ウォン、continue でスキップ) : {dormant_count}人")

    # ---------------------------------------------------------
    # [4] while: マーケティング予算の消化シミュレーション (break 活用)
    # ---------------------------------------------------------
    print("\n[4] while — 予算100万ウォンは何日持つか")

    budget = 1000000
    day = 0
    while budget > 0:               # 予算が残っている間は繰り返す
        day += 1
        spend = 60000 + random.randint(0, 50) * 1000   # 1日の支出 6万〜11万ウォン
        budget -= spend
        if day <= 3 or budget <= 0:                    # 最初の3日と最終日だけ出力
            print(f"  {day:>2}日目 支出 {spend:>7,}ウォン -> 残額 {max(budget, 0):>9,}ウォン")
        if day >= 60:               # 安全装置: 60日を超えたら強制終了
            print("  60日の上限に到達、break で終了")
            break

    print(f"  -> 予算は {day}日目に尽きました。")

    # ---------------------------------------------------------
    # [5] range と enumerate: 購入額の上位5人ランキング
    # ---------------------------------------------------------
    print("\n[5] 購入額の上位5人 (enumerate で順位を付ける)")

    top5 = sorted(purchases, reverse=True)[:5]     # 降順ソートして先頭5件
    for rank, amount in enumerate(top5, start=1):  # 1から順位番号
        print(f"  {rank}位: {amount:,}ウォン")

    print("\n  range の確認: range(5) ->", list(range(5)), "/ range(1, 6) ->", list(range(1, 6)))

    # ---------------------------------------------------------
    # 結論
    # ---------------------------------------------------------
    print("\n[終] コード行数の比較")
    print("  手計算方式   : 顧客数だけ行が増える (100人 = 100行+)")
    print("  繰り返し方式 : 常に4行 (データが増えてもコードはそのまま)")
    print("  「リストのすべての項目に同じ手順を」— これが自動化の核心となる一文です。")


if __name__ == "__main__":
    main()
