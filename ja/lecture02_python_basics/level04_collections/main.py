"""リスト・タプル・辞書・集合 — 文具卸売店の在庫管理ミニ例題。

4種類のコレクション (collection) を在庫管理業務に当てはめてみます。
リスト = 入庫記録台帳 / タプル = 変更不可の規格表 /
辞書 = 商品名->数量の引き出し / 集合 = 重複のない品目名簿。
最後に実務の標準形である「辞書のリスト」で在庫表を扱います。
"""


def main():
    print("=" * 56)
    print(" コレクション — 文具卸売店の在庫管理")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] リスト: 順序のある入庫記録台帳
    # ---------------------------------------------------------
    print("\n[1] リスト — 今日の入庫数量の記録")

    inbound = [30, 12, 45]          # 午前の入庫3件
    inbound.append(20)              # 午後に1件追加
    print(f"  入庫記録        : {inbound}")
    print(f"  最初の件 inbound[0]  = {inbound[0]}  (インデックスは0から)")
    print(f"  最後の件 inbound[-1] = {inbound[-1]}")
    print(f"  スライス [1:3]  = {inbound[1:3]}  (3番の直前まで)")
    print(f"  件数 {len(inbound)} / 合計 {sum(inbound)} / ソート {sorted(inbound)}")

    # コピーの勘違いに注意: b = a は同じ台帳に名札をもう1枚付けただけ
    alias = inbound
    real_copy = inbound.copy()
    alias.append(99)
    print(f"  alias.append(99) の後の元リスト -> {inbound}  (元も変わる!)")
    print(f"  copy() したものは安全           -> {real_copy}")

    # ---------------------------------------------------------
    # [2] タプル: 変えてはいけない商品規格表
    # ---------------------------------------------------------
    print("\n[2] タプル — 商品規格表 (コード, 名前, 単価)")

    product = ("P001", "ボールペン", 1200)
    code, name, unit_price = product          # アンパック
    print(f"  規格表: {product}")
    print(f"  アンパック -> コード {code} / 名前 {name} / 単価 {unit_price:,}ウォン")

    try:
        product[2] = 1500                     # 単価をこっそり修正しようとする!
    except TypeError as e:
        print(f"  単価の修正を試みる -> TypeError: {e}")
        print("  -> 「うっかりでも変えられない」のがタプルの安全装置です。")

    # ---------------------------------------------------------
    # [3] 辞書: 商品名 -> 在庫数量の引き出し
    # ---------------------------------------------------------
    print("\n[3] 辞書 — 在庫の引き出し")

    stock = {"ボールペン": 37, "A4用紙": 12, "ホッチキス": 4}
    print(f"  現在の在庫              : {stock}")
    print(f"  stock['ボールペン']     = {stock['ボールペン']}個  (キーで即座に照会)")

    stock["蛍光ペン"] = 20                    # 新規入庫
    stock["A4用紙"] += 30                     # 追加入庫
    stock["ホッチキス"] -= 2                  # 出庫
    print(f"  入出庫の反映後          : {stock}")

    # 存在しないキーを安全に照会: get(キー, デフォルト値)
    print(f"  stock.get('消しゴム', 0) = {stock.get('消しゴム', 0)}  (KeyError なしでデフォルト値)")
    print(f"  '消しゴム' in stock      = {'消しゴム' in stock}")

    print("  引き出し全体の巡回:")
    for item, qty in stock.items():
        print(f"    - {item:6s}: {qty:>3}個")

    # ---------------------------------------------------------
    # [4] 集合: 発注/入庫の名簿突き合わせ
    # ---------------------------------------------------------
    print("\n[4] 集合 — 発注 vs 入庫の突き合わせ")

    ordered = {"ボールペン", "蛍光ペン", "消しゴム", "テープ", "ボールペン"}   # 重複は自動除去
    arrived = {"ボールペン", "蛍光ペン"}
    print(f"  発注品目 (重複を入れても1つだけ): {sorted(ordered)}")
    print(f"  入庫品目                        : {sorted(arrived)}")
    print(f"  未入庫 = 発注 - 入庫 (差集合)   : {sorted(ordered - arrived)}")
    print(f"  発注して入庫も済み (積集合)     : {sorted(ordered & arrived)}")
    print(f"  今週登場した全品目 (和集合)     : {sorted(ordered | arrived)}")

    # ---------------------------------------------------------
    # [5] 総合: 「辞書のリスト」= 実務の標準データ形
    # ---------------------------------------------------------
    print("\n[5] 在庫表 (辞書のリスト) と発注レポート")

    SAFETY = 10       # 安全在庫の基準
    inventory = [
        {"code": "P001", "name": "ボールペン", "qty": 37, "unit_price": 1200},
        {"code": "P002", "name": "A4用紙",     "qty": 42, "unit_price": 25000},
        {"code": "P003", "name": "ホッチキス", "qty": 2,  "unit_price": 8900},
        {"code": "P004", "name": "蛍光ペン",   "qty": 20, "unit_price": 1500},
        {"code": "P005", "name": "テープ",     "qty": 6,  "unit_price": 2300},
    ]

    total_value = 0
    shortage = []                             # 在庫不足の品目を集めるリスト
    for row in inventory:                     # 表の各行 (辞書) について
        value = row["qty"] * row["unit_price"]
        total_value += value
        flag = " <- 発注が必要!" if row["qty"] < SAFETY else ""
        print(f"  {row['code']} {row['name']:6s} {row['qty']:>3}個 x {row['unit_price']:>6,}ウォン"
              f" = {value:>9,}ウォン{flag}")
        if row["qty"] < SAFETY:
            shortage.append(row["name"])

    print(f"\n  在庫資産の総額: {total_value:,}ウォン")
    print(f"  発注が必要な品目 (安全在庫 {SAFETY}個未満): {shortage}")
    print("\n[終] 入れ物の選択基準 — 記録はリスト、不変のまとまりはタプル、")
    print("     キー照会は辞書、重複除去・名簿の突き合わせは集合。")


if __name__ == "__main__":
    main()
