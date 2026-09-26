"""Lists, tuples, dictionaries, sets — a stationery wholesaler's inventory mini example.

We map the four collections onto inventory-management work.
List = goods-received ledger / tuple = unchangeable spec sheet /
dictionary = product-name->quantity drawer cabinet / set = duplicate-free item roster.
At the end we handle the inventory table in the real-world standard shape,
a 'list of dictionaries'.
"""


def main():
    print("=" * 56)
    print(" Collections — stationery wholesaler inventory")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] List: the ordered goods-received ledger
    # ---------------------------------------------------------
    print("\n[1] List — today's received quantities")

    inbound = [30, 12, 45]          # 3 deliveries in the morning
    inbound.append(20)              # 1 more in the afternoon
    print(f"  Received record  : {inbound}")
    print(f"  First inbound[0]   = {inbound[0]}  (indexes start at 0)")
    print(f"  Last inbound[-1]   = {inbound[-1]}")
    print(f"  Slice [1:3]      = {inbound[1:3]}  (up to, not including, index 3)")
    print(f"  Count {len(inbound)} / total {sum(inbound)} / sorted {sorted(inbound)}")

    # Beware the copy illusion: b = a just adds a second label to the same ledger
    alias = inbound
    real_copy = inbound.copy()
    alias.append(99)
    print(f"  After alias.append(99), the original -> {inbound}  (the original changed!)")
    print(f"  The copy() version is safe            -> {real_copy}")

    # ---------------------------------------------------------
    # [2] Tuple: the product spec sheet that must not change
    # ---------------------------------------------------------
    print("\n[2] Tuple — product spec sheet (code, name, unit price)")

    product = ("P001", "Ballpoint pen", 1200)
    code, name, unit_price = product          # unpacking
    print(f"  Spec sheet: {product}")
    print(f"  Unpacked -> code {code} / name {name} / unit price {unit_price:,} KRW")

    try:
        product[2] = 1500                     # sneaky attempt to change the price!
    except TypeError as e:
        print(f"  Price-edit attempt -> TypeError: {e}")
        print("  -> 'Can't change it even by accident' is the tuple's safety guard.")

    # ---------------------------------------------------------
    # [3] Dictionary: product name -> stock quantity drawer cabinet
    # ---------------------------------------------------------
    print("\n[3] Dictionary — the stock drawer cabinet")

    stock = {"Ballpoint pen": 37, "A4 paper": 12, "Stapler": 4}
    print(f"  Current stock            : {stock}")
    print(f"  stock['Ballpoint pen']   = {stock['Ballpoint pen']} units  (instant lookup by key)")

    stock["Highlighter"] = 20                 # new arrival
    stock["A4 paper"] += 30                   # additional arrival
    stock["Stapler"] -= 2                     # shipped out
    print(f"  After the moves          : {stock}")

    # Safely look up a missing key: get(key, default)
    print(f"  stock.get('Eraser', 0) = {stock.get('Eraser', 0)}  (default value, no KeyError)")
    print(f"  'Eraser' in stock      = {'Eraser' in stock}")

    print("  Walking the whole cabinet:")
    for item, qty in stock.items():
        print(f"    - {item:13s}: {qty:>3} units")

    # ---------------------------------------------------------
    # [4] Set: matching orders against arrivals
    # ---------------------------------------------------------
    print("\n[4] Set — ordered vs received")

    ordered = {"Ballpoint pen", "Highlighter", "Eraser", "Tape", "Ballpoint pen"}   # duplicates auto-removed
    arrived = {"Ballpoint pen", "Highlighter"}
    print(f"  Ordered items (dupes collapse to one): {sorted(ordered)}")
    print(f"  Received items                       : {sorted(arrived)}")
    print(f"  Missing = ordered - received (difference): {sorted(ordered - arrived)}")
    print(f"  Ordered AND received (intersection)      : {sorted(ordered & arrived)}")
    print(f"  Every item mentioned this week (union)   : {sorted(ordered | arrived)}")

    # ---------------------------------------------------------
    # [5] Together: 'list of dictionaries' = the real-world standard shape
    # ---------------------------------------------------------
    print("\n[5] Inventory table (list of dictionaries) and reorder report")

    SAFETY = 10       # safety-stock threshold
    inventory = [
        {"code": "P001", "name": "Ballpoint pen", "qty": 37, "unit_price": 1200},
        {"code": "P002", "name": "A4 paper",      "qty": 42, "unit_price": 25000},
        {"code": "P003", "name": "Stapler",       "qty": 2,  "unit_price": 8900},
        {"code": "P004", "name": "Highlighter",   "qty": 20, "unit_price": 1500},
        {"code": "P005", "name": "Tape",          "qty": 6,  "unit_price": 2300},
    ]

    total_value = 0
    shortage = []                             # list to collect low-stock items
    for row in inventory:                     # for each row (dictionary) of the table
        value = row["qty"] * row["unit_price"]
        total_value += value
        flag = " <- reorder needed!" if row["qty"] < SAFETY else ""
        print(f"  {row['code']} {row['name']:13s} {row['qty']:>3} x {row['unit_price']:>6,} KRW"
              f" = {value:>9,} KRW{flag}")
        if row["qty"] < SAFETY:
            shortage.append(row["name"])

    print(f"\n  Total inventory value: {total_value:,} KRW")
    print(f"  Items to reorder (below safety stock of {SAFETY}): {shortage}")
    print("\n[End] Container choice — records go in lists, immutable bundles in tuples,")
    print("      key lookups in dictionaries, de-duplication and list matching in sets.")


if __name__ == "__main__":
    main()
