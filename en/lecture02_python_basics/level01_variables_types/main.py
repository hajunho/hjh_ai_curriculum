"""Variables and data types — cafe revenue calculation example.

We create the four basic types int/float/str/bool,
calculate revenue with variables, reproduce a type-conversion mistake,
and print a human-friendly report with f-string formatting.
"""


def main():
    print("=" * 52)
    print(" Variables and types — a cafe's daily revenue")
    print("=" * 52)

    # ---------------------------------------------------------
    # [1] Creating variables: put values into labeled boxes
    # ---------------------------------------------------------
    print("\n[1] Creating variables and checking types")

    menu_name = "Americano"      # str  : string
    unit_price = 4500            # int  : integer (price in KRW)
    cups_sold = 120              # int  : cups sold
    vat_rate = 0.1               # float: VAT rate 10%
    is_open = True               # bool : open for business today?

    print(f"  menu_name  = {menu_name!r:14} -> {type(menu_name).__name__}")
    print(f"  unit_price = {unit_price!r:14} -> {type(unit_price).__name__}")
    print(f"  cups_sold  = {cups_sold!r:14} -> {type(cups_sold).__name__}")
    print(f"  vat_rate   = {vat_rate!r:14} -> {type(vat_rate).__name__}")
    print(f"  is_open    = {is_open!r:14} -> {type(is_open).__name__}")

    # ---------------------------------------------------------
    # [2] Calculation: '=' means "put the right side into the left box"
    # ---------------------------------------------------------
    print("\n[2] Revenue calculation (sequential execution)")

    revenue = unit_price * cups_sold        # revenue = unit price x quantity
    vat = int(revenue * vat_rate)           # VAT (as a whole KRW integer)
    total_with_vat = revenue + vat          # total including VAT

    print(f"  Revenue     = {unit_price} x {cups_sold} = {revenue} KRW")
    print(f"  VAT (10%)   = {vat} KRW")
    print(f"  Total       = {total_with_vat} KRW")

    # The count = count + 1 pattern: add to the current value and put it back
    cups_sold = cups_sold + 5               # 5 extra cups sold just before closing
    print(f"  After the extra sale, cups_sold = {cups_sold} (added 5 to the old value and put it back)")

    # ---------------------------------------------------------
    # [3] Type conversion: the #1 workplace accident, the "text number"
    # ---------------------------------------------------------
    print("\n[3] Type conversion — the string-number trap")

    typed_price = "4500"        # values read from CSV/input arrive as strings like this
    typed_qty = "2"

    wrong = typed_price * 2                  # string * 2 = concatenation!
    print(f"  '4500' * 2          = {wrong!r}  <- not 9000 but string repetition")

    right = int(typed_price) * int(typed_qty)  # convert, then calculate
    print(f"  int('4500')*int('2') = {right}   <- convert and the math works")

    # ---------------------------------------------------------
    # [4] Building a report with f-string formatting
    # ---------------------------------------------------------
    print("\n[4] f-string revenue report")

    target = 600000                              # today's revenue target
    achieve_rate = total_with_vat / target       # achievement rate

    print(f"  Menu          : {menu_name}")
    print(f"  Total revenue : {total_with_vat:,} KRW (VAT included)")   # thousands separator
    print(f"  Target        : {target:,} KRW")
    print(f"  Achievement   : {achieve_rate:.1%}")                      # percentage, 1 decimal
    print(f"  Per-cup price : {unit_price:,.0f} KRW")                   # comma format, no decimals

    # ---------------------------------------------------------
    # [5] The tiny inaccuracy of floats
    # ---------------------------------------------------------
    print("\n[5] Checking the float error")

    result = 0.1 + 0.2
    print(f"  0.1 + 0.2            = {result}  <- not exactly 0.3 (a limit of binary storage)")
    print(f"  round(0.1 + 0.2, 2)  = {round(result, 2)}  <- fixed with rounding")
    print("  For money, calculating in whole integer units (KRW) is safest.")

    print("\n[End] A variable = a labeled box; a type = the kind of contents inside.")


if __name__ == "__main__":
    main()
