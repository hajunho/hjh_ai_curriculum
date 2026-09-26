"""Functions — tax/discount calculation, before and after refactoring.

We perform the same payment calculation (grade discount + VAT)
[1] copy-paste style (before refactoring, with a deliberately planted
    inconsistency bug) and
[2] function style (after refactoring), then compare the results.
Default arguments, keyword arguments, multiple returns, and scope are
demonstrated along the way.
"""

VAT_RATE = 0.1          # VAT rate 10%


# =============================================================
# 'After' refactoring: the logic lives in exactly one place
# =============================================================
def get_discount_rate(grade):
    """Return the discount rate for a customer grade."""
    if grade == "VIP":
        return 0.10
    elif grade == "GOLD":
        return 0.05
    else:
        return 0.0


def calc_vat(amount):
    """Compute VAT (10%) as a whole-KRW integer. A rate change edits this one line."""
    return int(amount * VAT_RATE)


def calc_payment(amount, grade="Regular"):
    """Compute the final payment with discount and VAT applied.

    Returns: a (discount, VAT, final payment) tuple
    """
    discount = int(amount * get_discount_rate(grade))   # a function delegating to a function
    base = amount - discount
    vat = calc_vat(base)
    return discount, vat, base + vat


def main():
    print("=" * 56)
    print(" Functions — tax/discount before and after refactoring")
    print("=" * 56)

    # Order data: (customer name, grade, order amount)
    orders = [
        ("Kim", "VIP", 200000),
        ("Lee", "GOLD", 150000),
        ("Park", "Regular", 80000),
    ]

    # ---------------------------------------------------------
    # [1] Before refactoring: the same logic copy-pasted per order
    # ---------------------------------------------------------
    print("\n[1] Before refactoring — the copy-paste way")

    before_results = []

    # --- Order 1 (copy 1)
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
    print(f"  {name1}: payment {pay1:,} KRW")

    # --- Order 2 (copy 2 — a typo crept into the tax rate! 0.1 became 0.01)
    name2, grade2, amount2 = orders[1]
    if grade2 == "VIP":
        rate2 = 0.10
    elif grade2 == "GOLD":
        rate2 = 0.05
    else:
        rate2 = 0.0
    discount2 = int(amount2 * rate2)
    vat2 = int((amount2 - discount2) * 0.01)   # <- a typo born of copying (bug!)
    pay2 = amount2 - discount2 + vat2
    before_results.append(pay2)
    print(f"  {name2}: payment {pay2:,} KRW   <- something is off, but hard to notice")

    # --- Order 3 (copy 3)
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
    print(f"  {name3}: payment {pay3:,} KRW")
    print("  -> Three copies of the same logic. A rate change means fixing 3 spots (dozens, in real code).")

    # ---------------------------------------------------------
    # [2] After refactoring: three function calls
    # ---------------------------------------------------------
    print("\n[2] After refactoring — the function way")

    after_results = []
    for name, grade, amount in orders:
        discount, vat, pay = calc_payment(amount, grade)    # one line of delegation
        after_results.append(pay)
        print(f"  {name}: order {amount:,} KRW - discount {discount:,} KRW + VAT {vat:,} KRW"
              f" = payment {pay:,} KRW")
    print("  -> The logic lives only in calc_payment. A rate change is one line in calc_vat.")

    # ---------------------------------------------------------
    # [3] Comparing before/after: spotting the copy-paste bug
    # ---------------------------------------------------------
    print("\n[3] Before/after comparison")
    for (name, _, _), b, a in zip(orders, before_results, after_results):
        mark = "match" if b == a else f"MISMATCH! difference {abs(b - a):,} KRW (the copy's tax-rate typo)"
        print(f"  {name}: before {b:,} KRW / after {a:,} KRW -> {mark}")

    # ---------------------------------------------------------
    # [4] Function syntax recap
    # ---------------------------------------------------------
    print("\n[4] Mini function-syntax recap")

    # Default argument: no grade given -> treated as 'Regular'
    _, _, pay_default = calc_payment(100000)
    print(f"  calc_payment(100000)            -> {pay_default:,} KRW (grade defaults to 'Regular')")

    # Keyword arguments: clarity by name instead of order
    _, _, pay_kw = calc_payment(amount=100000, grade="VIP")
    print(f"  calc_payment(amount=..., grade='VIP') -> {pay_kw:,} KRW (keyword arguments)")

    # Multiple returns: received by tuple unpacking
    d, v, p = calc_payment(50000, "GOLD")
    print(f"  Multiple-return unpacking -> discount {d:,} / VAT {v:,} / payment {p:,}")

    # Scope: a function's local variables are invisible outside
    def inner_demo():
        local_memo = "a note on the function's desk"       # local variable
        return len(local_memo)

    inner_demo()
    try:
        print(local_memo)                         # attempt to access from outside
    except NameError as e:
        print(f"  Accessing a local from outside -> NameError: {e}")
        print("  -> Variables inside a function vanish when it ends (the 'own desk' principle).")

    print("\n[End] The moment you copy code a second time is the moment to extract a function (DRY).")


if __name__ == "__main__":
    main()
