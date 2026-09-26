"""Type hints, testing, code quality — the contract-and-inspection demo.

We attach type hints (the contract) and docstrings (the manual) to
tax/discount functions, [3] experience the principle behind mypy with a
micro type checker, then [4] verify normal/edge/error cases with an
assert-based test runner. We also reproduce how the tests catch a
deliberately planted boundary bug.
"""

VAT_RATE: float = 0.1        # a named constant instead of a magic number (module-level hint)


# =============================================================
# 'Work-grade' functions with type hints + docstrings
# =============================================================
def calc_vat(amount: int, rate: float = VAT_RATE) -> int:
    """Compute VAT as a whole-KRW integer.

    Args:
        amount: pre-tax amount (KRW). 0 or more.
        rate: tax rate. Default 10%.
    Returns:
        VAT (KRW, floored to an integer).
    """
    return int(amount * rate)


def net_price(price: int, discount_rate: float) -> int:
    """Compute the discounted price.

    Args:
        price: list price (KRW). 0 or more.
        discount_rate: discount rate. 0.0 to 1.0.
    Returns:
        Discounted price (KRW, floored to an integer).
    Raises:
        ValueError: if the discount rate is outside 0.0-1.0.
    """
    if not 0.0 <= discount_rate <= 1.0:
        raise ValueError(f"Discount rate must be 0.0-1.0: {discount_rate}")
    return int(price * (1 - discount_rate))


def grade_of(purchase_total: int) -> str:
    """Grade a customer by annual purchases. 1M+ is GOLD, 5M+ is VIP.

    (Fixed version: uses >= so the boundary value 'exactly 1,000,000' counts as GOLD.)
    """
    if purchase_total >= 5000000:
        return "VIP"
    elif purchase_total >= 1000000:      # fix: >= instead of > (boundary included)
        return "GOLD"
    return "Regular"


def grade_of_buggy(purchase_total: int) -> str:
    """The buggy old version: a > comparison lets 'exactly 1,000,000' leak into Regular."""
    if purchase_total >= 5000000:
        return "VIP"
    elif purchase_total > 1000000:       # bug: exactly 1,000,000 falls through
        return "GOLD"
    return "Regular"


# =============================================================
# [3] A micro type checker: the mypy principle (signature vs actual values)
# =============================================================
def tiny_type_check(func, *args) -> list[str]:
    """Match a function's type hints against actual argument types; return the violations."""
    hints = {k: v for k, v in func.__annotations__.items() if k != "return"}
    problems: list[str] = []
    for (name, expected), value in zip(hints.items(), args):
        # Compare with type() directly so a bool in an int slot is caught too (concept demo)
        if expected in (int, float, str, bool) and type(value) is not expected:
            problems.append(
                f"{func.__name__}(): argument '{name}' is contracted as {expected.__name__} but "
                f"received {type(value).__name__} value {value!r}"
            )
    return problems


# =============================================================
# [4] A mini test runner: verify the normal/edge/error trio with assert
# =============================================================
def run_tests(target, label: str) -> tuple[int, int]:
    """Run the grading function `target` through the checklist; return (passed, failed)."""
    tests = [
        # (description, input, expected) — expected values derived by hand from the policy,
        # never copied from the implementation
        ("normal: 300k KRW is Regular",       300000,   "Regular"),
        ("normal: 2M KRW is GOLD",            2000000,  "GOLD"),
        ("normal: 7M KRW is VIP",             7000000,  "VIP"),
        ("edge: 0 KRW is Regular",            0,        "Regular"),
        ("edge: exactly 1M KRW is GOLD",      1000000,  "GOLD"),   # where the buggy version trips
        ("edge: exactly 5M KRW is VIP",       5000000,  "VIP"),
        ("edge: 999,999 KRW is Regular",      999999,   "Regular"),
    ]
    passed = failed = 0
    print(f"  --- {label} checklist ---")
    for desc, given, expected in tests:
        try:
            actual = target(given)
            assert actual == expected, f"expected {expected!r}, got {actual!r}"
            passed += 1
            print(f"    PASS {desc}")
        except AssertionError as e:
            failed += 1
            print(f"    FAIL {desc} -> {e}")

    # Error cases: does bad input raise the INTENDED exception? (demoed with net_price)
    for desc, bad_rate in [("error: discount rate 1.5 raises ValueError", 1.5),
                           ("error: discount rate -0.1 raises ValueError", -0.1)]:
        try:
            net_price(10000, bad_rate)
            failed += 1
            print(f"    FAIL {desc} -> no exception raised (a silent wrong answer is the worst)")
        except ValueError:
            passed += 1
            print(f"    PASS {desc}")

    print(f"  ==> {passed} passed, {failed} failed")
    return passed, failed


def main() -> None:
    print("=" * 56)
    print(" Type hints, testing, code quality")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] Reading the contract: type hints and __annotations__
    # ---------------------------------------------------------
    print("\n[1] Function signature = the contract")
    print("  def calc_vat(amount: int, rate: float = 0.1) -> int")
    print("  def net_price(price: int, discount_rate: float) -> int")
    print(f"  The contract Python remembers: calc_vat.__annotations__ = {calc_vat.__annotations__}")
    print(f"  Used as contracted: calc_vat(50000) = {calc_vat(50000)} / "
          f"net_price(20000, 0.3) = {net_price(20000, 0.3)}")

    # ---------------------------------------------------------
    # [2] Hints are not enforced at runtime
    # ---------------------------------------------------------
    print("\n[2] A contract violation... that still runs?!")
    sloppy = calc_vat("500", 2)           # str/int in int/float slots... and yet it runs
    print(f"  calc_vat('500', 2) = {sloppy!r}")
    print("  <- '500' * 2 becomes '500500', int('500500') succeeds, and a silent wrong answer emerges.")
    sneaky = calc_vat(True)               # bool counts as int, so it sails through
    print(f"  calc_vat(True)     = {sneaky!r}  <- a bool in an int slot passes unchallenged")
    print("  -> Python does not enforce hints at runtime. Hence the need for a 'checker'.")

    # ---------------------------------------------------------
    # [3] The micro type checker (the mypy principle)
    # ---------------------------------------------------------
    print("\n[3] Micro type checker — catching contract violations before running")
    for args in [(50000,), ("500500",), (50000, "10%")]:
        problems = tiny_type_check(calc_vat, *args)
        shown = ", ".join(repr(a) for a in args)
        if problems:
            for p in problems:
                print(f"  calc_vat({shown}) -> VIOLATION! {p}")
        else:
            print(f"  calc_vat({shown}) -> contract honored")
    print("  -> mypy is the professional checker that performs this matching across all code, 'without running it'.")

    # ---------------------------------------------------------
    # [4] The mini test runner: buggy vs fixed
    # ---------------------------------------------------------
    print("\n[4] Test runner — catching the boundary bug")
    print("  Grading policy: 1M KRW 'or more' is GOLD, 5M KRW 'or more' is VIP")

    buggy_pass, buggy_fail = run_tests(grade_of_buggy, "buggy grade_of_buggy (>)")
    print()
    fixed_pass, fixed_fail = run_tests(grade_of, "fixed grade_of (>=)")

    print(f"\n  Buggy version: {buggy_fail} failure(s) -> the 'exactly 1M KRW' customer gets demoted to Regular")
    print(f"  Fixed version: {fixed_fail} failure(s) -> all green. Now refactoring is safe")
    assert fixed_fail == 0, "the fixed version must pass everything"

    # ---------------------------------------------------------
    # [5] Docstrings: the user manual that help() shows
    # ---------------------------------------------------------
    print("\n[5] Docstring — a help(net_price) digest")
    doc = net_price.__doc__ or ""
    for line in doc.strip().splitlines()[:5]:
        print(f"  | {line.strip()}")
    print("  -> Type hints = the spec; docstring = the manual carrying units, ranges, and exceptions.")

    print("\n[End] From 'code that runs' to 'code you can trust'.")
    print("      Contract (hints) + checklist (tests) + manual (docstring) = the work-code trio.")


if __name__ == "__main__":
    main()
