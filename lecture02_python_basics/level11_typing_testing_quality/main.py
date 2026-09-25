"""타입 힌트·테스트·코드 품질 — 계약서와 품질검사 데모.

세금·할인 계산 함수에 타입 힌트(계약서)와 docstring(설명서)을 붙이고,
[3] 초미니 타입 검사기로 mypy 의 원리를 체험한 뒤
[4] assert 기반 자체 테스트 러너로 정상·경계·오류 케이스를 검증합니다.
일부러 심은 경계값 버그를 테스트가 잡아내는 과정도 재현합니다.
"""

VAT_RATE: float = 0.1        # 마법 숫자 대신 이름 붙은 상수 (모듈 수준 힌트)


# =============================================================
# 타입 힌트 + docstring 이 붙은 '업무용' 함수들
# =============================================================
def calc_vat(amount: int, rate: float = VAT_RATE) -> int:
    """부가세를 원 단위 정수로 계산한다.

    Args:
        amount: 공급가액(원). 0 이상.
        rate: 세율. 기본 10%.
    Returns:
        부가세(원, 정수 내림).
    """
    return int(amount * rate)


def net_price(price: int, discount_rate: float) -> int:
    """할인 적용가를 계산한다.

    Args:
        price: 정가(원). 0 이상.
        discount_rate: 할인율. 0.0 ~ 1.0.
    Returns:
        할인 적용가(원, 정수 내림).
    Raises:
        ValueError: 할인율이 0.0~1.0 범위를 벗어나면.
    """
    if not 0.0 <= discount_rate <= 1.0:
        raise ValueError(f"할인율은 0.0~1.0 이어야 합니다: {discount_rate}")
    return int(price * (1 - discount_rate))


def grade_of(purchase_total: int) -> str:
    """연간 구매액으로 고객 등급을 판정한다. 100만 이상 GOLD, 500만 이상 VIP.

    (버그 수정판: 경계값 '딱 100만 원'도 GOLD 에 포함되도록 >= 사용)
    """
    if purchase_total >= 5000000:
        return "VIP"
    elif purchase_total >= 1000000:      # 수정: > 가 아니라 >= (경계 포함)
        return "GOLD"
    return "일반"


def grade_of_buggy(purchase_total: int) -> str:
    """버그 있는 예전 버전: 경계값 '딱 100만 원'이 일반으로 새는 > 비교."""
    if purchase_total >= 5000000:
        return "VIP"
    elif purchase_total > 1000000:       # 버그: 정확히 1,000,000 이 탈락한다
        return "GOLD"
    return "일반"


# =============================================================
# [3] 초미니 타입 검사기: mypy 원리 체험 (서명 vs 실제 값 대조)
# =============================================================
def tiny_type_check(func, *args) -> list[str]:
    """함수의 타입 힌트와 실제 인자 타입을 대조해 위반 목록을 돌려준다."""
    hints = {k: v for k, v in func.__annotations__.items() if k != "return"}
    problems: list[str] = []
    for (name, expected), value in zip(hints.items(), args):
        # int 자리에 bool 이 오는 것도 잡도록 type() 을 직접 비교 (개념 데모용)
        if expected in (int, float, str, bool) and type(value) is not expected:
            problems.append(
                f"{func.__name__}(): 인자 '{name}' 는 {expected.__name__} 계약인데 "
                f"{type(value).__name__} 값 {value!r} 이 들어옴"
            )
    return problems


# =============================================================
# [4] 미니 테스트 러너: assert 로 정상·경계·오류 3종 세트 검증
# =============================================================
def run_tests(target, label: str) -> tuple[int, int]:
    """등급 판정 함수 target 을 체크리스트로 검사해 (통과, 실패)를 돌려준다."""
    tests = [
        # (설명, 입력, 기대값) — 기대값은 구현을 베끼지 않고 규정에서 손으로 구함
        ("정상: 30만 원은 일반",        300000,   "일반"),
        ("정상: 200만 원은 GOLD",      2000000,  "GOLD"),
        ("정상: 700만 원은 VIP",       7000000,  "VIP"),
        ("경계: 0원은 일반",           0,        "일반"),
        ("경계: 딱 100만 원은 GOLD",   1000000,  "GOLD"),   # 버그판이 걸리는 지점
        ("경계: 딱 500만 원은 VIP",    5000000,  "VIP"),
        ("경계: 99만9999원은 일반",    999999,   "일반"),
    ]
    passed = failed = 0
    print(f"  --- {label} 체크리스트 ---")
    for desc, given, expected in tests:
        try:
            actual = target(given)
            assert actual == expected, f"기대 {expected!r}, 실제 {actual!r}"
            passed += 1
            print(f"    PASS {desc}")
        except AssertionError as e:
            failed += 1
            print(f"    FAIL {desc} -> {e}")

    # 오류 케이스: 잘못된 입력이 '의도한 예외'를 내는지 (net_price 로 시연)
    for desc, bad_rate in [("오류: 할인율 1.5 는 ValueError", 1.5),
                           ("오류: 할인율 -0.1 은 ValueError", -0.1)]:
        try:
            net_price(10000, bad_rate)
            failed += 1
            print(f"    FAIL {desc} -> 예외가 나지 않음(조용한 오답이 최악)")
        except ValueError:
            passed += 1
            print(f"    PASS {desc}")

    print(f"  ==> {passed} passed, {failed} failed")
    return passed, failed


def main() -> None:
    print("=" * 56)
    print(" 타입 힌트·테스트·코드 품질")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 계약서 읽기: 타입 힌트와 __annotations__
    # ---------------------------------------------------------
    print("\n[1] 함수 서명 = 계약서")
    print("  def calc_vat(amount: int, rate: float = 0.1) -> int")
    print("  def net_price(price: int, discount_rate: float) -> int")
    print(f"  파이썬이 기억하는 계약: calc_vat.__annotations__ = {calc_vat.__annotations__}")
    print(f"  계약대로 사용: calc_vat(50000) = {calc_vat(50000)} / "
          f"net_price(20000, 0.3) = {net_price(20000, 0.3)}")

    # ---------------------------------------------------------
    # [2] 힌트는 실행 시 강제되지 않는다
    # ---------------------------------------------------------
    print("\n[2] 계약 위반인데 실행은 된다?!")
    sloppy = calc_vat("500", 2)           # int/float 계약 자리에 str/int... 그런데 돌아간다
    print(f"  calc_vat('500', 2) = {sloppy!r}")
    print("  <- '500' * 2 = '500500' 이 되고 int('500500') 이 성립해 조용히 오답을 냅니다.")
    sneaky = calc_vat(True)               # bool 도 int 취급이라 그냥 통과
    print(f"  calc_vat(True)     = {sneaky!r}  <- bool 이 int 자리에 들어가도 무사통과")
    print("  -> 파이썬은 실행 시 힌트를 강제하지 않습니다. 그래서 '검사기'가 필요합니다.")

    # ---------------------------------------------------------
    # [3] 초미니 타입 검사기 (mypy 원리)
    # ---------------------------------------------------------
    print("\n[3] 초미니 타입 검사기 — 실행 전에 계약 위반 잡기")
    for args in [(50000,), ("500500",), (50000, "10%")]:
        problems = tiny_type_check(calc_vat, *args)
        shown = ", ".join(repr(a) for a in args)
        if problems:
            for p in problems:
                print(f"  calc_vat({shown}) -> 위반! {p}")
        else:
            print(f"  calc_vat({shown}) -> 계약 준수")
    print("  -> mypy 는 이 대조를 '실행 없이' 코드 전체에 수행하는 전문 검사기입니다.")

    # ---------------------------------------------------------
    # [4] 미니 테스트 러너: 버그판 vs 수정판
    # ---------------------------------------------------------
    print("\n[4] 테스트 러너 — 경계값 버그를 잡는 과정")
    print("  등급 규정: 100만 원 '이상' GOLD, 500만 원 '이상' VIP")

    buggy_pass, buggy_fail = run_tests(grade_of_buggy, "버그판 grade_of_buggy (>)")
    print()
    fixed_pass, fixed_fail = run_tests(grade_of, "수정판 grade_of (>=)")

    print(f"\n  버그판: {buggy_fail}건 실패 -> '딱 100만 원' 고객이 일반으로 강등되는 버그")
    print(f"  수정판: {fixed_fail}건 실패 -> 전부 통과. 이제 안심하고 리팩터링 가능")
    assert fixed_fail == 0, "수정판은 반드시 전부 통과해야 합니다"

    # ---------------------------------------------------------
    # [5] docstring: help() 가 보여 주는 사용 설명서
    # ---------------------------------------------------------
    print("\n[5] docstring — help(net_price) 요약")
    doc = net_price.__doc__ or ""
    for line in doc.strip().splitlines()[:5]:
        print(f"  | {line.strip()}")
    print("  -> 타입 힌트 = 규격, docstring = 단위·범위·예외까지 담는 설명서.")

    print("\n[끝] '돌아가는 코드'에서 '믿을 수 있는 코드'로.")
    print("     계약(힌트) + 체크리스트(테스트) + 설명서(docstring) = 업무용 코드의 3종 세트.")


if __name__ == "__main__":
    main()
