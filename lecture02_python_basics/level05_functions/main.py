"""함수 — 세금·할인 계산의 리팩터링 전/후 비교.

같은 결제 계산(등급 할인 + 부가세)을
[1] 복사-붙여넣기 방식(리팩터링 전, 일부러 심은 불일치 버그 포함)과
[2] 함수 방식(리팩터링 후)으로 수행해 결과를 비교합니다.
기본값 인자, 키워드 인자, 다중 반환, 스코프도 함께 시연합니다.
"""

VAT_RATE = 0.1          # 부가세율 10%


# =============================================================
# 리팩터링 '후': 로직을 함수 한 곳에만 둔다
# =============================================================
def get_discount_rate(grade):
    """고객 등급에 따른 할인율을 반환한다."""
    if grade == "VIP":
        return 0.10
    elif grade == "GOLD":
        return 0.05
    else:
        return 0.0


def calc_vat(amount):
    """부가세(10%)를 원 단위 정수로 계산한다. 세율 변경 시 여기 한 줄만 수정."""
    return int(amount * VAT_RATE)


def calc_payment(amount, grade="일반"):
    """할인과 부가세를 적용한 최종 결제액을 계산한다.

    반환: (할인액, 부가세, 최종 결제액) 튜플
    """
    discount = int(amount * get_discount_rate(grade))   # 함수가 함수에게 위임
    base = amount - discount
    vat = calc_vat(base)
    return discount, vat, base + vat


def main():
    print("=" * 56)
    print(" 함수 — 세금·할인 계산 리팩터링 전/후")
    print("=" * 56)

    # 주문 데이터: (고객명, 등급, 주문액)
    orders = [
        ("김주임", "VIP", 200000),
        ("이과장", "GOLD", 150000),
        ("박대리", "일반", 80000),
    ]

    # ---------------------------------------------------------
    # [1] 리팩터링 전: 같은 로직을 주문마다 복사-붙여넣기
    # ---------------------------------------------------------
    print("\n[1] 리팩터링 전 — 복사-붙여넣기 방식")

    before_results = []

    # --- 주문 1 (복사본 1)
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
    print(f"  {name1}: 결제액 {pay1:,}원")

    # --- 주문 2 (복사본 2 — 세율에 오타가 났다! 0.1 이 0.01 로)
    name2, grade2, amount2 = orders[1]
    if grade2 == "VIP":
        rate2 = 0.10
    elif grade2 == "GOLD":
        rate2 = 0.05
    else:
        rate2 = 0.0
    discount2 = int(amount2 * rate2)
    vat2 = int((amount2 - discount2) * 0.01)   # <- 복사하다 생긴 오타 (버그!)
    pay2 = amount2 - discount2 + vat2
    before_results.append(pay2)
    print(f"  {name2}: 결제액 {pay2:,}원   <- 어딘가 이상하지만 눈치채기 어렵다")

    # --- 주문 3 (복사본 3)
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
    print(f"  {name3}: 결제액 {pay3:,}원")
    print("  -> 같은 로직이 3벌. 세율이 바뀌면 3곳(실무라면 수십 곳)을 고쳐야 합니다.")

    # ---------------------------------------------------------
    # [2] 리팩터링 후: 함수 호출 3번
    # ---------------------------------------------------------
    print("\n[2] 리팩터링 후 — 함수 방식")

    after_results = []
    for name, grade, amount in orders:
        discount, vat, pay = calc_payment(amount, grade)    # 위임 한 줄
        after_results.append(pay)
        print(f"  {name}: 주문 {amount:,}원 - 할인 {discount:,}원 + 부가세 {vat:,}원"
              f" = 결제액 {pay:,}원")
    print("  -> 로직은 calc_payment 한 곳뿐. 세율 변경도 calc_vat 한 줄 수정으로 끝.")

    # ---------------------------------------------------------
    # [3] 전/후 결과 비교: 복사 방식의 불일치 버그 발견
    # ---------------------------------------------------------
    print("\n[3] 전/후 결과 비교")
    for (name, _, _), b, a in zip(orders, before_results, after_results):
        mark = "일치" if b == a else f"불일치! 차액 {abs(b - a):,}원 (복사본의 세율 오타)"
        print(f"  {name}: 전 {b:,}원 / 후 {a:,}원 -> {mark}")

    # ---------------------------------------------------------
    # [4] 함수 문법 정리
    # ---------------------------------------------------------
    print("\n[4] 함수 문법 미니 정리")

    # 기본값 인자: 등급을 안 주면 '일반'으로 처리
    _, _, pay_default = calc_payment(100000)
    print(f"  calc_payment(100000)            -> {pay_default:,}원 (grade 기본값 '일반')")

    # 키워드 인자: 순서 대신 이름으로 명확하게
    _, _, pay_kw = calc_payment(amount=100000, grade="VIP")
    print(f"  calc_payment(amount=..., grade='VIP') -> {pay_kw:,}원 (키워드 인자)")

    # 다중 반환: 튜플 언패킹으로 받기
    d, v, p = calc_payment(50000, "GOLD")
    print(f"  다중 반환 언패킹 -> 할인 {d:,} / 부가세 {v:,} / 결제 {p:,}")

    # 스코프: 함수 안의 지역 변수는 밖에서 보이지 않는다
    def inner_demo():
        local_memo = "함수 책상 위의 메모"       # 지역 변수
        return len(local_memo)

    inner_demo()
    try:
        print(local_memo)                         # 바깥에서 접근 시도
    except NameError as e:
        print(f"  지역 변수 바깥 접근 -> NameError: {e}")
        print("  -> 함수 안 변수는 함수가 끝나면 사라집니다('각자의 책상' 원칙).")

    print("\n[끝] 두 번 복사하는 순간이 함수로 뽑을 타이밍입니다 (DRY 원칙).")


if __name__ == "__main__":
    main()
