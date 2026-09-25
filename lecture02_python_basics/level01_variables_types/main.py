"""변수와 자료형 — 카페 매출 계산 예제.

int/float/str/bool 네 가지 기본 자료형을 만들어 보고,
변수로 매출을 계산하고, 자료형 변환 실수를 재현하고,
f-string 서식으로 사람이 읽기 좋은 리포트를 출력합니다.
"""


def main():
    print("=" * 52)
    print(" 변수와 자료형 — 카페 하루 매출 계산")
    print("=" * 52)

    # ---------------------------------------------------------
    # [1] 변수 만들기: 이름표 붙은 상자에 값을 넣는다
    # ---------------------------------------------------------
    print("\n[1] 변수 만들기와 자료형 확인")

    menu_name = "아메리카노"     # str  : 문자열
    unit_price = 4500            # int  : 정수 (원 단위 가격)
    cups_sold = 120              # int  : 판매 잔 수
    vat_rate = 0.1               # float: 부가세율 10%
    is_open = True               # bool : 오늘 영업 중인가

    print(f"  menu_name  = {menu_name!r:14} -> {type(menu_name).__name__}")
    print(f"  unit_price = {unit_price!r:14} -> {type(unit_price).__name__}")
    print(f"  cups_sold  = {cups_sold!r:14} -> {type(cups_sold).__name__}")
    print(f"  vat_rate   = {vat_rate!r:14} -> {type(vat_rate).__name__}")
    print(f"  is_open    = {is_open!r:14} -> {type(is_open).__name__}")

    # ---------------------------------------------------------
    # [2] 계산: '=' 는 "오른쪽 값을 왼쪽 상자에 넣어라"
    # ---------------------------------------------------------
    print("\n[2] 매출 계산 (순차 실행)")

    revenue = unit_price * cups_sold        # 매출 = 단가 x 수량
    vat = int(revenue * vat_rate)           # 부가세 (원 단위 정수로)
    total_with_vat = revenue + vat          # 부가세 포함 합계

    print(f"  매출        = {unit_price} x {cups_sold} = {revenue}원")
    print(f"  부가세(10%) = {vat}원")
    print(f"  합계        = {total_with_vat}원")

    # count = count + 1 형태: 지금 값에 더해서 다시 넣기
    cups_sold = cups_sold + 5               # 마감 직전 5잔 추가 판매
    print(f"  추가 판매 후 cups_sold = {cups_sold} (기존 값에 5를 더해 다시 넣음)")

    # ---------------------------------------------------------
    # [3] 자료형 변환: 실무 사고 1순위 "문자 숫자"
    # ---------------------------------------------------------
    print("\n[3] 자료형 변환 — 문자열 숫자의 함정")

    typed_price = "4500"        # CSV/입력에서 읽은 값은 이렇게 문자열로 들어온다
    typed_qty = "2"

    wrong = typed_price * 2                  # 문자열 * 2 = 이어붙이기!
    print(f"  '4500' * 2          = {wrong!r}  <- 9000이 아니라 문자열 반복")

    right = int(typed_price) * int(typed_qty)  # 변환 후 계산
    print(f"  int('4500')*int('2') = {right}   <- 변환하면 정상 계산")

    # ---------------------------------------------------------
    # [4] f-string 서식으로 리포트 만들기
    # ---------------------------------------------------------
    print("\n[4] f-string 매출 리포트")

    target = 600000                              # 오늘 목표 매출
    achieve_rate = total_with_vat / target       # 달성률

    print(f"  메뉴      : {menu_name}")
    print(f"  총 매출   : {total_with_vat:,}원 (부가세 포함)")   # 천 단위 쉼표
    print(f"  목표      : {target:,}원")
    print(f"  달성률    : {achieve_rate:.1%}")                    # 백분율 소수 1자리
    print(f"  잔당 단가 : {unit_price:,.0f}원")                   # 소수 없는 쉼표 서식

    # ---------------------------------------------------------
    # [5] float 의 미세 오차
    # ---------------------------------------------------------
    print("\n[5] float 오차 확인")

    result = 0.1 + 0.2
    print(f"  0.1 + 0.2            = {result}  <- 정확히 0.3이 아님(2진법 저장 한계)")
    print(f"  round(0.1 + 0.2, 2)  = {round(result, 2)}  <- 반올림으로 해결")
    print("  금액 계산은 가급적 정수(원 단위)로 하는 것이 안전합니다.")

    print("\n[끝] 변수 = 이름표 붙은 상자, 자료형 = 상자 속 내용물의 종류.")


if __name__ == "__main__":
    main()
