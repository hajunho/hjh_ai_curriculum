"""반복문(for/while/range/break/continue) — 고객 100명 구매 데이터 집계.

같은 집계를 '손계산 방식'(한 줄씩 전부 쓰기)과 '반복문 방식'으로 비교해
자동화의 이득을 확인합니다. 누적 변수 패턴, 반복문+조건문 결합,
while 시뮬레이션, break/continue, enumerate 까지 한 번에 익힙니다.
"""

import random


def main():
    print("=" * 56)
    print(" 반복문 — 고객 100명 구매 데이터 집계")
    print("=" * 56)

    # 재현성을 위해 seed 고정: 누가 실행해도 같은 데이터가 나온다
    random.seed(42)

    # 고객 100명의 이번 달 구매액(원). 일부는 0원(휴면 고객).
    purchases = []
    for i in range(100):
        if random.random() < 0.15:          # 15%는 휴면 고객
            purchases.append(0)
        else:
            purchases.append(random.randint(10, 600) * 1000)  # 1만~60만 원

    # ---------------------------------------------------------
    # [1] 손계산 방식: 반복문이 없다면?
    # ---------------------------------------------------------
    print("\n[1] 반복문이 없다면 (3명분만 흉내)")
    total_by_hand = purchases[0] + purchases[1] + purchases[2]
    print(f"  total = purchases[0] + purchases[1] + purchases[2]  # = {total_by_hand:,}원")
    print("  ... 100명이면 이런 덧셈을 100항 쓰거나 100줄 나열해야 합니다.")
    print("  고객이 101명이 되는 순간 코드도 고쳐야 합니다.")

    # ---------------------------------------------------------
    # [2] for 반복문: 4줄로 100명 집계
    # ---------------------------------------------------------
    print("\n[2] for 반복문으로 집계")

    total = 0                       # 누적 변수는 반복문 '앞'에서 초기화
    best_amount = 0                 # 최고 구매액
    for amount in purchases:        # 100명분을 한 건씩 꺼내서
        total += amount             # 매 바퀴 누적
        if amount > best_amount:    # 최고 기록 갱신
            best_amount = amount

    average = total / len(purchases)
    print(f"  고객 수     : {len(purchases)}명")
    print(f"  구매 총액   : {total:,}원")
    print(f"  1인 평균    : {average:,.0f}원")
    print(f"  최고 구매액 : {best_amount:,}원")
    print("  -> 고객이 1만 명이 되어도 위 코드는 한 글자도 바뀌지 않습니다.")

    # ---------------------------------------------------------
    # [3] 반복문 + 조건문: 우수 고객 선별 (continue 활용)
    # ---------------------------------------------------------
    print("\n[3] 우수 고객 선별 (if/continue 결합)")

    VIP_THRESHOLD = 300000          # 우수 고객 기준: 30만 원 이상
    vip_count = 0
    dormant_count = 0
    for amount in purchases:
        if amount == 0:             # 휴면 고객은 건너뛰고 다음 사람으로
            dormant_count += 1
            continue
        if amount >= VIP_THRESHOLD:
            vip_count += 1

    print(f"  우수 고객(>= {VIP_THRESHOLD:,}원) : {vip_count}명")
    print(f"  휴면 고객(0원, continue 로 스킵)  : {dormant_count}명")

    # ---------------------------------------------------------
    # [4] while: 마케팅 예산 소진 시뮬레이션 (break 활용)
    # ---------------------------------------------------------
    print("\n[4] while — 예산 100만 원은 며칠 버틸까")

    budget = 1000000
    day = 0
    while budget > 0:               # 예산이 남아 있는 동안 반복
        day += 1
        spend = 60000 + random.randint(0, 50) * 1000   # 하루 지출 6만~11만 원
        budget -= spend
        if day <= 3 or budget <= 0:                    # 처음 3일과 마지막 날만 출력
            print(f"  {day:>2}일차 지출 {spend:>7,}원 -> 잔액 {max(budget, 0):>9,}원")
        if day >= 60:               # 안전장치: 60일 넘으면 강제 종료
            print("  60일 상한 도달, break 로 종료")
            break

    print(f"  -> 예산은 {day}일차에 소진되었습니다.")

    # ---------------------------------------------------------
    # [5] range 와 enumerate: 구매액 상위 5명 순위표
    # ---------------------------------------------------------
    print("\n[5] 구매액 상위 5명 (enumerate 로 순위 붙이기)")

    top5 = sorted(purchases, reverse=True)[:5]     # 내림차순 정렬 후 앞 5개
    for rank, amount in enumerate(top5, start=1):  # 1부터 순위 번호
        print(f"  {rank}위: {amount:,}원")

    print("\n  range 확인: range(5) ->", list(range(5)), "/ range(1, 6) ->", list(range(1, 6)))

    # ---------------------------------------------------------
    # 결론
    # ---------------------------------------------------------
    print("\n[끝] 코드 줄 수 비교")
    print("  손계산 방식 : 고객 수만큼 줄이 늘어남 (100명 = 100줄+)")
    print("  반복문 방식 : 항상 4줄 (데이터가 늘어도 코드는 그대로)")
    print("  '목록의 모든 항목에 같은 절차를' — 이것이 자동화의 핵심 문장입니다.")


if __name__ == "__main__":
    main()
