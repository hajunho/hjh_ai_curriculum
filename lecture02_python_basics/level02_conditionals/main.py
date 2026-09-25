"""조건문(if/elif/else) — 출장경비 자동 승인 판정기.

사내 경비 규정을 if/elif/else 사다리로 옮겨 자동 판정을 해 봅니다.
비교 연산자(==, <=, ...)와 논리 연산자(and/or/not)의 True/False 를 확인하고,
조건 '순서'가 규정의 우선순위라는 것을 실험으로 증명합니다.
"""


def judge_expense(amount, has_receipt):
    """경비 1건을 사내 규정에 따라 판정한다.

    규정(위에서부터 처음 해당되는 조항 하나만 적용):
      1) 5만 원 이하            -> 자동 승인
      2) 15만 원 이하 + 영수증  -> 팀장 전결
      3) 50만 원 이하           -> 본부장 결재
      4) 그 외                  -> 반려(소명 요청)
    """
    if amount <= 50000:                       # 조항 1
        return "자동 승인"
    elif amount <= 150000 and has_receipt:    # 조항 2 (and: 둘 다 참이어야)
        return "팀장 전결"
    elif amount <= 500000:                    # 조항 3
        return "본부장 결재"
    else:                                     # 조항 4: 위 모두 해당 없음
        return "반려(소명 요청)"


def judge_wrong_order(amount, has_receipt):
    """일부러 '넓은 조건을 먼저' 검사하는 잘못된 규정. 순서의 중요성 비교용."""
    if amount <= 500000:                      # 넓은 조건이 맨 위에 오면...
        return "본부장 결재"
    elif amount <= 150000 and has_receipt:    # 이 조항은 영원히 실행되지 않는다!
        return "팀장 전결"
    elif amount <= 50000:                     # 이 조항도 마찬가지
        return "자동 승인"
    else:
        return "반려(소명 요청)"


def main():
    print("=" * 56)
    print(" 조건문 — 출장경비 자동 승인 판정기")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 단일 판정: 경비 1건을 규정에 태워 보기
    # ---------------------------------------------------------
    print("\n[1] 단일 판정")
    amount = 120000          # 신청 금액(원)
    has_receipt = True       # 영수증 제출 여부

    print(f"  신청 금액 {amount:,}원 / 영수증 {'있음' if has_receipt else '없음'}")
    print(f"  -> 판정: {judge_expense(amount, has_receipt)}")
    print("  (5만 초과라 조항1 탈락 -> 15만 이하+영수증이라 조항2에서 확정, 이후 조항은 안 봄)")

    # ---------------------------------------------------------
    # [2] 비교·논리 연산자 실험실
    # ---------------------------------------------------------
    print("\n[2] 비교·논리 연산자의 True/False")
    print(f"  amount <= 150000        -> {amount <= 150000}")
    print(f"  amount == 120000        -> {amount == 120000}")
    print(f"  amount != 120000        -> {amount != 120000}")
    print(f"  amount <= 150000 and has_receipt -> {amount <= 150000 and has_receipt}")
    print(f"  amount <= 50000 or has_receipt   -> {amount <= 50000 or has_receipt}")
    print(f"  not has_receipt         -> {not has_receipt}")
    print(f"  50000 < amount <= 150000 (범위 비교) -> {50000 < amount <= 150000}")
    grade = "VIP"
    print(f"  grade in ('VIP','VVIP') -> {grade in ('VIP', 'VVIP')}")

    # ---------------------------------------------------------
    # [3] 조건 '순서'의 중요성: 같은 조항, 다른 순서
    # ---------------------------------------------------------
    print("\n[3] 순서를 바꾸면 규정이 망가진다")
    test_amount = 30000      # 원래는 '자동 승인'이어야 할 금액
    ok = judge_expense(test_amount, True)
    bad = judge_wrong_order(test_amount, True)
    print(f"  3만 원 경비, 올바른 순서(좁은 조건 먼저) -> {ok}")
    print(f"  3만 원 경비, 잘못된 순서(넓은 조건 먼저) -> {bad}")
    print("  -> 넓은 조건이 위에 있으면 아래의 엄격한 조항은 영원히 실행되지 않습니다.")

    # ---------------------------------------------------------
    # [4] 일괄 판정: 신청 6건을 같은 규정으로 처리 (반복문 맛보기)
    # ---------------------------------------------------------
    print("\n[4] 이번 주 경비 신청 6건 일괄 판정")

    # (신청자, 금액, 영수증 여부)
    requests = [
        ("김주임", 32000, True),
        ("이과장", 120000, True),
        ("박대리", 120000, False),   # 같은 금액, 영수증 없음 -> 결과가 다름
        ("최부장", 480000, True),
        ("정사원", 750000, True),
        ("한대리", 50000, False),
    ]

    approved = 0     # 자동 승인 건수
    escalated = 0    # 결재 필요 건수
    rejected = 0     # 반려 건수

    for name, amt, receipt in requests:
        decision = judge_expense(amt, receipt)
        print(f"  {name} | {amt:>8,}원 | 영수증 {'O' if receipt else 'X'} -> {decision}")
        if decision == "자동 승인":
            approved += 1
        elif decision == "반려(소명 요청)":
            rejected += 1
        else:
            escalated += 1

    print(f"\n  집계: 자동 승인 {approved}건 / 결재 필요 {escalated}건 / 반려 {rejected}건")
    print("\n[끝] 규정 문서의 조항 = if/elif/else 가지. 조항 순서 = 조건 순서.")


if __name__ == "__main__":
    main()
