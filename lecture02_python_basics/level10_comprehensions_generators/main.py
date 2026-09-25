"""컴프리헨션·제너레이터·람다·데코레이터 — 파이썬다운 표현 맛보기.

[1] 매출 데이터 변환을 루프 vs 컴프리헨션으로 비교하고,
[2] 거래 100만 건을 리스트(창고) vs 제너레이터(수도꼭지)로 처리해
    메모리 사용량을 실측 비교합니다.
[3] 람다로 정렬 기준을 전달하고 [4] 데코레이터 도장을 맛봅니다.
"""

import random
import sys
import time


# =============================================================
# [4]용 데코레이터: 실행 시간을 재는 '결재 도장'
# =============================================================
def stopwatch(func):
    """함수를 받아 '시간 측정 기능이 덧입혀진 함수'를 돌려준다."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)          # 원래 함수는 그대로 실행
        elapsed = time.perf_counter() - start
        print(f"    (stopwatch: {func.__name__} 실행 {elapsed * 1000:.1f}ms)")
        return result
    return wrapper


@stopwatch                       # = sum_with_list = stopwatch(sum_with_list)
def sum_with_list(n):
    """거래 n건을 리스트로 전부 만든 뒤 합산 (창고 방식)."""
    rows = [i % 1000 * 100 for i in range(n)]   # n건을 메모리에 몽땅 적재
    return sum(rows), sys.getsizeof(rows)


@stopwatch
def sum_with_generator(n):
    """거래 n건을 제너레이터로 흘려보내며 합산 (수도꼭지 방식)."""
    stream = (i % 1000 * 100 for i in range(n))  # 아직 아무것도 안 만듦
    return sum(stream), sys.getsizeof(stream)


def transaction_stream(n):
    """yield 데모: 거래를 한 건씩 내보내는 제너레이터 함수."""
    for i in range(n):
        yield {"id": i, "amount": (i * 37) % 900 * 1000}   # 한 건 주고 멈춰 대기


def main():
    print("=" * 56)
    print(" 컴프리헨션·제너레이터·람다·데코레이터")
    print("=" * 56)

    random.seed(42)
    # 지점별 거래 데이터 (딕셔너리들의 리스트)
    rows = [
        {"store": store, "amount": random.randint(3, 80) * 10000}
        for store in ["강남", "서초", "판교", "분당", "일산", "수원", "인천", "부산"]
    ]

    # ---------------------------------------------------------
    # [1] 루프 vs 컴프리헨션: 같은 변환, 두 가지 표기
    # ---------------------------------------------------------
    print("\n[1] 루프 vs 컴프리헨션 — 10만 원 이상 거래에 부가세 반영")

    # 루프 방식 (4줄)
    loop_result = []
    for r in rows:
        if r["amount"] >= 100000:
            loop_result.append(int(r["amount"] * 1.1))

    # 컴프리헨션 방식 (1줄): "10만 이상인 r 각각의 금액 x1.1 목록"
    comp_result = [int(r["amount"] * 1.1) for r in rows if r["amount"] >= 100000]

    print(f"  루프 방식(4줄)      : {loop_result}")
    print(f"  컴프리헨션(1줄)     : {comp_result}")
    print(f"  두 결과가 같은가?   : {loop_result == comp_result}")

    # 딕셔너리 컴프리헨션: 지점 -> 금액 표 만들기
    by_store = {r["store"]: r["amount"] for r in rows}
    print(f"  딕셔너리 컴프리헨션 : {by_store}")

    # ---------------------------------------------------------
    # [2] 제너레이터: 대용량 스트림의 메모리 감각
    # ---------------------------------------------------------
    print("\n[2] 리스트(창고) vs 제너레이터(수도꼭지) — 거래 100만 건 합산")

    N = 1_000_000
    total_l, mem_l = sum_with_list(N)
    print(f"  리스트 방식     합계 {total_l:>13,} / 메모리 {mem_l:>10,} bytes (~{mem_l / 1e6:.1f}MB)")
    total_g, mem_g = sum_with_generator(N)
    print(f"  제너레이터 방식 합계 {total_g:>13,} / 메모리 {mem_g:>10,} bytes")
    print(f"  -> 같은 답, 메모리 차이 약 {mem_l // mem_g:,}배. 데이터가 클수록 수도꼭지!")

    # yield 동작을 눈으로: 앞 3건만 꺼내 보기
    tap = transaction_stream(N)                  # 아직 한 건도 안 만들어짐
    print("  yield 데모(필요한 만큼만 꺼냄):")
    for _ in range(3):
        print(f"    next() -> {next(tap)}")
    print("    ... 나머지는 만들지도 않았으므로 비용 0")

    # 한 번 소비한 제너레이터는 재사용 불가
    small = (x for x in range(3))
    print(f"  1차 소비: {list(small)} / 2차 소비: {list(small)}  <- 빈 결과(한 번 쓰면 끝)")

    # ---------------------------------------------------------
    # [3] 람다: 정렬·최대의 '기준'을 포스트잇으로 전달
    # ---------------------------------------------------------
    print("\n[3] 람다 — key 인자에 기준 전달")

    top3 = sorted(rows, key=lambda r: r["amount"], reverse=True)[:3]
    print("  매출 상위 3개 지점:")
    for rank, r in enumerate(top3, start=1):
        print(f"    {rank}위 {r['store']} {r['amount']:,}원")

    best = max(rows, key=lambda r: r["amount"])
    worst = min(rows, key=lambda r: r["amount"])
    print(f"  최고 {best['store']} {best['amount']:,}원 / 최저 {worst['store']} {worst['amount']:,}원")

    # 제너레이터식 + 람다 없이도 되는 집계: sum(식 for ...)
    big_total = sum(r["amount"] for r in rows if r["amount"] >= 300000)
    print(f"  30만 원 이상 거래 합계(제너레이터식): {big_total:,}원 (중간 리스트 없음)")

    # ---------------------------------------------------------
    # [4] 데코레이터 정리
    # ---------------------------------------------------------
    print("\n[4] 데코레이터 — 이미 [2]에서 도장이 찍혀 있었습니다")
    print("  sum_with_list / sum_with_generator 위의 @stopwatch 가 데코레이터.")
    print("  함수 본문은 한 줄도 안 고치고 '실행 시간 출력' 기능이 덧입혀졌습니다.")
    print("  @도장 표기 = 함수를 감싸는 함수. 로그·권한 확인 등 공통 절차에 씁니다.")

    print("\n[끝] 컴프리헨션은 '한 문장으로 읽힐 때만', 제너레이터는 '데이터가 클 때',")
    print("     람다는 '기준 전달용 포스트잇', 데코레이터는 '읽을 줄 알면 충분'.")


if __name__ == "__main__":
    main()
