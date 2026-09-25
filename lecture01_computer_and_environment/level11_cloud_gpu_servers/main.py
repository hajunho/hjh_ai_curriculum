"""
Lecture 01 / Level 11 — 원격 서버·클라우드·GPU 환경

GPU 가 AI 에 필수인 이유인 '병렬성'을 CPU 코어 몇 개로 직접 체감합니다.
쪼개기 좋은 계산 일감(구간별 소수 개수 세기)을
(a) 혼자(단일 프로세스) vs (b) 나눠서(멀티프로세스) 처리해 시간을 비교하고,
이 원리가 일꾼 수천 명짜리 GPU 와 클라우드 비용으로 어떻게 이어지는지 봅니다.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

# 일감 설정 (직접 해보기에서 바꿔 보세요)
RANGE_END = 400_000   # 2 ~ 이 숫자까지에서 소수를 셉니다
N_CHUNKS = 8          # 일감을 몇 조각으로 쪼갤지
N_WORKERS = 4         # 동시에 일할 일꾼(프로세스) 수


def count_primes(bounds):
    """구간 [start, end) 의 소수 개수를 셉니다 — 쪼개기 좋은 단순 계산 일감."""
    start, end = bounds
    count = 0
    for n in range(max(start, 2), end):
        is_prime = True
        divisor = 2
        while divisor * divisor <= n:     # 제곱근까지만 나눠 보면 충분
            if n % divisor == 0:
                is_prime = False
                break
            divisor += 1
        if is_prime:
            count += 1
    return count


def make_chunks(end, n_chunks):
    """[1] 일감 준비: 전체 구간을 n 조각으로 나눕니다 (항상 같은 일감 — 재현성)."""
    size = end // n_chunks
    return [(i * size, end if i == n_chunks - 1 else (i + 1) * size) for i in range(n_chunks)]


def solo_run(chunks):
    """[2] 혼자 일하기: 한 프로세스가 조각을 차례로 처리."""
    started = time.perf_counter()
    total = sum(count_primes(chunk) for chunk in chunks)
    return total, time.perf_counter() - started


def team_run(chunks, n_workers):
    """[3] 나눠 일하기: 여러 프로세스가 조각을 동시에 처리."""
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(count_primes, chunks))  # 핵심 한 줄!
    return sum(results), time.perf_counter() - started


def main():
    print("=" * 60)
    print("병렬성 체험 — 혼자 vs 나눠서, 그리고 GPU 로 가는 길")
    print("=" * 60)

    cores = os.cpu_count()
    print(f"\n[1] 일감 준비: 2~{RANGE_END:,} 구간의 소수 개수 세기를 {N_CHUNKS}조각으로 분할")
    print(f"    이 컴퓨터의 CPU 코어(전문가 자리) 수: {cores}개 / 오늘 일꾼 수: {N_WORKERS}명")
    chunks = make_chunks(RANGE_END, N_CHUNKS)

    print("\n[2] 혼자 일하기 — 단일 프로세스가 조각을 차례로 처리")
    solo_total, solo_time = solo_run(chunks)
    print(f"    찾은 소수: {solo_total:,}개 / 걸린 시간: {solo_time:.2f}초")

    print(f"\n[3] 나눠 일하기 — {N_WORKERS}명의 일꾼(프로세스)이 동시에 처리")
    team_total, team_time = team_run(chunks, N_WORKERS)
    print(f"    찾은 소수: {team_total:,}개 / 걸린 시간: {team_time:.2f}초")
    print(f"    결과 일치 확인: {'같음 — 일을 쪼개도 답은 같습니다' if solo_total == team_total else '다름?!'}")

    # [4] 성적표: 배속과, 이론만큼 안 나오는 이유
    speedup = solo_time / team_time if team_time > 0 else float("inf")
    print("\n[4] 성적표")
    print(f"    배속: {speedup:.2f}배 (일꾼 {N_WORKERS}명 투입)")
    print(f"    이론 최대 {N_WORKERS}배에 못 미치는 이유:")
    print("      - 일꾼 소집 비용: 프로세스를 만들고 일감을 나눠 주는 데도 시간이 듦")
    print("      - 순차 구간: 결과 취합처럼 쪼갤 수 없는 일이 남음 (암달의 법칙)")
    print("      - 자리 한계: 코어 수보다 일꾼이 많아도 동시에 앉을 자리가 없음")

    # [5] GPU 와 클라우드로의 확장
    print("\n[5] 이 원리의 끝에 GPU 가 있습니다")
    print("    - 오늘: 일꾼 4명(CPU 코어) — AI 학습: 일꾼 수천 명(GPU 코어)")
    print("    - AI 학습 = 의존성 없는 곱셈·덧셈의 바다 -> 쪼개기 좋은 일감의 극단")
    print("    - 접속은 ssh(보안 전화선), 파일 전송은 scp(택배), 서버는 대부분 리눅스")
    print("\n    클라우드 견적 감각 (가상의 예):")
    hourly_big, hourly_small = 30_000, 1_000
    plan_a = hourly_big * 48
    plan_b = hourly_small * 3 * 2 + hourly_big * 48
    print(f"      계획 A: 대형 GPU(시간당 {hourly_big:,}원) 바로 48시간 = {plan_a:,}원")
    print(f"              (설정 실수 시 재실행 -> 최대 {plan_a * 2:,}원)")
    print(f"      계획 B: 소형(시간당 {hourly_small:,}원) 사전실험 3시간x2회 후 본학습")
    print(f"              = {plan_b:,}원 — 실수를 6천 원짜리 실험에서 미리 잡습니다")
    print("      교훈: 실험은 싸게, 본 학습만 비싸게. 그리고 인스턴스는 끄기!")

    print("\n정리: 병렬화의 판단 기준은 '일을 쪼갤 수 있는가'입니다.")
    print("      쪼갤 수 있는 일감 + 일꾼 수천 명(GPU) = AI 시대의 계산력입니다.")


if __name__ == "__main__":
    main()
