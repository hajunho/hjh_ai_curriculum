"""
NumPy 배열 기초 — 파이썬 반복문 vs 벡터화 속도를 직접 측정하고,
매출 데이터를 ndarray 로 바꿔 shape/dtype·집계·브로드캐스팅·불리언 마스크를 익힙니다.
핵심 메시지: '숫자 하나'가 아니라 '숫자 묶음'을 연산의 단위로 삼으면
코드는 짧아지고 속도는 수십 배 빨라집니다.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

N_SPEED = 1_000_000     # 속도 실험에 쓸 숫자 개수


def python_sum_of_squares(values: list[float]) -> float:
    """파이썬 반복문 방식 — 한 개씩 집어서 제곱하고 누적합니다."""
    total = 0.0
    for v in values:
        total += v * v
    return total


def main() -> None:
    rng = np.random.default_rng(42)          # seed 고정 → 항상 같은 난수

    print(f"[1] 속도 대결 — {N_SPEED:,}개 숫자의 제곱 합")
    data_arr = rng.random(N_SPEED)           # 0~1 사이 실수 100만 개
    data_list = data_arr.tolist()            # 같은 값을 파이썬 리스트로도 준비

    t0 = time.perf_counter()
    loop_result = python_sum_of_squares(data_list)
    loop_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_result = float((data_arr * data_arr).sum())   # 벡터화: 반복문 없음
    vec_sec = time.perf_counter() - t0

    print(f"    파이썬 반복문: {loop_sec * 1000:8.1f} ms")
    print(f"    NumPy 벡터화 : {vec_sec * 1000:8.1f} ms")
    print(f"    → 약 {loop_sec / vec_sec:.0f}배 빠릅니다. "
          f"(두 답의 차이 {abs(loop_result - vec_result):.6f} → 같은 계산입니다)")
    print()

    print("[2] 매출 데이터를 배열로 — shape 과 dtype 확인")
    rows = hjh_data.sales_table(n_days=90, seed=42)
    # 결측(None)을 제외하고 매출만 뽑아 배열로 만듭니다
    rev = np.array([r["revenue"] for r in rows if r["revenue"] is not None])
    weekend_mask_src = [r["weekday"] in ("토", "일")
                        for r in rows if r["revenue"] is not None]
    is_weekend = np.array(weekend_mask_src)
    n_missing = len(rows) - len(rev)
    print(f"    원본 {len(rows)}행 중 결측 {n_missing}건 제외 → 배열 {len(rev)}개")
    print(f"    rev.shape = {rev.shape}, rev.dtype = {rev.dtype}")
    print()

    print("[3] 집계 한 줄 — 반복문 없이 통계 뽑기")
    print(f"    합계: {rev.sum():>16,}원")
    print(f"    평균: {rev.mean():>16,.0f}원")
    print(f"    표준편차: {rev.std():>12,.0f}원")
    print(f"    최대: {rev.max():>16,}원 / 최소: {rev.min():,}원 (음수 오염 포함)")
    print()

    print("[4] 브로드캐스팅 — 부가세 10% 를 '도장 한 번'으로")
    with_vat = rev * 1.1                       # 숫자 하나가 배열 전체로 확장
    rounded = np.round(with_vat, -3)           # 천 원 단위 반올림도 벡터로
    print(f"    세전 평균: {rev.mean():>14,.0f}원")
    print(f"    세후 평균: {with_vat.mean():>14,.0f}원 (= 세전 × 1.1)")
    print(f"    천 원 반올림 예시: {rev[:3]} → {rounded[:3].astype(int)}")
    print()

    print("[5] 불리언 마스크 — 조건으로 골라내기")
    weekend_avg = rev[is_weekend].mean()       # True 위치의 값만 선택
    weekday_avg = rev[~is_weekend].mean()      # ~ 는 True/False 뒤집기
    print(f"    주말 평균: {weekend_avg:>14,.0f}원 ({int(is_weekend.sum())}건)")
    print(f"    평일 평균: {weekday_avg:>14,.0f}원 ({int((~is_weekend).sum())}건)")
    print(f"    → 주말이 평일의 {weekend_avg / weekday_avg:.2f}배")
    negative = rev < 0
    print(f"    음수 오염 건수: {int(negative.sum())}건 "
          f"(True 개수 세기 = 마스크의 sum)")
    print(f"    음수 제외 평균: {rev[~negative].mean():,.0f}원")
    print()
    print("    다음 레벨에서는 이 배열에 '열 이름'을 입힌 DataFrame 을 배웁니다.")


if __name__ == "__main__":
    main()
