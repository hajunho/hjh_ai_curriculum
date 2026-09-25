"""
대용량 데이터 처리 전략 실습.
8년 치 매출 데이터(73,000행)를 놓고
  - memory_usage(deep=True) 로 열별 메모리 측정
  - dtype 다이어트 (category / int32 / float32) 전후 비교
  - read_csv(chunksize=...) 청크 스트리밍 집계와 결과 검증
  - usecols/dtype 지정 읽기 최적화 시간·메모리 비교
를 수행하고, "언제 pandas 를 떠나는가" 기준을 정리합니다.
"""

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

# 공용 데이터 모듈(hjh_data)을 불러오기 위한 경로 설정
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
CSV_PATH = OUT_DIR / "big_sales.csv"


def fmt_bytes(n: float) -> str:
    """바이트를 사람이 읽기 좋은 단위로 바꿉니다."""
    return f"{n / 1024 / 1024:.2f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"


def step1_measure(df: pd.DataFrame) -> int:
    """[1] 체중 재기: 열별 메모리 사용량을 deep=True 로 측정합니다."""
    print(f"\n[1] memory_usage(deep=True) — 데이터 {len(df):,}행의 열별 메모리")
    mem = df.memory_usage(deep=True)
    for col, n_bytes in mem.items():
        dtype = "-" if col == "Index" else str(df[col].dtype)
        print(f"    {str(col):10s} ({dtype:7s}): {fmt_bytes(n_bytes):>10s}")
    total = int(mem.sum())
    print(f"    합계: {fmt_bytes(total)}")
    print("    -> 문자열(object) 열이 숫자 열보다 몇 배 무거운 '메모리 하마'입니다.")
    return total


def step2_dtype_diet(df: pd.DataFrame, before: int) -> pd.DataFrame:
    """[2] dtype 다이어트: category / int32 / float32 로 바꾸고 절감률을 계산합니다."""
    print("\n[2] dtype 다이어트 — 같은 데이터, 더 작은 옷")
    opt = df.copy()
    # 반복이 많은 문자열 -> category (사전 + 정수 코드로 압축)
    for col in ["date", "weekday", "store", "category"]:
        opt[col] = opt[col].astype("category")
    # 값 범위가 int32 에 충분히 들어가는 정수 열 -> 절반 크기로
    opt["day_index"] = opt["day_index"].astype("int32")
    opt["ad_cost"] = opt["ad_cost"].astype("int32")
    # 분석용 실수 열 -> float32 (회계 보고용이면 float64 유지가 안전)
    opt["revenue"] = opt["revenue"].astype("float32")

    after = int(opt.memory_usage(deep=True).sum())
    saving = (1 - after / before) * 100
    print(f"    최적화 전: {fmt_bytes(before)}")
    print(f"    최적화 후: {fmt_bytes(after)}  (절감률 {saving:.1f}%)")

    # 값이 훼손되지 않았는지 검증 (category 는 원래 문자열로 되돌려 비교)
    assert (opt["store"].astype(str) == df["store"]).all()
    assert (opt["day_index"].astype("int64") == df["day_index"]).all()
    ok = np.allclose(opt["revenue"].astype("float64"), df["revenue"], equal_nan=True)
    print(f"    검증: 문자열·정수 열 완전 일치, revenue 는 float32 정밀도 내 일치 ({ok})")
    print("    -> category 의 원리: '강남점'을 73,000번 쓰는 대신 사전에 1번 쓰고 번호만 저장")
    return opt


def step3_chunked_aggregation(df: pd.DataFrame) -> None:
    """[3] 청크 스트리밍 집계: 나눠 읽어도 전체 집계와 같음을 검증합니다."""
    print("\n[3] read_csv(chunksize) — 메모리보다 큰 파일을 다루는 패턴")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"    CSV 저장: {CSV_PATH} ({fmt_bytes(CSV_PATH.stat().st_size)})")

    # 청크 순회: 부분 집계(합계)를 누적해서 합칩니다.
    total_by_store = None
    n_chunks = 0
    for chunk in pd.read_csv(CSV_PATH, chunksize=10_000):
        part = chunk.groupby("store")["revenue"].sum()
        total_by_store = part if total_by_store is None else total_by_store.add(part, fill_value=0)
        n_chunks += 1
    print(f"    10,000행씩 {n_chunks}개 청크로 나눠 지점별 합계를 누적 집계했습니다.")

    # 한 번에 읽은 결과와 비교 검증
    full = pd.read_csv(CSV_PATH).groupby("store")["revenue"].sum()
    match = np.allclose(total_by_store.sort_index(), full.sort_index())
    print(f"    검증: 청크 집계 == 전체 집계 ? {match}")
    top = total_by_store.sort_values(ascending=False)
    print("    지점별 매출 합계(억원): "
          + ", ".join(f"{s} {v / 1e8:.0f}" for s, v in top.items()))
    print("    -> 합·개수·최대최소는 이 패턴 OK, 중앙값처럼 전체가 필요한 통계는 불가.")


def step4_read_optimized() -> None:
    """[4] 읽기 최적화: usecols + dtype 지정의 시간·메모리 효과를 비교합니다."""
    print("\n[4] 읽기 최적화 — 필요한 열만, 알맞은 타입으로")
    t0 = time.perf_counter()
    naive = pd.read_csv(CSV_PATH)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    smart = pd.read_csv(CSV_PATH, usecols=["store", "revenue"],
                        dtype={"store": "category", "revenue": "float32"})
    t_smart = time.perf_counter() - t0

    mem_naive = naive.memory_usage(deep=True).sum()
    mem_smart = smart.memory_usage(deep=True).sum()
    print(f"    전체 읽기      : {t_naive * 1000:6.0f} ms / 메모리 {fmt_bytes(mem_naive)}")
    print(f"    usecols+dtype  : {t_smart * 1000:6.0f} ms / 메모리 {fmt_bytes(mem_smart)}"
          f"  (메모리 {(1 - mem_smart / mem_naive) * 100:.0f}% 절감)")
    print("    -> 파일이 작으면 시간 차이는 작지만, 메모리 절감은 언제나 확실합니다.")
    # 참고: 파케이(Parquet) 열 지향 포맷이라면 아래 한 줄로 끝납니다 (pyarrow 필요).
    #   df.to_parquet("big_sales.parquet")
    #   pd.read_parquet("big_sales.parquet", columns=["store", "revenue"])
    # 열만 골라 읽기·타입 보존·압축이 내장되어 반복 사용 데이터의 표준입니다.


def step5_when_to_leave_pandas() -> None:
    """[5] 판단 기준: 언제 pandas 를 떠나 DB/Spark 로 가는가."""
    print("\n[5] 언제 pandas 를 떠나는가 — 판단 기준 요약")
    rules = [
        ("데이터가 메모리의 1/3 이하", "그냥 pandas"),
        ("메모리에 빠듯하게 들어옴", "dtype 다이어트 + 파케이 저장"),
        ("메모리보다 크지만 집계가 목적", "chunksize 스트리밍, 또는 Polars/DuckDB"),
        ("여러 사람이 동시에 조회·갱신", "데이터베이스(DB) — lecture04"),
        ("수억 행 이상, 서버 여러 대 필요", "Spark 등 분산 처리"),
    ]
    for cond, action in rules:
        print(f"    {cond:26s} -> {action}")
    print("    -> 기준은 'GB 수'가 아니라 작업의 성격입니다. 분산 시스템은 그 자체가 비용입니다.")


def main() -> None:
    print("=" * 60)
    print("Level 11 — 대용량 데이터 처리 전략")
    print("=" * 60)
    # 8년 치(2,920일) x 5개 지점 x 5개 카테고리 = 73,000행. seed 고정으로 재현 가능.
    df = pd.DataFrame(hjh_data.sales_table(n_days=2920, seed=42))
    before = step1_measure(df)
    step2_dtype_diet(df, before)
    step3_chunked_aggregation(df)
    step4_read_optimized()
    step5_when_to_leave_pandas()
    print("\n완료! lecture03 을 마쳤습니다. 다음은 lecture04 — 데이터베이스와 SQL 입니다.")


if __name__ == "__main__":
    main()
