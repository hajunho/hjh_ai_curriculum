"""
병합(merge)·연결(concat) 실습.
매출표 + 지점정보표 + 목표표 세 개의 표를 이어 붙여
지점별 목표 달성률 순위를 만듭니다.
inner/left/outer 의 행 수 차이, indicator 검증,
키 중복이 일으키는 행 수 폭발 함정까지 직접 확인합니다.
"""

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def build_performance() -> pd.DataFrame:
    """1년치 매출을 정제하고 지점별 총실적 표를 만듭니다."""
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))  # seed 고정
    df.loc[df["revenue"] < 0, "revenue"] = np.nan  # 도메인 규칙 정제
    df = df.dropna(subset=["revenue"])
    perf = (df.groupby("store", as_index=False)
              .agg(actual=("revenue", "sum")))
    return df, perf


def main() -> None:
    # ------------------------------------------------------------------
    print("[1] 지점별 실적 집계 (level07 복습)")
    df, perf = build_performance()
    perf["actual_mil"] = (perf["actual"] / 1e6).round(1)  # 백만원 단위
    print(perf[["store", "actual_mil"]].to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[2] 보조 표 준비 — 총무팀 명부와 기획팀 목표표 (코드로 직접 생성)")
    # 일부러 부산점을 빼고, 매출표에 없는 신규 '판교점'을 넣었습니다.
    store_info = pd.DataFrame({
        "store": ["강남점", "홍대점", "대전점", "수원점", "판교점"],
        "region": ["수도권", "수도권", "충청권", "수도권", "수도권"],
        "open_year": [2015, 2017, 2019, 2021, 2025],
        "manager": ["김영주", "박선호", "이도윤", "최하은", "정민재"],
    })
    targets = pd.DataFrame({
        "store": ["강남점", "홍대점", "부산점", "대전점", "수원점"],
        "target_mil": [800, 850, 900, 1000, 1100],  # 연간 목표(백만원)
    })
    print(f"  store_info {len(store_info)}행 (부산점 누락, 판교점 신규)")
    print(f"  targets    {len(targets)}행")

    # ------------------------------------------------------------------
    print("\n[3] how 4종 비교 — 같은 두 표, 다른 행 수")
    for how in ["inner", "left", "outer"]:
        merged = pd.merge(perf, store_info, on="store", how=how)
        print(f"  how='{how:5s}' -> {len(merged)}행")
    print("  (right 는 좌우를 바꾼 left 와 같으므로 생략)")
    audit = pd.merge(perf, store_info, on="store", how="outer", indicator=True)
    print("  indicator 검증 (_merge 열):")
    print(audit["_merge"].value_counts().to_string())
    only_left = audit.loc[audit["_merge"] == "left_only", "store"].tolist()
    only_right = audit.loc[audit["_merge"] == "right_only", "store"].tolist()
    print(f"  -> 매출은 있는데 명부에 없음: {only_left} / 명부에만 있음: {only_right}")

    # ------------------------------------------------------------------
    print("\n[4] 목표표 병합 -> 달성률 계산과 순위")
    result = pd.merge(perf, targets, on="store", how="left")
    result["achieve_pct"] = (result["actual_mil"] / result["target_mil"] * 100).round(1)
    result = result.sort_values("achieve_pct", ascending=False).reset_index(drop=True)
    result.index = result.index + 1  # 1위부터 표시
    print(result[["store", "actual_mil", "target_mil", "achieve_pct"]].to_string())
    best = result.iloc[0]
    print(f"  -> 1위 {best['store']}: 목표 대비 {best['achieve_pct']}% 달성")

    # ------------------------------------------------------------------
    print("\n[5] 키 중복 함정 — 행이 소리 없이 불어난다")
    dup_targets = pd.concat(
        [targets, targets.iloc[[0]]], ignore_index=True)  # 강남점 목표가 2줄!
    print(f"  오염된 목표표: {len(dup_targets)}행 "
          f"(강남점 중복 {int(dup_targets['store'].duplicated().sum())}건)")
    boom = pd.merge(perf, dup_targets, on="store", how="left")
    print(f"  병합 결과: {len(perf)}행 -> {len(boom)}행 (강남점이 2줄로 복제!)")
    total_ok = perf["actual_mil"].sum()
    total_boom = boom["actual_mil"].sum()
    print(f"  총실적 합계: 정상 {total_ok:,.1f} vs 오염 {total_boom:,.1f} 백만원"
          f" (+{total_boom - total_ok:,.1f} 뻥튀기)")
    safe = pd.merge(perf, dup_targets.drop_duplicates(subset="store"),
                    on="store", how="left")
    print(f"  예방: drop_duplicates 후 병합 -> {len(safe)}행 (정상 복구)")
    print("  습관: 병합 전 duplicated() 확인 + 병합 전후 len() 비교!")

    # ------------------------------------------------------------------
    print("\n[6] concat — 옆으로 잇기가 아니라 '아래로 쌓기'")
    first_half = (df[df["day_index"] < 182].groupby("store", as_index=False)
                  .agg(revenue_mil=("revenue", "sum")))
    second_half = (df[df["day_index"] >= 182].groupby("store", as_index=False)
                   .agg(revenue_mil=("revenue", "sum")))
    first_half["half"], second_half["half"] = "상반기", "하반기"
    stacked = pd.concat([first_half, second_half], ignore_index=True)
    stacked["revenue_mil"] = (stacked["revenue_mil"] / 1e6).round(1)
    print(f"  상반기 {len(first_half)}행 + 하반기 {len(second_half)}행"
          f" = concat {len(stacked)}행 (같은 열 구조를 세로로 쌓음)")
    print(stacked.head(3).to_string(index=False))
    print("\n정리: 기준 표가 있으면 left, 감사할 때는 outer+indicator,")
    print("      병합 전 중복 키 점검 — 이 세 가지가 병합 사고를 막습니다.")


if __name__ == "__main__":
    main()
