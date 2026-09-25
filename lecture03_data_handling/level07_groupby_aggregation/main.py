"""
groupby 집계 실습.
1년치 카페 체인 매출을 정제한 뒤 split-apply-combine 패턴으로
지점별·요일별·카테고리별 분석 리포트를 만듭니다.
agg(다중 집계)와 transform(원본 크기 유지)의 차이,
요일 순서 재정렬(reindex)까지 실무 보고서 흐름 그대로 연습합니다.
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
WEEKDAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]


def load_clean_sales() -> pd.DataFrame:
    """매출 데이터를 불러와 level06 방식으로 정제합니다 (음수->NaN, 결측 삭제)."""
    rows = hjh_data.sales_table(n_days=365, seed=42)  # seed 고정
    df = pd.DataFrame(rows)
    df.loc[df["revenue"] < 0, "revenue"] = np.nan  # 도메인 규칙: 매출은 음수 불가
    df = df.dropna(subset=["revenue"])             # 이번 레벨은 집계가 목적이라 단순 삭제
    return df


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_clean_sales()
    print(f"정제 완료: {len(df):,}행 (음수 제거 + 결측 삭제)\n")

    # ------------------------------------------------------------------
    print("[1] 지점별 요약 — agg 로 총매출·평균·건수를 한 번에")
    store_report = (
        df.groupby("store")
          .agg(total=("revenue", "sum"),
               avg=("revenue", "mean"),
               n=("revenue", "count"))
          .sort_values("total", ascending=False)
    )
    store_report["total_mil"] = (store_report["total"] / 1e6).round(1)  # 백만원 단위
    print(store_report[["total_mil", "avg", "n"]].round(0).to_string())
    print("  -> 수원·대전 쪽이 큰 이유: 데이터에 지점 규모 효과가 심어져 있습니다.")

    # ------------------------------------------------------------------
    print("\n[2] 요일별 평균 매출 — 가나다순 함정을 reindex 로 해결")
    by_weekday = df.groupby("weekday")["revenue"].mean().reindex(WEEKDAY_ORDER)
    print(by_weekday.round(0).to_string())
    weekend_avg = by_weekday[["토", "일"]].mean()
    weekday_avg = by_weekday[["월", "화", "수", "목", "금"]].mean()
    print(f"  주말 평균 {weekend_avg:,.0f} / 평일 평균 {weekday_avg:,.0f}"
          f" -> 주말 배율 {weekend_avg / weekday_avg:.2f}배")
    print("  -> 데이터에 심어진 주말 효과(약 1.2배)를 집계로 복원했습니다.")

    # ------------------------------------------------------------------
    print("\n[3] 카테고리별 총매출과 비중(%)")
    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    share = (by_cat / by_cat.sum() * 100).round(1)
    cat_table = pd.DataFrame({"총매출(백만원)": (by_cat / 1e6).round(1), "비중(%)": share})
    print(cat_table.to_string())

    # ------------------------------------------------------------------
    print("\n[4] 지점 x 카테고리 — 다중 키 groupby, 상위 5개 조합")
    combo = (
        df.groupby(["store", "category"])["revenue"].sum()
          .sort_values(ascending=False)
          .head(5)
    )
    print((combo / 1e6).round(1).to_string())
    print("  -> MultiIndex 결과입니다. 표로 쓰려면 reset_index() 를 붙이세요.")

    # ------------------------------------------------------------------
    print("\n[5] transform — '지점 내 매출 비중' 파생 열 만들기")
    store_total = df.groupby("store")["revenue"].transform("sum")  # 행 수 유지!
    df = df.assign(share_in_store=df["revenue"] / store_total)
    sample = df.loc[df["store"] == "강남점",
                    ["date", "category", "revenue", "share_in_store"]].head(3)
    print(sample.to_string(index=False))
    check = df.groupby("store")["share_in_store"].sum().round(6)
    print(f"  검증: 지점별 비중 합 = {[float(v) for v in check.unique()]} (전부 1.0 이어야 정상)")

    # ------------------------------------------------------------------
    print("\n[6] 최종 리포트 저장")
    csv_path = os.path.join(OUT_DIR, "store_report.csv")
    store_report.reset_index().to_csv(csv_path, index=False, encoding="utf-8-sig")
    top_store = store_report.index[0]
    top_cat = by_cat.index[0]
    print(f"  지점별 요약표 저장: {csv_path}")
    print("  ---- 한 줄 요약 리포트 ----")
    print(f"  최고 매출 지점: {top_store} ({store_report.loc[top_store, 'total_mil']:,}백만원)")
    print(f"  최고 매출 카테고리: {top_cat} (비중 {share[top_cat]}%)")
    print(f"  주말 효과: 평일 대비 {weekend_avg / weekday_avg:.2f}배")
    print("\n정리: '~별로 보면?' 이라는 질문은 전부 groupby 한 줄로 바뀝니다.")


if __name__ == "__main__":
    main()
