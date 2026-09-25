"""
피벗·리셰이프·윈도우 연산 실습.
엑셀 피벗 테이블을 pandas 로 재현합니다.
  - pivot_table (index/columns/values/aggfunc/margins)
  - melt 로 wide -> long 되돌리기
  - rank / pct_change / cumsum 윈도우 연산과 그룹 내 순위
최종 피벗 리포트는 outputs/ 에 CSV 로 저장합니다.
"""

import os
import pathlib
import sys

import pandas as pd

# 공용 데이터 모듈(hjh_data)을 불러오기 위한 경로 설정
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
WEEKDAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]


def load_clean_sales() -> pd.DataFrame:
    """매출 데이터 생성 후 결측·음수를 제거하고 진짜 날짜를 붙입니다."""
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    df = df.dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    # date 문자열은 오염되어 있으므로(level09 참고) day_index 로 날짜를 만듭니다.
    df["real_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["day_index"], unit="D")
    print(f"    정제 후 {len(df):,}행 (long 포맷: 한 행 = 하루·지점·카테고리 매출 1건)")
    return df


def step1_store_category_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """[1] 지점 x 카테고리 연간 총매출 피벗 (margins=총합계 포함)."""
    print("\n[1] 지점 x 카테고리 총매출 피벗 — 엑셀 피벗 테이블의 pandas 판")
    pivot = pd.pivot_table(df, index="store", columns="category",
                           values="revenue", aggfunc="sum",
                           margins=True, margins_name="합계")
    print((pivot / 1e8).round(2).to_string())
    print("    (단위: 억원, margins=True 로 '합계' 행·열 자동 추가)")
    return pivot


def step2_weekday_pivot(df: pd.DataFrame) -> None:
    """[2] 요일 x 지점 평균 매출 피벗 — aggfunc 만 바꾸면 다른 질문에 답합니다."""
    print("\n[2] 요일 x 지점 '평균' 매출 피벗 (aggfunc='mean')")
    pivot = pd.pivot_table(df, index="weekday", columns="store",
                           values="revenue", aggfunc="mean")
    pivot = pivot.reindex(WEEKDAY_ORDER)  # 가나다순 -> 월~일 업무 순서로 복원
    print((pivot / 1e4).round(0).astype(int).to_string())
    print("    (단위: 만원 — 토·일 행이 평일보다 높은 '주말 효과'를 확인하세요)")


def step3_melt(df: pd.DataFrame) -> None:
    """[3] melt: wide 피벗을 long 포맷으로 되돌립니다."""
    print("\n[3] melt — 서랍장(wide)을 옷걸이(long)로 되돌리기")
    wide = pd.pivot_table(df, index="store", columns="category",
                          values="revenue", aggfunc="sum")  # margins 없는 버전
    long = wide.reset_index().melt(id_vars="store",
                                   var_name="category", value_name="revenue")
    print(f"    wide: {wide.shape[0]}행 x {wide.shape[1]}열 (사람이 읽기 좋음)")
    print(f"    long: {long.shape[0]}행 x {long.shape[1]}열 (기계가 다루기 좋음)")
    print("    long 앞 3행:")
    print(long.head(3).to_string(index=False))
    print("    -> 분석·그래프·병합은 long, 보고서 표는 wide 가 정석입니다.")


def step4_rankings(df: pd.DataFrame) -> None:
    """[4] rank: 전사 순위와 카테고리 그룹 내 순위."""
    print("\n[4] 순위 분석 — rank 와 groupby+rank")
    store_total = df.groupby("store")["revenue"].sum()
    ranking = store_total.rank(ascending=False).astype(int).sort_values()
    print("    지점별 총매출 전사 순위:")
    for store, r in ranking.items():
        print(f"      {r}위 {store} ({store_total[store] / 1e8:.1f}억)")

    # 카테고리 그룹 내 지점 순위: "커피 부문에서 강남점은 몇 등인가?"
    cat_store = df.groupby(["category", "store"], as_index=False)["revenue"].sum()
    cat_store["rank_in_category"] = (cat_store.groupby("category")["revenue"]
                                     .rank(ascending=False).astype(int))
    coffee = cat_store[cat_store["category"] == "커피"].sort_values("rank_in_category")
    print("    '커피' 카테고리 그룹 내 순위:")
    for _, row in coffee.iterrows():
        print(f"      {row['rank_in_category']}위 {row['store']} ({row['revenue'] / 1e8:.1f}억)")


def step5_monthly_flow(df: pd.DataFrame) -> None:
    """[5] 월별 매출에 pct_change(전월 대비)와 cumsum(누적)을 붙입니다."""
    print("\n[5] 월별 흐름 — pct_change 와 cumsum")
    monthly = (df.groupby(df["real_date"].dt.to_period("M"))["revenue"].sum())
    report = pd.DataFrame({
        "revenue_100m": (monthly / 1e8).round(2),                 # 월 매출(억)
        "mom_pct": (monthly.pct_change() * 100).round(1),         # 전월 대비 %
        "cum_100m": (monthly.cumsum() / 1e8).round(1),            # 누적 매출(억)
    })
    print(report.to_string())
    print("    (mom_pct 첫 행 NaN 은 '비교할 전월이 없음'이라는 정직한 표시입니다)")


def step6_save_report(pivot: pd.DataFrame) -> None:
    """[6] 완성된 피벗 리포트를 CSV 로 저장합니다."""
    print("\n[6] 리포트 저장")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = OUT_DIR / "pivot_report.csv"
    (pivot / 1e8).round(3).to_csv(out_path, encoding="utf-8-sig")  # 엑셀 호환 인코딩
    print(f"    저장 완료: {out_path}")
    print("    엑셀에서 열면 손으로 만들던 피벗 표가 코드 한 번에 재현된 것을 볼 수 있습니다.")


def main() -> None:
    print("=" * 60)
    print("Level 10 — 피벗·리셰이프·윈도우 연산")
    print("=" * 60)
    df = load_clean_sales()
    pivot = step1_store_category_pivot(df)
    step2_weekday_pivot(df)
    step3_melt(df)
    step4_rankings(df)
    step5_monthly_flow(df)
    step6_save_report(pivot)
    print("\n완료! 같은 pivot_table 로 aggfunc 만 바꿔 가며 다른 질문에 답해 보세요.")


if __name__ == "__main__":
    main()
