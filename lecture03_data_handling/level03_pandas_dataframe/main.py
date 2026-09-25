"""
Pandas Series 와 DataFrame 의 구조를 해부하는 실습입니다.
카페 체인 90일치 매출(hjh_data.sales_table)을 DataFrame 으로 만들어
shape / index / columns / dtypes / head / info / describe 를 차례로 확인하고,
열 하나를 Series 로 꺼내 보고, 파생 열을 만들고, 결측 개수까지 진단합니다.
"""

import io
import pathlib
import sys

import pandas as pd

# 공용 데이터 모듈(common/hjh_data.py)을 불러올 수 있게 경로를 추가합니다.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 110)
    pd.set_option("display.max_columns", 10)

    # ------------------------------------------------------------------
    print("[1] 딕셔너리 리스트 -> DataFrame")
    rows = hjh_data.sales_table(n_days=90, seed=42)   # seed 고정: 항상 같은 데이터
    print(f"    원본: 파이썬 딕셔너리 리스트, {len(rows)}개 (90일 x 5지점 x 5카테고리)")
    df = pd.DataFrame(rows)
    print(f"    변환: pd.DataFrame(rows) 한 줄 -> 타입 {type(df).__name__}")

    # ------------------------------------------------------------------
    print("\n[2] 표의 3요소: 값 / 행 이름표(index) / 열 이름(columns)")
    print(f"    shape   : {df.shape}  (행 {df.shape[0]}개, 열 {df.shape[1]}개)")
    print(f"    index   : {df.index}")
    print(f"    columns : {list(df.columns)}")

    # ------------------------------------------------------------------
    print("\n[3] 열마다 하나씩 갖는 자료형(dtype)")
    print(df.dtypes.to_string())
    print("    -> revenue 가 float64 인 이유: 결측치(NaN)가 섞이면 정수 열도 실수가 됩니다.")

    # ------------------------------------------------------------------
    print("\n[4] head() — 앞 5행 미리보기 (엑셀에서 스크롤 맨 위를 보는 것과 같습니다)")
    print(df.head().to_string())

    # ------------------------------------------------------------------
    print("\n[5] info() — 데이터 건강 검진 (열별 결측 아닌 값 개수 + dtype + 메모리)")
    buf = io.StringIO()                 # info() 는 반환값이 없어 출력만 하므로 버퍼로 받습니다.
    df.info(buf=buf)
    print(buf.getvalue())

    # ------------------------------------------------------------------
    print("[6] describe() — 숫자 열 요약 통계")
    print(df[["ad_cost", "revenue"]].describe().round(1).to_string())
    print("    -> revenue 의 min 이 음수! 이상치가 섞여 있다는 신호입니다 (level06 에서 처리).")

    # ------------------------------------------------------------------
    print("\n[7] 열 하나를 꺼내면 Series — 이름표(index) 달린 1차원 값 묶음")
    revenue = df["revenue"]
    print(f"    type(df['revenue']) = {type(revenue).__name__}")
    print(f"    길이 {len(revenue)}, dtype {revenue.dtype}")
    print(f"    평균 {revenue.mean():,.0f}원 / 최대 {revenue.max():,.0f}원 / 최소 {revenue.min():,.0f}원")
    print("    앞 3개 값 (왼쪽 숫자가 index, 오른쪽이 값):")
    print(revenue.head(3).to_string())

    # ------------------------------------------------------------------
    print("\n[8] 새 열 만들기 — 광고비 대비 매출 비율(roas)")
    df["roas"] = df["revenue"] / df["ad_cost"]        # 엑셀의 '수식 넣고 드래그'가 한 줄
    print(df[["store", "category", "ad_cost", "revenue", "roas"]].head(3).round(2).to_string())
    print(f"    roas 평균: {df['roas'].mean():.2f} (광고비 1원당 매출 원)")

    # ------------------------------------------------------------------
    print("\n[9] value_counts() — 범주 열 구성비 (엑셀 COUNTIF 반복을 한 줄로)")
    print("    지점별 행 수:")
    print(df["store"].value_counts().to_string())
    print("    요일별 행 수 (90일이 7로 나누어떨어지지 않아 요일마다 일수가 다릅니다):")
    print(df["weekday"].value_counts().to_string())

    # ------------------------------------------------------------------
    print("\n[10] 결측치 개수 진단 — isna().sum()")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())
    ratio = df["revenue"].isna().mean() * 100
    print(f"    -> revenue 결측 비율 {ratio:.2f}%. 처리 전략은 level06 에서 배웁니다.")

    print("\n정리: DataFrame = index + columns + 값. 열 하나를 꺼내면 Series.")
    print("새 데이터를 받으면 shape -> head -> info -> describe 순서로 인사하세요.")


if __name__ == "__main__":
    main()
