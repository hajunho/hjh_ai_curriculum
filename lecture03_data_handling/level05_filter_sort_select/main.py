"""
필터링·정렬·선택으로 업무 질문에 답하는 실습입니다.
카페 체인 180일치 매출에서 불리언 마스크, 조건 결합(&, |, ~, isin),
loc/iloc, sort_values/nlargest 를 사용해
"강남점 주말 커피 매출 상위 10" 같은 실제 질문 5개를 코드로 번역합니다.
"""

import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 120)

    # ------------------------------------------------------------------
    print("[0] 데이터 준비 — 180일치 매출, 결측 행은 이번 레벨에서는 간단히 제거")
    df = pd.DataFrame(hjh_data.sales_table(n_days=180, seed=42))  # seed 고정
    n_missing = df["revenue"].isna().sum()
    df = df.dropna(subset=["revenue"]).copy()   # 정식 처리 전략은 level06 에서
    print(f"    결측 revenue {n_missing}건 제거 -> {len(df)}행으로 시작합니다.")

    # ------------------------------------------------------------------
    print("\n[질문1] 강남점의 주말 커피 매출 상위 10건을 뽑아 주세요.")
    q1 = df[
        (df["store"] == "강남점")
        & (df["weekday"].isin(["토", "일"]))     # 값 여러 개 중 하나 -> isin
        & (df["category"] == "커피")
    ]
    top10 = q1.nlargest(10, "revenue")           # '상위 N' 은 nlargest 가 깔끔
    print(f"    조건에 맞는 행 {len(q1)}건 중 상위 10건:")
    print(top10[["date", "weekday", "revenue"]].to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[질문2] 매출 50만원 이상이면서 광고비는 20만원 이하였던 '효율 좋은' 행은?")
    q2 = df[(df["revenue"] >= 500_000) & (df["ad_cost"] <= 200_000)]
    print(f"    해당 행 {len(q2)}건 (전체의 {len(q2) / len(df) * 100:.1f}%)")
    print("    매출 상위 3건:")
    print(q2.nlargest(3, "revenue")[["date", "store", "category", "ad_cost", "revenue"]]
          .to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[질문3] 부산점·대전점의 디저트·베이커리 실적만 골라 주세요. (isin 두 번)")
    q3 = df[df["store"].isin(["부산점", "대전점"])
            & df["category"].isin(["디저트", "베이커리"])]
    print(f"    해당 행 {len(q3)}건. 지점 x 카테고리 조합별 평균 매출:")
    summary = q3.groupby(["store", "category"])["revenue"].mean().round(0)
    print(summary.to_string())
    print("    (groupby 는 level07 에서 정식으로 배웁니다. 여기서는 맛보기!)")

    # ------------------------------------------------------------------
    print("\n[질문4] loc(이름표)과 iloc(위치)은 어떻게 다른가요?")
    busan = df[df["store"] == "부산점"]
    first_label = busan.index[0]
    print(f"    부산점만 필터한 표의 index 앞부분: {list(busan.index[:5])}")
    print("    -> 0,1,2... 가 아니라 원본의 행 번호를 그대로 들고 있습니다!")
    print(f"    busan.iloc[0]  : 위치 기준 '첫 행'  -> date={busan.iloc[0]['date']}, "
          f"revenue={busan.iloc[0]['revenue']:,.0f}")
    print(f"    busan.loc[{first_label}]: 이름표 {first_label}번 행 -> 같은 행입니다.")
    print("    busan.loc[0] 은? 이름표 0번이 부산점 표에 없으면 KeyError 가 납니다.")
    # 행 조건 + 열 선택 동시에: 실무 최다 빈출 문형
    picked = df.loc[df["revenue"] >= 900_000, ["date", "store", "category", "revenue"]]
    print(f"    df.loc[조건, 열목록] 예: 매출 90만 이상 {len(picked)}건 중 앞 3건")
    print(picked.head(3).to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[질문5] 광고비 대비 매출 효율(roas)이 가장 좋았던 행은 어디였나요?")
    df["roas"] = df["revenue"] / df["ad_cost"]      # 파생 열
    q5 = df.sort_values("roas", ascending=False)    # 내림차순 정렬
    print("    roas 상위 5건:")
    print(q5.head(5)[["date", "store", "category", "ad_cost", "revenue", "roas"]]
          .round(2).to_string(index=False))
    print("    다중 기준 정렬 예: 지점 오름차순 + 매출 내림차순 상위 1건씩 확인")
    multi = df.sort_values(["store", "revenue"], ascending=[True, False])
    print(multi.groupby("store").head(1)[["store", "date", "category", "revenue"]]
          .to_string(index=False))

    print("\n정리: 업무 질문 = (조건 결합으로 골라내기) + (정렬/상위N) + (loc 으로 열 선택).")
    print("      조건마다 괄호, and/or 대신 &/|, 첫 행은 iloc[0] — 이 세 가지만 지켜도 오류 대부분을 예방합니다.")


if __name__ == "__main__":
    main()
