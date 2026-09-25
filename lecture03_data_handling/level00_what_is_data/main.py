"""
데이터란 무엇인가 — 표의 구조를 딕셔너리 리스트로 직접 해부합니다.
행(row)=사례, 열(column)=속성이라는 원리를 눈으로 확인하고,
열 추출·행 찾기·타입/결측 관찰·미니 스키마 요약을 표준 라이브러리만으로 수행합니다.
마지막에 비정형 텍스트와 비교해 '표'가 왜 집계에 유리한지 봅니다.
"""

import pathlib
import sys

# 공용 데이터 모듈(hjh_data)을 불러오기 위한 경로 설정
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def extract_column(rows: list[dict], col: str) -> list:
    """표에서 열 하나를 리스트로 뽑아냅니다. (열 = 같은 속성의 값 모음)"""
    return [r[col] for r in rows]


def summarize_schema(rows: list[dict]) -> list[dict]:
    """열마다 타입 구성과 결측 개수를 세어 '미니 스키마 요약'을 만듭니다."""
    summary = []
    for col in rows[0].keys():
        values = extract_column(rows, col)
        none_count = sum(1 for v in values if v is None)
        # None 을 제외한 값들의 타입 이름을 모읍니다
        type_names = sorted({type(v).__name__ for v in values if v is not None})
        summary.append({
            "column": col,
            "types": "/".join(type_names),
            "missing": none_count,
            "example": next(v for v in values if v is not None),
        })
    return summary


def main() -> None:
    # 30일치 가상 카페 매출표 생성 (seed 고정 → 항상 같은 결과)
    rows = hjh_data.sales_table(n_days=30, seed=42)

    print("[1] 표의 크기와 스키마(열 이름)")
    print(f"    행(사례) 수: {len(rows)}")
    print(f"    열(속성) 목록: {list(rows[0].keys())}")
    print()

    print("[2] 표 미리보기 — 받은 표는 무조건 먼저 눈으로 봅니다")
    hjh_data.head(rows, n=5)
    print()

    print("[3] 행 하나 해부 — 이 표의 행 하나는 '무엇 한 건'인가?")
    first = rows[0]
    for key, value in first.items():
        print(f"    {key:>10} = {value!r}  ({type(value).__name__})")
    print("    → 행 하나 = 특정 날짜·지점·카테고리의 '하루 매출' 1건입니다.")
    print()

    print("[4] 열 추출 — revenue 열만 뽑아 보기")
    revenues = extract_column(rows, "revenue")
    print(f"    revenue 열 길이: {len(revenues)} (행 수와 같습니다)")
    print(f"    앞 8개 값: {revenues[:8]}")
    print()

    print("[5] 조건으로 행 찾기 — 강남점의 커피 매출만")
    gangnam_coffee = [r for r in rows
                     if r["store"] == "강남점" and r["category"] == "커피"]
    print(f"    조건에 맞는 행: {len(gangnam_coffee)}개 (30일이므로 30개가 정상)")
    hjh_data.head(gangnam_coffee, n=3)
    print()

    print("[6] 미니 스키마 요약 — 열별 타입과 결측(None) 개수")
    schema = summarize_schema(rows)
    for s in schema:
        print(f"    {s['column']:>10} | 타입: {s['types']:<8} | "
              f"결측: {s['missing']:>2}개 | 예시: {s['example']!r}")
    missing_total = sum(s["missing"] for s in schema)
    negative_count = sum(1 for v in revenues if v is not None and v < 0)
    print(f"    → 전체 결측 {missing_total}칸, 음수 매출 {negative_count}건이 숨어 있습니다.")
    print("      (일부러 심어 둔 오염입니다. level06 에서 처리법을 배웁니다)")
    print()

    print("[7] 정형 vs 비정형 — 같은 정보, 다른 모양")
    unstructured = "어제 강남점 갔는데 커피 맛있었어요. 사람이 많아서 매출 꽤 나왔을 듯!"
    structured = {"date": "2025-01-01", "store": "강남점",
                  "category": "커피", "revenue": 512000}
    print(f"    비정형(자유 문장): {unstructured!r}")
    print(f"    정형(표의 행 1개): {structured!r}")
    print("    → 문장에는 SUM 을 걸 수 없지만, 표의 revenue 열은 바로 합칠 수 있습니다.")
    print("      비정형 데이터도 분석하려면 결국 표로 바꾸는 과정을 거칩니다.")


if __name__ == "__main__":
    main()
