"""
엑셀에서 파이썬으로 — 매출표를 CSV 로 저장/재로딩한 뒤,
엑셀에서 하던 SUM·SUMIF·자동 필터·정렬을 표준 라이브러리 코드로 재현합니다.
결측·음수 오염을 건너뛴 건수를 스스로 보고하게 하여
'검증 가능한 절차'로서의 코드와 재현성의 가치를 체험합니다.
"""

import csv
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

BASE_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"


def load_clean_rows(csv_path: str) -> tuple[list[dict], int, int]:
    """CSV 를 읽고 revenue 를 숫자로 복원합니다.
    결측(빈 문자열)·음수 행은 건너뛰고 그 개수를 함께 돌려줍니다."""
    clean, n_missing, n_negative = [], 0, 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw = r["revenue"]
            if raw == "" or raw == "None":          # 결측 → 건너뜀
                n_missing += 1
                continue
            revenue = int(raw)
            if revenue < 0:                          # 음수 오염 → 건너뜀
                n_negative += 1
                continue
            r["revenue"] = revenue                   # 타입 복원 (str → int)
            r["ad_cost"] = int(r["ad_cost"])
            clean.append(r)
    return clean, n_missing, n_negative


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] 매출표 60일치를 CSV 로 저장 (엑셀에서도 열리는 형식)")
    rows = hjh_data.sales_table(n_days=60, seed=42)
    csv_path = hjh_data.to_csv(rows, str(OUT_DIR / "sales.csv"))
    print(f"    저장 완료: {csv_path} ({len(rows)}행)")
    print()

    print("[2] CSV 다시 읽기 — 모든 값이 '문자열'로 들어옵니다")
    with open(csv_path, newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    print(f"    첫 행: {first}")
    print(f"    revenue 의 타입: {type(first['revenue']).__name__} ← 숫자가 아닙니다!")
    print()

    print("[3] 타입 복원 + 오염 행 건너뛰기 (버린 것은 반드시 보고)")
    clean, n_missing, n_negative = load_clean_rows(csv_path)
    print(f"    사용 {len(clean)}행 / 결측 건너뜀 {n_missing}건 / 음수 건너뜀 {n_negative}건")
    print()

    print("[4] SUM — 엑셀의 =SUM(G:G) 에 해당")
    total = sum(r["revenue"] for r in clean)
    print(f"    60일 전체 매출 합계: {total:,}원")
    print()

    print("[5] SUMIF — 지점별 합계 (딕셔너리 누적)")
    by_store: dict[str, int] = {}
    for r in clean:
        by_store[r["store"]] = by_store.get(r["store"], 0) + r["revenue"]
    for store, subtotal in sorted(by_store.items(), key=lambda kv: kv[1]):
        print(f"    {store}: {subtotal:>15,}원")
    print()

    print("[6] 자동 필터 — '강남점 & 주말' 행만 골라 평일과 비교")
    gangnam = [r for r in clean if r["store"] == "강남점"]
    weekend = [r for r in gangnam if r["weekday"] in ("토", "일")]
    weekday_rows = [r for r in gangnam if r["weekday"] not in ("토", "일")]
    avg_weekend = sum(r["revenue"] for r in weekend) / len(weekend)
    avg_weekday = sum(r["revenue"] for r in weekday_rows) / len(weekday_rows)
    print(f"    강남점 주말 평균: {avg_weekend:>12,.0f}원 ({len(weekend)}행)")
    print(f"    강남점 평일 평균: {avg_weekday:>12,.0f}원 ({len(weekday_rows)}행)")
    print(f"    → 주말이 평일의 {avg_weekend / avg_weekday:.2f}배입니다.")
    print()

    print("[7] 정렬 — 매출 상위 5 (엑셀의 내림차순 정렬)")
    top5 = sorted(clean, key=lambda r: r["revenue"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        print(f"    {i}위 | {r['date']} {r['weekday']} | {r['store']} "
              f"{r['category']} | {r['revenue']:,}원")
    print()

    print("[8] 재현성 — 이 스크립트의 진짜 가치")
    print("    지금 본 모든 숫자는 '고정 seed + 기록된 절차'에서 나왔습니다.")
    print("    내일 다시 실행해도, 다른 사람이 실행해도 완전히 같은 결과가 나옵니다.")
    print("    엑셀 클릭은 기억에 남지만, 코드는 문서로 남습니다.")


if __name__ == "__main__":
    main()
