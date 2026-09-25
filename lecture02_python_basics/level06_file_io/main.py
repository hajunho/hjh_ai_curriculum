"""파일 읽고 쓰기 — 거래내역 CSV 생성 -> 집계 -> 요약 리포트 저장.

with open / csv 모듈 / encoding="utf-8" 을 실습합니다.
[1] 거래내역 CSV 를 만들고 [2] DictReader 로 읽어 집계한 뒤
[3] 사람이 읽는 요약 리포트 텍스트를 저장합니다.
[4] 추가("a") 모드 로그와 인코딩 깨짐 재현까지 확인합니다.
산출물은 이 파일이 있는 폴더의 outputs/ 아래에 저장됩니다.
"""

import csv
import os
import random

# 이 파일이 있는 폴더 기준으로 outputs/ 경로를 만든다 (실행 위치와 무관하게 동작)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")


def create_transactions_csv(path):
    """[1] 거래내역 20건짜리 CSV 파일을 생성한다 (시스템이 준 파일 역할)."""
    random.seed(7)                     # 재현성: 항상 같은 데이터
    categories = ["사무용품", "식비", "교통비", "소프트웨어"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "category", "amount"])          # 헤더
        for day in range(1, 21):                                 # 9월 1~20일
            category = random.choice(categories)
            amount = random.randint(5, 300) * 1000               # 5천~30만 원
            writer.writerow([f"2026-09-{day:02d}", category, amount])


def summarize_transactions(path):
    """[2] CSV 를 읽어 (총액, 건수, 카테고리별 합계, 최대 거래) 를 반환한다."""
    total = 0
    count = 0
    by_category = {}                   # 카테고리 -> 합계 서랍장
    biggest = ("", "", 0)              # (날짜, 카테고리, 금액)

    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):              # 각 행이 딕셔너리로 온다
            amount = int(row["amount"])            # CSV 값은 전부 문자열 -> 변환 필수!
            total += amount
            count += 1
            by_category[row["category"]] = by_category.get(row["category"], 0) + amount
            if amount > biggest[2]:
                biggest = (row["date"], row["category"], amount)

    return total, count, by_category, biggest


def write_report(path, total, count, by_category, biggest):
    """[3] 집계 결과를 사람이 읽는 리포트 텍스트로 저장한다."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== 9월 거래내역 요약 리포트 ===\n")
        f.write(f"총 지출   : {total:,}원 ({count}건)\n")
        f.write(f"건당 평균 : {total // count:,}원\n")
        f.write("카테고리별 지출:\n")
        for category, amount in sorted(by_category.items(), key=lambda kv: -kv[1]):
            share = amount / total
            f.write(f"  - {category:6s}: {amount:>9,}원 ({share:.1%})\n")
        f.write(f"최대 거래 : {biggest[0]} {biggest[1]} {biggest[2]:,}원\n")


def main():
    print("=" * 56)
    print(" 파일 입출력 — 거래내역 CSV 와 요약 리포트")
    print("=" * 56)

    os.makedirs(OUT_DIR, exist_ok=True)            # outputs/ 가 없으면 생성

    # ---------------------------------------------------------
    # [1] 거래내역 CSV 생성 ("w" 쓰기 모드)
    # ---------------------------------------------------------
    print("\n[1] 거래내역 CSV 생성")
    csv_path = os.path.join(OUT_DIR, "transactions.csv")
    create_transactions_csv(csv_path)
    print(f"  저장 완료 -> {csv_path}")

    with open(csv_path, encoding="utf-8") as f:    # 앞 3줄만 미리보기
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  미리보기 {i}: {line.strip()}")   # strip 으로 줄 끝 \n 제거

    # ---------------------------------------------------------
    # [2] CSV 읽어서 집계 (DictReader)
    # ---------------------------------------------------------
    print("\n[2] CSV 읽기와 집계")
    total, count, by_category, biggest = summarize_transactions(csv_path)
    print(f"  총 지출 {total:,}원 / {count}건")
    for category, amount in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"  {category:6s}: {amount:>9,}원")
    print(f"  최대 거래: {biggest[0]} {biggest[1]} {biggest[2]:,}원")

    # ---------------------------------------------------------
    # [3] 요약 리포트 저장 후 다시 읽어 확인
    # ---------------------------------------------------------
    print("\n[3] 요약 리포트 저장")
    report_path = os.path.join(OUT_DIR, "daily_report.txt")
    write_report(report_path, total, count, by_category, biggest)
    print(f"  저장 완료 -> {report_path}")

    with open(report_path, encoding="utf-8") as f:
        for line in f:                              # 한 줄씩 읽기 (메모리 절약 패턴)
            print(f"  | {line.rstrip()}")

    # ---------------------------------------------------------
    # [4] "a" 추가 모드 로그 + 인코딩 깨짐 재현
    # ---------------------------------------------------------
    print("\n[4] 추가 모드와 인코딩")

    log_path = os.path.join(OUT_DIR, "run_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:      # "a": 기존 내용 뒤에 이어 쓰기
        f.write(f"리포트 생성 완료: 총 {total:,}원 / {count}건\n")
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  run_log.txt 누적 {len(lines)}줄 (재실행할수록 늘어남 — 'a' 모드)")

    # utf-8 로 쓴 한글을 latin-1 규격으로 읽으면? (다른 콘센트에 플러그 꽂기)
    sample_path = os.path.join(OUT_DIR, "encoding_sample.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("월간 매출 보고")
    with open(sample_path, encoding="utf-8") as f:
        ok_text = f.read()
    with open(sample_path, encoding="latin-1") as f:      # 일부러 잘못된 인코딩
        broken_text = f.read()
    print(f"  utf-8 로 읽음   : {ok_text!r}")
    print(f"  latin-1 로 읽음 : {broken_text!r}  <- 글자 깨짐!")
    print("  -> 읽기/쓰기 모두 encoding='utf-8' 명시가 한국어 데이터의 생존 수칙.")

    print("\n[끝] 프로그램이 끝나도 outputs/ 의 파일은 남습니다. 직접 열어 보세요.")


if __name__ == "__main__":
    main()
