"""
Lecture 04 · Level 00 — 데이터베이스가 왜 필요한가
'엑셀 파일 돌려쓰기'의 세 가지 사고(동시 수정 손실, 정합성 오염, 권한 부재)를
파이썬 시뮬레이션으로 재현하고, 같은 시나리오를 데이터베이스(sqlite3) 방식으로
다시 수행해 무엇이 달라지는지 대비합니다. SQL 문법은 다음 레벨부터 배우므로
여기서는 출력 결과의 '차이'에만 집중하면 됩니다.
"""

import copy
import pathlib
import sqlite3
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data  # 공용 데이터 생성기 (실습용 쇼핑몰 DB를 만들어 줍니다)


def show_rows(title, rows):
    """딕셔너리 목록을 간단한 표로 출력합니다."""
    print(f"  {title}")
    for r in rows:
        print("   ", " | ".join(f"{k}={v}" for k, v in r.items()))


def excel_style_disaster():
    """[1] 동시 수정: 두 사람이 같은 파일 사본을 고친 뒤 차례로 저장하면?"""
    print("[1] 엑셀 방식 — 동시 수정 사고 (갱신 손실)")
    shared_file = [  # 공유 폴더의 '고객명단.xlsx' 라고 상상하세요
        {"고객번호": 1, "이름": "김민준", "등급": "SILVER"},
        {"고객번호": 2, "이름": "이서연", "등급": "BASIC"},
    ]
    # 두 사람이 각자 파일을 '다운로드' (복사본 생성)
    kim_copy = copy.deepcopy(shared_file)
    park_copy = copy.deepcopy(shared_file)

    kim_copy[0]["등급"] = "VIP"      # 김대리: 1번 고객을 VIP 로 승급
    park_copy[1]["등급"] = "GOLD"    # 박과장: 2번 고객을 GOLD 로 승급

    shared_file = kim_copy    # 김대리가 먼저 저장 (파일 통째로 덮어쓰기)
    shared_file = park_copy   # 박과장이 나중에 저장 → 김대리의 작업이 사라짐!

    show_rows("최종 저장된 파일:", shared_file)
    print("  → 1번 고객이 여전히 SILVER. 김대리의 승급 작업이 '에러 없이' 증발했습니다.\n")


def integrity_disaster():
    """[2] 정합성: 같은 거래처가 표기만 다르게 두 번 입력된 장부."""
    print("[2] 엑셀 방식 — 정합성 오염 (같은 거래처, 다른 표기)")
    dirty_ledger = [
        {"거래처": "(주)한빛", "금액": 300},
        {"거래처": "한빛 주식회사", "금액": 200},   # 사실 위와 같은 회사
        {"거래처": "미래상사", "금액": 150},
    ]
    totals = {}
    for row in dirty_ledger:
        totals[row["거래처"]] = totals.get(row["거래처"], 0) + row["금액"]
    for name, amount in totals.items():
        print(f"    {name}: {amount}만원")
    print("  → 한빛은 실제 500만원 거래처인데 300/200 으로 쪼개져 보고서가 틀립니다.\n")


def permission_disaster():
    """[3] 권한: 파일을 보내는 순간 민감한 열까지 통째로 전달됩니다."""
    print("[3] 엑셀 방식 — 권한 부재 (파일 전달 = 전체 유출)")
    hr_file = [
        {"이름": "한지민", "부서": "영업", "연봉": 9000},
        {"이름": "김도윤", "부서": "개발", "연봉": 7200},
    ]
    print("  '부서 현황만 참고하세요' 라며 파일을 첨부하면, 받는 사람 화면에는:")
    show_rows("전달된 파일 내용:", hr_file)
    print("  → 연봉 열을 뺄 방법이 '파일을 새로 만드는 것'뿐. 원본 통제가 불가능합니다.\n")


def database_way():
    """[4] DB 방식: 한 곳의 원본 + 요청 창구. 같은 시나리오를 다시 수행합니다."""
    print("[4] 데이터베이스 방식 — 같은 시나리오, 다른 결과")
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))   # 실습용 쇼핑몰 DB 생성 (설치 불필요)
    print(f"  실습 DB 생성: {db_path.name} (customers 등 5개 테이블)")

    # 두 개의 연결 = 두 명의 사용자. 원본은 하나, 각자 '요청'만 보냅니다.
    kim = sqlite3.connect(db_path)
    park = sqlite3.connect(db_path)

    kim.execute("UPDATE customers SET grade='VIP' WHERE customer_id=1")
    kim.commit()      # 김대리: 1번 고객 승급 요청 → 창구가 원본에 기록
    park.execute("UPDATE customers SET grade='GOLD' WHERE customer_id=2")
    park.commit()     # 박과장: 2번 고객 승급 요청 → 역시 원본에 기록

    cur = kim.execute(
        "SELECT customer_id, name, grade FROM customers WHERE customer_id IN (1, 2)")
    rows = [{"고객번호": r[0], "이름": r[1], "등급": r[2]} for r in cur.fetchall()]
    show_rows("원본 테이블의 현재 상태:", rows)
    print("  → 두 사람의 수정이 모두 남았습니다. 원본이 하나이므로 덮어쓰기 사고가 없습니다.")

    # 정합성: 규칙(제약조건)을 어기는 데이터는 저장 자체가 거부됩니다.
    try:
        kim.execute("INSERT INTO customers (customer_id, name) VALUES (1, '유령고객')")
    except sqlite3.IntegrityError as e:
        print(f"  중복 고객번호 저장 시도 → DB가 거부: {e}")

    # 권한: 민감한 열을 뺀 '조회 전용 창(뷰)'만 열어 줄 수 있습니다.
    kim.execute("CREATE VIEW IF NOT EXISTS emp_public AS "
                "SELECT name, dept FROM employees")   # salary 열은 아예 제외
    cur = kim.execute("SELECT * FROM emp_public LIMIT 2")
    rows = [{"이름": r[0], "부서": r[1]} for r in cur.fetchall()]
    show_rows("공유용 창구(뷰)로 보이는 내용:", rows)
    print("  → 연봉 열은 창구에 존재하지 않으므로 유출 자체가 불가능합니다.\n")

    kim.close()
    park.close()


def main():
    print("=" * 62)
    print(" 엑셀 돌려쓰기의 세 가지 지옥 vs 데이터베이스")
    print("=" * 62 + "\n")
    excel_style_disaster()
    integrity_disaster()
    permission_disaster()
    database_way()
    print("[5] 정리")
    print("  엑셀 방식: 사본이 여러 개 → 사고가 나도 '에러조차' 없다.")
    print("  DB 방식  : 원본 하나 + 요청 창구 → 순서 보장, 규칙 검사, 권한 관리.")
    print("  다음 레벨부터 이 창구에 말을 거는 언어, SQL 을 배웁니다.")


if __name__ == "__main__":
    main()
