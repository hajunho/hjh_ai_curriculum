"""
Lecture 04 · Level 03 — WHERE: 조건 검색
업무에서 나올 법한 질문 6개를 한국어 → SQL 로 번역해 실행합니다.
비교 연산(=, >=), AND/OR 와 괄호, IN, LIKE, BETWEEN, IS NULL 을 모두
다루며, '= NULL' 이라고 잘못 쓰면 왜 0건이 나오는지 함정도 시연합니다.
"""

import pathlib
import sqlite3
import sys
import textwrap
import unicodedata

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def disp_width(text):
    """한글은 2칸 폭이므로 표 정렬용 표시 폭을 계산합니다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width):
    return str(text) + " " * (width - disp_width(text))


def run(cur, step, question, sql, note=""):
    """업무 질문 → SQL 문장 → 실행 결과 표 순서로 출력하는 공용 실행기."""
    print(f"[{step}] 업무 질문: {question}")
    for line in textwrap.dedent(sql).strip().splitlines():
        print(f"  SQL> {line}")
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    widths = [max(disp_width(c), *(disp_width(r[i]) for r in rows)) if rows else disp_width(c)
              for i, c in enumerate(cols)]
    print("  " + " | ".join(pad(c, w) for c, w in zip(cols, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))
    print(f"  → {len(rows)}건" + (f" | {note}" if note else ""))
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"실습 DB 준비 완료: {db_path.name}")
    print("WHERE 는 행마다 참/거짓을 판정해 '참인 행만' 통과시키는 필터입니다.\n")

    run(cur, "Q1", "서울에 사는 VIP 고객 명단 주세요 (= 와 AND)", """
        SELECT name, city, grade
        FROM customers
        WHERE city = '서울' AND grade = 'VIP'
        LIMIT 6
    """, note="두 조건을 '모두' 만족하는 행만 통과")

    run(cur, "Q2", "10만 원 이상이거나 마진율 50%를 넘는 상품은? (OR 와 괄호)", """
        SELECT name, price, ROUND(100.0 * (price - cost) / price, 1) AS margin_pct
        FROM products
        WHERE (price >= 100000) OR (100.0 * (price - cost) / price > 50)
    """, note="OR 가 섞이면 괄호로 의도를 못박는 습관!")

    run(cur, "Q3", "서울·인천 또는 부산 고객 중 VIP/GOLD 만 (IN 조합)", """
        SELECT name, city, grade
        FROM customers
        WHERE city IN ('서울', '인천', '부산')
          AND grade IN ('VIP', 'GOLD')
        LIMIT 6
    """, note="IN 은 OR 나열의 깔끔한 줄임말")

    run(cur, "Q4", "김씨 성을 가진 고객을 찾아 주세요 (LIKE 패턴)", """
        SELECT name, city
        FROM customers
        WHERE name LIKE '김%'
        LIMIT 6
    """, note="% 는 '아무 글자 0개 이상' 와일드카드")

    run(cur, "Q5", "3분기(7~9월)의 취소 주문 내역은? (BETWEEN + AND)", """
        SELECT order_id, customer_id, ordered_at, status
        FROM orders
        WHERE ordered_at BETWEEN '2025-07-01' AND '2025-09-30'
          AND status = '취소'
        LIMIT 6
    """, note="BETWEEN 은 양 끝값을 포함")

    run(cur, "Q6", "상사가 없는 직원(조직도 최상단)은 누구? (IS NULL)", """
        SELECT employee_id, name, dept
        FROM employees
        WHERE manager_id IS NULL
    """, note="NULL 은 '기록되지 않음' — 전용 문법 IS NULL 사용")

    # 함정 시연: NULL 을 = 로 비교하면 항상 거짓 → 0건
    run(cur, "Q6-함정", "같은 질문을 '= NULL' 로 잘못 쓰면?", """
        SELECT employee_id, name, dept
        FROM employees
        WHERE manager_id = NULL
    """, note="0건! '모름 = 모름'의 답도 모름이라 참이 될 수 없습니다")

    con.close()
    print("[정리] 한국어 질문의 조건 표현이 WHERE 절과 1:1 로 대응됩니다.")
    print("  이상/이하 → >= <= | ~중 하나 → IN | ~로 시작 → LIKE '..%'")
    print("  기간 → BETWEEN | 값이 없음 → IS NULL")


if __name__ == "__main__":
    main()
