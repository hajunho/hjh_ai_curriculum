"""
Lecture 04 · Level 04 — 정렬·중복 제거·상위 N
ORDER BY(줄 세우기), LIMIT/OFFSET(상위 N·페이지), DISTINCT(중복 제거)를
연습합니다. '최고가 상품 톱5' 같은 랭킹 표를 만들고, ORDER BY 없이
LIMIT 만 쓰면 톱 N 이 아니라는 함정도 눈으로 확인합니다.
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


def run(cur, step, title, sql, note=""):
    """SQL 문장을 보여 주고 실행 결과를 표로 출력하는 공용 실행기."""
    print(f"[{step}] {title}")
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
    if note:
        print(f"  → {note}")
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"실습 DB 준비 완료: {db_path.name}")
    print("실행 순서: FROM → WHERE → SELECT → ORDER BY → LIMIT (줄 세운 뒤 끊는다)\n")

    run(cur, 1, "최고가 상품 톱5 — 랭킹 표의 기본형", """
        SELECT name, category, price
        FROM products
        ORDER BY price DESC
        LIMIT 5
    """, note="DESC = 내림차순(비싼 것부터). 톱 N 은 ORDER BY + LIMIT 세트")

    run(cur, 2, "마진 큰 상품 톱3 — 계산식을 정렬 기준으로", """
        SELECT name, price, cost, price - cost AS margin
        FROM products
        ORDER BY margin DESC
        LIMIT 3
    """, note="별칭(margin)을 ORDER BY 에서 그대로 쓸 수 있습니다")

    run(cur, 3, "최신 주문 5건 — 날짜 내림차순은 '최근 내역' 패턴", """
        SELECT order_id, customer_id, ordered_at, status
        FROM orders
        ORDER BY ordered_at DESC
        LIMIT 5
    """)

    run(cur, 4, "다중 정렬 — 도시 가나다순, 같은 도시는 등급순", """
        SELECT city, grade, name
        FROM customers
        ORDER BY city ASC, grade ASC
        LIMIT 8
    """, note="쉼표 순서가 우선순위: 1차 도시, 2차 등급")

    run(cur, 5, "DISTINCT — 고객들이 실제로 사는 도시 '종류'", """
        SELECT DISTINCT city
        FROM customers
        ORDER BY city
    """, note="고객 200행이 도시 종류만 남아 몇 행으로 줄었는지 보세요")

    run(cur, "5b", "DISTINCT 두 열 — (도시, 등급) '조합'의 종류", """
        SELECT DISTINCT city, grade
        FROM customers
        ORDER BY city, grade
        LIMIT 8
    """, note="DISTINCT 는 선택한 열 조합 전체에 적용됩니다")

    run(cur, 6, "페이지 나누기 — 가격 랭킹 6~10위 (2페이지)", """
        SELECT name, price
        FROM products
        ORDER BY price DESC
        LIMIT 5 OFFSET 5
    """, note="OFFSET 5 = 앞 5행을 건너뛰고 다음 5행")

    run(cur, 7, "함정 — ORDER BY 없이 LIMIT 5 만 쓰면?", """
        SELECT name, price
        FROM products
        LIMIT 5
    """, note="그냥 '아무 5행'입니다. [1]의 톱5와 비교해 보세요!")

    con.close()
    print("[정리] 랭킹 = ORDER BY (DESC) + LIMIT / 종류 = DISTINCT")
    print("  원본 테이블의 순서나 내용은 전혀 바뀌지 않습니다 (결과 화면만 손질).")


if __name__ == "__main__":
    main()
