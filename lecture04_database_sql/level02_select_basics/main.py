"""
Lecture 04 · Level 02 — SELECT 기초
SQL 의 출발점인 SELECT/FROM 을 연습합니다. 전체 열(*), 열 골라 보기,
별칭(AS), 계산된 열, 문자열 조합까지 — 각 단계마다 SQL 문장을 먼저
출력하고 바로 아래에 실행 결과를 표로 보여 줍니다. SQL 이 주인공이고
파이썬은 실행기입니다.
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


def run(cur, step, title, sql):
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
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"실습 DB 준비 완료: {db_path.name}")
    print("읽는 요령: FROM(어느 테이블에서)을 먼저 찾고 SELECT(어떤 열을)를 읽습니다.\n")

    run(cur, 1, "전체 열 훑어보기 — 테이블 첫인사는 * 로", """
        SELECT *
        FROM customers
        LIMIT 5
    """)

    run(cur, 2, "필요한 열만 골라 보기 — 요청서에 항목 명시", """
        SELECT name, city
        FROM customers
        LIMIT 5
    """)

    run(cur, 3, "별칭(AS) — 결과 표의 열 제목을 보고서용으로", """
        SELECT name  AS 상품명,
               category AS 카테고리,
               price AS 판매가
        FROM products
        LIMIT 5
    """)

    run(cur, 4, "계산된 열 — 행마다 마진과 마진율을 그 자리에서 계산", """
        SELECT name AS 상품명,
               price AS 판매가,
               cost  AS 원가,
               price - cost AS 마진,
               ROUND(100.0 * (price - cost) / price, 1) AS 마진율
        FROM products
        LIMIT 5
    """)

    run(cur, 5, "문자열 조합(||) — '이름 (등급)' 표시용 열 만들기", """
        SELECT name || ' (' || grade || ')' AS 고객표시명,
               city AS 도시
        FROM customers
        LIMIT 5
    """)

    con.close()
    print("[6] 정리")
    print("  - SELECT 는 읽기 전용: 어떤 문장을 실행해도 원본은 변하지 않습니다.")
    print("  - 별칭과 계산 열은 '보이는 화면'만 바꿉니다.")
    print("  - 다음 레벨: WHERE 로 '조건에 맞는 행만' 골라내기.")


if __name__ == "__main__":
    main()
