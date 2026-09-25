"""
Lecture 04 · Level 07 — 서브쿼리와 CTE
쿼리의 결과를 다른 쿼리의 재료로 쓰는 법을 연습합니다.
스칼라 서브쿼리(값 하나), IN 서브쿼리(목록), 상관 서브쿼리(행마다 다시 계산),
그리고 WITH(CTE)로 '평균 이상 구매 고객' 같은 다단계 분석을 읽기 좋게
정리하는 것까지 다룹니다.
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
    print("서브쿼리 = 괄호 속 쪽지: 안쪽 답을 바깥 쿼리의 빈칸에 끼워 넣기\n")

    # 쪽지 1: 기준값을 먼저 눈으로 확인
    cur.execute("SELECT ROUND(AVG(price), 0) FROM products")
    print(f"(쪽지 1) 상품 평균가 = {cur.fetchone()[0]:,.0f}원 — 이 값이 아래 괄호에 끼워집니다.\n")

    run(cur, 1, "스칼라 서브쿼리 — 평균보다 비싼 상품", """
        SELECT name, price
        FROM products
        WHERE price > (SELECT AVG(price) FROM products)
        ORDER BY price DESC
    """, note="괄호가 먼저 실행되어 '값 하나'로 바뀐다고 읽으면 됩니다")

    run(cur, 2, "IN 서브쿼리 — '노트북'을 산 적 있는 고객 (일부)", """
        SELECT customer_id, name, city
        FROM customers
        WHERE customer_id IN (
            SELECT o.customer_id
            FROM orders AS o
            WHERE o.order_id IN (
                SELECT oi.order_id
                FROM order_items AS oi
                JOIN products AS p ON p.product_id = oi.product_id
                WHERE p.name = '노트북'))
        ORDER BY customer_id
        LIMIT 6
    """, note="안쪽 답(주문번호 목록 → 고객번호 목록)이 IN 의 목록 자리에")

    run(cur, "3a", "준비 — 부서별 평균 급여 (대조용)", """
        SELECT dept, ROUND(AVG(salary), 0) AS avg_salary
        FROM employees
        GROUP BY dept
    """)

    run(cur, "3b", "상관 서브쿼리 — 자기 부서 평균보다 급여가 높은 직원", """
        SELECT e.name, e.dept, e.salary
        FROM employees AS e
        WHERE e.salary > (SELECT AVG(e2.salary)
                          FROM employees AS e2
                          WHERE e2.dept = e.dept)
        ORDER BY e.dept, e.salary DESC
    """, note="안쪽이 바깥 행의 e.dept 를 참조 → 행마다 기준이 달라집니다")

    run(cur, 4, "CTE(WITH) — 평균 이상 구매 고객 (완료 주문 기준)", """
        WITH customer_totals AS (
            SELECT o.customer_id,
                   SUM(oi.quantity * p.price) AS total
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '완료'
            GROUP BY o.customer_id
        )
        SELECT c.name, c.grade, t.total
        FROM customer_totals AS t
        JOIN customers AS c ON c.customer_id = t.customer_id
        WHERE t.total > (SELECT AVG(total) FROM customer_totals)
        ORDER BY t.total DESC
        LIMIT 6
    """, note="1단계에 이름을 붙이고 2단계에서 '두 번' 재사용 — CTE 의 힘")

    run(cur, 5, "CTE 여러 단계 — 월별 매출을 만들고, 최고 매출 달 찾기", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '완료'
            GROUP BY month
        )
        SELECT month, revenue
        FROM monthly
        WHERE revenue = (SELECT MAX(revenue) FROM monthly)
    """, note="리포트 쿼리의 전형: 단계별 CTE → 마지막에 답 하나")

    con.close()
    print("[정리] 값 하나 → 스칼라 / 목록 → IN / 행마다 기준 → 상관")
    print("  괄호가 두 겹이 되면 CTE 로 승격해 '읽히는 쿼리'를 만드세요.")


if __name__ == "__main__":
    main()
