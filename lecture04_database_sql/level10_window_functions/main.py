"""
Lecture 04 · Level 10 — 윈도우 함수와 분석 쿼리
행을 접지 않고 옆에 요약을 붙이는 윈도우 함수(OVER)를 연습합니다.
GROUP BY 와의 차이, ROW_NUMBER 로 고객별 구매 순번, RANK 3형제의 동점
처리, 부서별 매출 순위, 누적 매출, 3개월 이동합계까지 — 분석 보고서의
단골 패턴을 전부 실행해 봅니다.
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
    print("관전 포인트: 각 결과에서 '행이 접혔는가, 유지됐는가'를 확인하세요.\n")

    run(cur, "1a", "GROUP BY — 카테고리 평균가 (행이 5줄로 '접힘')", """
        SELECT category, ROUND(AVG(price), 0) AS avg_price
        FROM products
        GROUP BY category
    """)

    run(cur, "1b", "윈도우 — 같은 평균가를 '접지 않고' 옆에 붙임 (10행 유지)", """
        SELECT name, category, price,
               ROUND(AVG(price) OVER (PARTITION BY category), 0) AS cat_avg,
               price - ROUND(AVG(price) OVER (PARTITION BY category), 0) AS diff
        FROM products
        ORDER BY category, price DESC
    """, note="행마다 '자기 카테고리 창'을 내다보고 평균을 포스트잇으로 붙입니다")

    run(cur, 2, "ROW_NUMBER — 고객별 구매 순번, CTE 로 감싸 '첫 주문'만", """
        WITH numbered AS (
            SELECT customer_id, order_id, ordered_at,
                   ROW_NUMBER() OVER (PARTITION BY customer_id
                                      ORDER BY ordered_at, order_id) AS nth
            FROM orders
        )
        SELECT c.name, n.order_id, n.ordered_at, n.nth
        FROM numbered AS n
        JOIN customers AS c ON c.customer_id = n.customer_id
        WHERE n.nth = 1
        ORDER BY n.ordered_at
        LIMIT 5
    """, note="'그룹마다 최초 1건' = ROW_NUMBER + 바깥 필터, 실무 최빈 공식")

    run(cur, 3, "동점 처리 3형제 — 상품 가격 순위 (동점 가격이 있어야 차이가 보임)", """
        SELECT name, price,
               ROW_NUMBER() OVER (ORDER BY price DESC) AS row_num,
               RANK()       OVER (ORDER BY price DESC) AS rnk,
               DENSE_RANK() OVER (ORDER BY price DESC) AS dense_rnk
        FROM products
        ORDER BY price DESC
    """, note="같은 가격(동점)에서 RANK 는 다음 순위를 건너뛰고 DENSE_RANK 는 이어 갑니다")

    run(cur, 4, "부서별 매출 순위 — 직원별 처리 매출(CTE) 후 부서 안 RANK", """
        WITH emp_sales AS (
            SELECT e.employee_id, e.name, e.dept,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN employees AS e   ON e.employee_id = o.employee_id
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p     ON p.product_id = oi.product_id
            WHERE o.status = '완료'
            GROUP BY e.employee_id, e.name, e.dept
        )
        SELECT dept, name, revenue,
               RANK() OVER (PARTITION BY dept ORDER BY revenue DESC) AS dept_rank
        FROM emp_sales
        ORDER BY dept, dept_rank
        LIMIT 10
    """, note="'지점별/부서별 순위' 보고서의 골격: 집계 CTE → PARTITION BY 순위")

    run(cur, 5, "누적 매출 — 월별 매출 옆에 연초부터의 누적", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '완료'
            GROUP BY month
        )
        SELECT month, revenue,
               SUM(revenue) OVER (ORDER BY month) AS cum_revenue
        FROM monthly
        LIMIT 6
    """, note="ORDER BY 가 창을 '처음~현재 행'으로 만들어 누적이 됩니다")

    run(cur, 6, "3개월 이동합계 — 창 크기를 직접 지정 (ROWS BETWEEN)", """
        WITH monthly AS (
            SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
                   SUM(oi.quantity * p.price) AS revenue
            FROM orders AS o
            JOIN order_items AS oi ON oi.order_id = o.order_id
            JOIN products AS p ON p.product_id = oi.product_id
            WHERE o.status = '완료'
            GROUP BY month
        )
        SELECT month, revenue,
               SUM(revenue) OVER (ORDER BY month
                                  ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS mov3
        FROM monthly
        LIMIT 6
    """, note="직전 2행+현재 행 = 최근 3개월. 추세를 부드럽게 보는 보고서 기법")

    con.close()
    print("[정리] 행 수가 줄면 GROUP BY, 유지되면 윈도우(OVER).")
    print("  창 정의 3요소: PARTITION BY(범위) / ORDER BY(순서) / ROWS(크기).")


if __name__ == "__main__":
    main()
