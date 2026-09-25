"""
Lecture 04 · Level 05 — 집계 함수와 GROUP BY
COUNT/SUM/AVG/MIN/MAX 로 요약하고, GROUP BY 로 도시별·카테고리별 소계를
만들고, HAVING 으로 '소계에 대한 조건'을 겁니다. WHERE(묶기 전 행 필터)와
HAVING(묶은 후 그룹 필터)의 차이가 이 레벨의 핵심입니다.
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
    print("실행 순서: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY\n")

    run(cur, 1, "전체 요약 — GROUP BY 없는 집계는 '테이블 전체 = 바구니 하나'", """
        SELECT COUNT(*) AS product_cnt,
               SUM(price) AS price_sum,
               ROUND(AVG(price), 1) AS price_avg,
               MIN(price) AS price_min,
               MAX(price) AS price_max
        FROM products
    """, note="여러 행이 요약값 '한 행'으로 접힙니다")

    run(cur, 2, "COUNT 의 세 표정 — *, 열, DISTINCT", """
        SELECT COUNT(*) AS all_rows,
               COUNT(manager_id) AS has_manager,
               COUNT(DISTINCT dept) AS dept_kinds
        FROM employees
    """, note="COUNT(열)은 NULL 을 빼고 셉니다 (사장님은 manager_id 가 NULL)")

    run(cur, 3, "도시별 고객 수 — GROUP BY 기본형", """
        SELECT city, COUNT(*) AS customer_cnt
        FROM customers
        GROUP BY city
        ORDER BY customer_cnt DESC
    """, note="결과 한 행 = 바구니(도시) 하나. 합계가 200인지 검산해 보세요")

    run(cur, 4, "도시×등급별 고객 수 — 그룹 기준 두 개", """
        SELECT city, grade, COUNT(*) AS cnt
        FROM customers
        GROUP BY city, grade
        ORDER BY city, grade
        LIMIT 8
    """, note="피벗 테이블에 필드 두 개를 넣은 것과 같습니다")

    run(cur, 5, "카테고리별 매출 — 주문상세×상품 연결 (다음 레벨 JOIN 맛보기)", """
        SELECT p.category,
               SUM(oi.quantity * p.price) AS revenue
        FROM order_items AS oi
        JOIN products AS p ON p.product_id = oi.product_id
        GROUP BY p.category
        ORDER BY revenue DESC
    """, note="수량×단가를 행마다 계산해 카테고리 바구니별로 SUM")

    run(cur, 6, "HAVING — 고객 35명 이상인 도시만 (소계에 거는 조건)", """
        SELECT city, COUNT(*) AS cnt
        FROM customers
        GROUP BY city
        HAVING COUNT(*) >= 35
        ORDER BY cnt DESC
    """, note="WHERE COUNT(*)>=35 는 에러 — 묶기 전에는 셀 수 없습니다")

    run(cur, 7, "WHERE + HAVING 합동 — 취소 제외 월별 주문, 80건 이상인 달만", """
        SELECT SUBSTR(ordered_at, 1, 7) AS month,
               COUNT(*) AS order_cnt
        FROM orders
        WHERE status <> '취소'
        GROUP BY month
        HAVING COUNT(*) >= 80
        ORDER BY month
    """, note="WHERE 는 행(주문 1건)을, HAVING 은 바구니(달)를 거릅니다")

    con.close()
    print("[정리] 소계 = GROUP BY + 집계 함수 / 소계 조건 = HAVING")
    print("  SELECT 에는 '바구니 이름표'와 '바구니 요약값'만 올 수 있습니다.")


if __name__ == "__main__":
    main()
