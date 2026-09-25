"""
Lecture 04 · Level 06 — JOIN: 여러 테이블 연결
번호(외래키)만 적힌 주문 장부에 고객·상품 정보를 붙여 '누가 무엇을 샀나'를
분석합니다. INNER JOIN, 다중 조인, JOIN+GROUP BY, LEFT JOIN 으로 주문 없는
고객 찾기, 그리고 조인이 집계를 부풀리는 팬아웃(fan-out) 함정까지 시연합니다.
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
    print("관계도: customers ──< orders >── employees / orders ──< order_items >── products")

    # LEFT JOIN 실습을 위해 '아직 주문이 없는' 신규 가입 고객 3명을 넣어 둡니다.
    cur.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?)",
        [(201, "신새봄", "서울", "BASIC", "2025-12-01"),
         (202, "한이든", "부산", "BASIC", "2025-12-02"),
         (203, "오가을", "대전", "BASIC", "2025-12-03")])
    con.commit()
    print("(준비) 주문 이력이 없는 신규 고객 3명(201~203)을 추가했습니다.\n")

    run(cur, 1, "두 테이블 조인 — 주문 장부의 번호가 이름으로 바뀝니다", """
        SELECT o.order_id, c.name AS customer, c.city, o.ordered_at, o.status
        FROM orders AS o
        JOIN customers AS c ON c.customer_id = o.customer_id
        ORDER BY o.order_id
        LIMIT 5
    """, note="ON 절 = 외래키(o.customer_id)와 기본키(c.customer_id)의 짝 맞추기")

    run(cur, 2, "네 테이블 사슬 조인 — 누가 무엇을 몇 개, 얼마어치 샀나", """
        SELECT c.name AS customer, p.name AS product,
               oi.quantity, oi.quantity * p.price AS amount
        FROM order_items AS oi
        JOIN orders    AS o ON o.order_id    = oi.order_id
        JOIN customers AS c ON c.customer_id = o.customer_id
        JOIN products  AS p ON p.product_id  = oi.product_id
        ORDER BY o.order_id
        LIMIT 5
    """, note="외래키 화살표를 따라 ON 을 이어 붙이면 길을 잃지 않습니다")

    run(cur, 3, "조인 + 집계 — 완료 주문 기준 고객별 총구매액 톱5", """
        SELECT c.name, c.grade,
               SUM(oi.quantity * p.price) AS total_amount
        FROM order_items AS oi
        JOIN orders    AS o ON o.order_id    = oi.order_id
        JOIN customers AS c ON c.customer_id = o.customer_id
        JOIN products  AS p ON p.product_id  = oi.product_id
        WHERE o.status = '완료'
        GROUP BY c.customer_id, c.name, c.grade
        ORDER BY total_amount DESC
        LIMIT 5
    """)

    run(cur, 4, "LEFT JOIN — 고객 '전원'과 각자의 주문 수 (없으면 0)", """
        SELECT c.customer_id, c.name,
               COUNT(o.order_id) AS order_cnt
        FROM customers AS c
        LEFT JOIN orders AS o ON o.customer_id = c.customer_id
        GROUP BY c.customer_id, c.name
        ORDER BY order_cnt ASC
        LIMIT 5
    """, note="INNER JOIN 이었다면 주문 0건 고객은 아예 사라졌을 것입니다")

    run(cur, 5, "안티 조인 — 한 번도 주문하지 않은 고객(휴면 고객) 명단", """
        SELECT c.customer_id, c.name, c.city, c.grade
        FROM customers AS c
        LEFT JOIN orders AS o ON o.customer_id = c.customer_id
        WHERE o.order_id IS NULL
    """, note="짝이 없어 NULL 로 남은 행만 고르기 = '없는 것 찾기' 정석 패턴")

    # ------------------------------------------------------------------
    print("[6] 팬아웃 함정 — 같은 '주문 수' 질문, 세 가지 답")
    fanout = []
    cur.execute("SELECT COUNT(*) FROM orders")
    fanout.append(("(a) orders 단독 COUNT(*)", cur.fetchone()[0], "정답"))
    cur.execute("""
        SELECT COUNT(*)
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id""")
    fanout.append(("(b) order_items 조인 후 COUNT(*)", cur.fetchone()[0],
                   "부풀려짐! 주문 1건이 상품 수만큼 복제"))
    cur.execute("""
        SELECT COUNT(DISTINCT o.order_id)
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id""")
    fanout.append(("(c) 조인 후 COUNT(DISTINCT order_id)", cur.fetchone()[0],
                   "DISTINCT 로 복제 제거 → 다시 정답"))
    widths = [max(disp_width(r[i]) for r in fanout) for i in range(3)]
    for r in fanout:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))
    print("  → 조인 후 집계가 '느낌보다 크면' 팬아웃부터 의심하세요.\n")

    con.close()
    print("[정리] JOIN = 키 들고 다른 장부 찾아가기 / LEFT = 왼쪽 전원 생존")
    print("  1:N 조인은 행을 불립니다 — 집계는 COUNT(DISTINCT 키)로 방어.")


if __name__ == "__main__":
    main()
