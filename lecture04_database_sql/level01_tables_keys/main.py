"""
Lecture 04 · Level 01 — 테이블·행·열·기본키·외래키
실습용 쇼핑몰 DB(hjh_shop.db)를 만들고, DB 스스로에게 구조를 물어봅니다.
테이블 목록 → 각 테이블의 열과 기본키 → 행 수 → 외래키 관계도 순서로
스키마(schema)를 읽는 요령을 익힙니다. 처음 보는 회사 DB를 파악할 때도
이 순서를 그대로 쓰면 됩니다.
"""

import pathlib
import sqlite3
import sys
import unicodedata

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def disp_width(text):
    """한글은 화면에서 2칸을 차지하므로, 표 정렬용 표시 폭을 계산합니다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width):
    return str(text) + " " * (width - disp_width(text))


def show_table(cols, rows):
    """조회 결과를 열 폭을 맞춘 표로 출력합니다."""
    widths = [max(disp_width(c), *(disp_width(r[i]) for r in rows)) if rows else disp_width(c)
              for i, c in enumerate(cols)]
    print("  " + " | ".join(pad(c, w) for c, w in zip(cols, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(pad(v, w) for v, w in zip(r, widths)))


def run(cur, sql):
    """SQL 문장을 보여 주고 실행한 뒤 결과를 표로 출력합니다."""
    print(f"  SQL> {sql}")
    cur.execute(sql)
    show_table([d[0] for d in cur.description], cur.fetchall())
    print()


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"실습 DB 준비 완료: {db_path.name}\n")

    # ------------------------------------------------------------------
    print("[1] 테이블 목록 — 이 DB에는 어떤 장부(테이블)들이 있나?")
    run(cur, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")

    # ------------------------------------------------------------------
    print("[2] 각 테이블의 구조 — 열 이름·자료형·기본키(pk가 1 이상이면 기본키)")
    tables = ["customers", "products", "employees", "orders", "order_items"]
    for t in tables:
        print(f"  SQL> PRAGMA table_info({t})")
        cur.execute(f"PRAGMA table_info({t})")
        rows = [(r[1], r[2], "PK" + str(r[5]) if r[5] else "") for r in cur.fetchall()]
        show_table(["column", "type", "key"], rows)
        print()

    # ------------------------------------------------------------------
    print("[3] 테이블별 행 수 — 데이터 규모 확인")
    counts = []
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")   # COUNT(*)는 행 수를 셉니다
        counts.append((t, cur.fetchone()[0]))
    show_table(["table", "rows"], counts)
    print()

    # ------------------------------------------------------------------
    print("[4] 외래키 관계도 — _id 열이 어느 테이블의 기본키를 가리키는가")
    relations = [
        ("orders.customer_id",      "→ customers.customer_id", "주문의 주인 고객"),
        ("orders.employee_id",      "→ employees.employee_id", "주문을 처리한 직원"),
        ("order_items.order_id",    "→ orders.order_id",       "어느 주문의 상세인가"),
        ("order_items.product_id",  "→ products.product_id",   "어떤 상품인가"),
        ("employees.manager_id",    "→ employees.employee_id", "직속 상사(자기참조)"),
    ]
    show_table(["foreign key", "references", "meaning"], relations)
    print("""
  customers ──< orders >── employees
                  │            └──(manager_id 자기참조)
                  └──< order_items >── products
  (──< 는 1:N 관계 — 고객 1명이 주문 N건을 가짐)
""")

    # ------------------------------------------------------------------
    print("[5] 참조 확인 — 주문 장부의 customer_id는 고객 명부의 행 번호다")
    run(cur, "SELECT order_id, customer_id, ordered_at, status "
             "FROM orders WHERE customer_id = 7 LIMIT 3")
    run(cur, "SELECT customer_id, name, city, grade "
             "FROM customers WHERE customer_id = 7")
    print("  → 주문 장부에는 이름 대신 '고객 명부 7번'이라는 참조만 적혀 있습니다.")
    print("    고객 정보가 바뀌어도 customers 한 줄만 고치면 되는 이유입니다.")

    con.close()
    print("\n[6] 정리: 처음 보는 DB는 (1)테이블 목록 (2)열·기본키 (3)행 수")
    print("    (4)외래키 관계 순서로 파악합니다. 다음 레벨부터 본격 조회(SELECT)!")


if __name__ == "__main__":
    main()
