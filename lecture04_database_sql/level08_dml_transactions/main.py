"""
Lecture 04 · Level 08 — 데이터 변경과 트랜잭션
INSERT/UPDATE/DELETE 기본기와 함께, 여러 변경을 '전부 성공 아니면 전부
취소'로 묶는 트랜잭션(BEGIN/COMMIT/ROLLBACK)을 시연합니다.
WHERE 없는 UPDATE 사고를 롤백으로 되살리는 장면과, 재고 차감 트랜잭션의
성공/실패(부족 재고 → 전체 롤백) 시나리오가 핵심입니다.
"""

import pathlib
import sqlite3
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def sql(cur, statement, params=()):
    """SQL 문장을 보여 주고 실행합니다 (이 레벨은 변경문이 주인공)."""
    for line in statement.strip().splitlines():
        print(f"  SQL> {line.strip()}")
    cur.execute(statement, params)
    return cur


def show_stock(cur, label):
    """재고 상태를 한 줄로 요약 출력합니다."""
    cur.execute("SELECT product_id, stock FROM inventory WHERE product_id IN (1, 2)")
    state = ", ".join(f"상품{pid} 재고={s}" for pid, s in cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM orders")
    print(f"  [{label}] {state}, 주문 수={cur.fetchone()[0]}")


def place_order(con, cur, order_id, product_id, qty):
    """재고 차감 트랜잭션: 주문 생성 + 재고 차감 + 검증 → 커밋 또는 전체 롤백."""
    print(f"  주문 시도: 상품{product_id} × {qty}개 (주문번호 {order_id})")
    cur.execute("BEGIN")
    print("  SQL> BEGIN  -- 봉투 열기: 이후 변경은 아직 '연필 메모'")
    try:
        sql(cur, "INSERT INTO orders (order_id, customer_id, employee_id, ordered_at, status) "
                 "VALUES (?, ?, 1, '2025-12-30', '완료')", (order_id, 1))
        sql(cur, "UPDATE inventory SET stock = stock - ? WHERE product_id = ?",
            (qty, product_id))
        # 검증: 차감 결과가 음수면 규칙 위반 → 전체 취소
        cur.execute("SELECT stock FROM inventory WHERE product_id = ?", (product_id,))
        stock_after = cur.fetchone()[0]
        if stock_after < 0:
            raise ValueError(f"재고 부족 (차감하면 {stock_after}개)")
        cur.execute("COMMIT")
        print("  SQL> COMMIT  -- 검증 통과: 볼펜으로 확정")
    except Exception as e:
        cur.execute("ROLLBACK")
        print(f"  SQL> ROLLBACK  -- 문제 발생({e}) → 봉투째 파기, 전부 없던 일로")


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    con.isolation_level = None      # 자동 커밋 모드: BEGIN/COMMIT 을 직접 SQL 로 제어
    cur = con.cursor()

    # 실습용 재고 테이블: 상품 1~10번에 각각 10개씩
    cur.execute("CREATE TABLE inventory (product_id INTEGER PRIMARY KEY, stock INTEGER)")
    cur.executemany("INSERT INTO inventory VALUES (?, ?)", [(i, 10) for i in range(1, 11)])
    print(f"실습 DB 준비 완료: {db_path.name} (+ inventory 재고 테이블, 상품당 10개)\n")

    # ------------------------------------------------------------------
    print("[1] INSERT — 신규 고객 추가")
    cur.execute("SELECT COUNT(*) FROM customers")
    print(f"  추가 전 고객 수: {cur.fetchone()[0]}")
    sql(cur, "INSERT INTO customers (customer_id, name, city, grade, joined_at) "
             "VALUES (204, '박겨울', '서울', 'BASIC', '2025-12-20')")
    cur.execute("SELECT COUNT(*) FROM customers")
    print(f"  추가 후 고객 수: {cur.fetchone()[0]} → 한 행이 늘었습니다\n")

    # ------------------------------------------------------------------
    print("[2] UPDATE — '조준(SELECT) → 발사(UPDATE)' 안전 수칙")
    sql(cur, "SELECT customer_id, name, grade FROM customers WHERE customer_id = 204")
    print(f"  조준 결과: {cur.fetchall()} ← 정확히 1행인지 확인!")
    sql(cur, "UPDATE customers SET grade = 'GOLD' WHERE customer_id = 204")
    cur.execute("SELECT grade FROM customers WHERE customer_id = 204")
    print(f"  발사 후 등급: {cur.fetchone()[0]}\n")

    # ------------------------------------------------------------------
    print("[3] WHERE 를 빼먹은 UPDATE 사고 — 그리고 ROLLBACK 지우개")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    before_vip = cur.fetchone()[0]
    print(f"  사고 전 VIP 수: {before_vip}")
    cur.execute("BEGIN")
    print("  SQL> BEGIN")
    sql(cur, "UPDATE customers SET grade = 'VIP'   -- WHERE 를 깜빡했다!")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    print(f"  사고 직후 VIP 수: {cur.fetchone()[0]} ← 전 고객이 VIP! (아직 연필 메모)")
    cur.execute("ROLLBACK")
    print("  SQL> ROLLBACK")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = 'VIP'")
    print(f"  롤백 후 VIP 수: {cur.fetchone()[0]} → 원상복구. 커밋 전엔 지우개가 있습니다\n")

    # ------------------------------------------------------------------
    print("[4] 재고 차감 트랜잭션 — 성공 사례 (주문 + 차감 + 검증 → COMMIT)")
    show_stock(cur, "시도 전")
    place_order(con, cur, order_id=1001, product_id=1, qty=3)
    show_stock(cur, "시도 후")
    print("  → 주문 1건 증가 + 재고 10→7. 두 변경이 함께 확정되었습니다.\n")

    # ------------------------------------------------------------------
    print("[5] 재고 차감 트랜잭션 — 실패 사례 (재고 7개인데 20개 주문)")
    show_stock(cur, "시도 전")
    place_order(con, cur, order_id=1002, product_id=1, qty=20)
    show_stock(cur, "시도 후")
    print("  → 주문 수도 재고도 시도 전과 동일. '주문만 저장되고 재고는 그대로'")
    print("    같은 반쪽짜리 상태가 없습니다 — 이것이 원자성(전부 아니면 전무).\n")

    con.close()
    print("[정리] 변경문의 생명은 WHERE, 묶음 변경의 생명은 트랜잭션.")
    print("  실무 뼈대: BEGIN → 변경들 → 검증 → COMMIT (문제 시 ROLLBACK).")


if __name__ == "__main__":
    main()
