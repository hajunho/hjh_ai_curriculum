"""
Lecture 04 · Level 09 — 인덱스와 쿼리 성능
30만 행짜리 big_orders 테이블을 만들어, 같은 검색을 인덱스 전/후로
실측합니다. EXPLAIN QUERY PLAN 으로 DB의 작전(SCAN vs SEARCH)을 읽고,
중간 일치 LIKE 처럼 인덱스가 소용없는 경우와 인덱스의 쓰기 비용까지
숫자로 확인합니다.
"""

import pathlib
import random
import sqlite3
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data

N_ROWS = 300_000      # 실험대 크기
N_REPEAT = 200        # 같은 검색의 반복 횟수 (시간을 잴 수 있게)


def explain(cur, query, params=()):
    """EXPLAIN QUERY PLAN 으로 DB의 실행 작전을 출력합니다."""
    print(f"  SQL> EXPLAIN QUERY PLAN {query}")
    cur.execute("EXPLAIN QUERY PLAN " + query, params)
    for row in cur.fetchall():
        print(f"       작전: {row[3]}")


def measure(cur, query, params_list):
    """쿼리를 반복 실행해 총 소요 시간(초)을 실측합니다."""
    t0 = time.perf_counter()
    for p in params_list:
        cur.execute(query, p)
        cur.fetchall()
    return time.perf_counter() - t0


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rng = random.Random(42)   # seed 고정: 매번 같은 실험 데이터

    # ------------------------------------------------------------------
    print(f"[1] 실험대 준비 — big_orders 테이블에 {N_ROWS:,}행 생성")
    cur.execute("CREATE TABLE big_orders ("
                "order_id INTEGER PRIMARY KEY, customer_id INTEGER, "
                "ordered_at TEXT, status TEXT, amount INTEGER)")
    rows = [(i,
             rng.randint(1, 50_000),
             f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
             rng.choice(["완료", "취소", "배송중"]),
             rng.randint(1_000, 500_000))
            for i in range(1, N_ROWS + 1)]
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO big_orders VALUES (?,?,?,?,?)", rows)
    con.commit()
    print(f"  생성 완료 ({time.perf_counter() - t0:.2f}초). 고객 5만 명의 주문 30만 건.\n")

    query = "SELECT * FROM big_orders WHERE customer_id = ?"
    targets = [(rng.randint(1, 50_000),) for _ in range(N_REPEAT)]

    # ------------------------------------------------------------------
    print(f"[2] 인덱스 없이 검색 — 고객번호 검색 {N_REPEAT}회 실측")
    explain(cur, query, (7,))
    t_before = measure(cur, query, targets)
    print(f"  소요 시간: {t_before:.3f}초  ← 검색 1회마다 {N_ROWS:,}행 전부 훑기(SCAN)\n")

    # ------------------------------------------------------------------
    print("[3] 인덱스 생성 — 색인 한 권 만들기")
    print("  SQL> CREATE INDEX idx_big_customer ON big_orders (customer_id)")
    t0 = time.perf_counter()
    cur.execute("CREATE INDEX idx_big_customer ON big_orders (customer_id)")
    con.commit()
    print(f"  생성 시간: {time.perf_counter() - t0:.2f}초 (한 번만 내는 비용)\n")

    # ------------------------------------------------------------------
    print(f"[4] 인덱스 후 같은 검색 {N_REPEAT}회 — 쿼리 문장은 그대로!")
    explain(cur, query, (7,))
    t_after = measure(cur, query, targets)
    speedup = t_before / t_after if t_after > 0 else float("inf")
    print(f"  소요 시간: {t_after:.4f}초")
    print(f"  → {t_before:.3f}초 → {t_after:.4f}초, 약 {speedup:,.0f}배 빨라짐\n")

    # ------------------------------------------------------------------
    print("[5] 인덱스가 소용없는 검색 — 중간 일치 LIKE 는 여전히 SCAN")
    explain(cur, "SELECT * FROM big_orders WHERE status LIKE '%송%'")
    print("  → 색인은 '첫 글자부터' 정렬이라 중간 일치는 못 찾습니다.")
    explain(cur, "SELECT * FROM big_orders WHERE customer_id = ? AND status = '완료'", (7,))
    print("  → 같은 쿼리라도 인덱스 열(customer_id)이 있으면 SEARCH 로 좁힌 뒤 거릅니다.\n")

    # ------------------------------------------------------------------
    print("[6] 인덱스의 청구서 — 쓰기(INSERT)는 오히려 느려집니다")
    extra = [(N_ROWS + i, rng.randint(1, 50_000), "2025-12-31", "완료", 1000)
             for i in range(1, 10_001)]
    cur.execute("CREATE TABLE plain_copy AS SELECT * FROM big_orders WHERE 0")  # 인덱스 없는 빈 사본
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO plain_copy VALUES (?,?,?,?,?)", extra)
    con.commit()
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO big_orders VALUES (?,?,?,?,?)", extra)  # 인덱스 있는 테이블
    con.commit()
    t_indexed = time.perf_counter() - t0
    print(f"  1만 행 INSERT — 인덱스 없음: {t_plain:.3f}초 / 인덱스 있음: {t_indexed:.3f}초")
    print("  → 인덱스는 검색을 사고 쓰기를 지불하는 거래입니다. 필요한 열에만!\n")

    con.close()
    print("[정리] 느린 쿼리 진단 2단계: EXPLAIN 으로 SCAN 확인 → WHERE 열에 인덱스.")
    print("  인덱스 후보 = WHERE/JOIN 에 자주 쓰는 열. 전부 걸면 쓰기가 웁니다.")


if __name__ == "__main__":
    main()
