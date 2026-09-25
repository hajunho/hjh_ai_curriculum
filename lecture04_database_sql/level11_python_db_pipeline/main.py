"""
Lecture 04 · Level 11 — 파이썬 연동과 데이터 파이프라인
SQL 과 파이썬을 잇는 마지막 레벨입니다. 파라미터 바인딩(?)이 SQL 인젝션을
막는 원리를 안전한 로컬 예제로 시연하고, pandas.read_sql 로 조회 결과를
DataFrame 으로 받아 '추출(SQL) → 가공(pandas) → 적재(CSV)+요약 리포트'의
미니 ETL 파이프라인을 완성합니다.
"""

import pathlib
import sqlite3
import sys

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[1] / "common"))
import hjh_data


def search_city_unsafe(cur, user_input):
    """[나쁜 예] 사용자 입력을 문자열로 이어 붙여 SQL 을 조립 — 인젝션에 노출."""
    sql = f"SELECT name, city FROM customers WHERE city = '{user_input}'"
    print(f"  (조립된 문장) {sql}")
    cur.execute(sql)
    return cur.fetchall()


def search_city_safe(cur, user_input):
    """[좋은 예] 문장은 서식(?), 값은 별도 전달 — 입력이 문장이 될 수 없음."""
    sql = "SELECT name, city FROM customers WHERE city = ?"
    print(f"  (문장) {sql}   (값) {user_input!r}")
    cur.execute(sql, (user_input,))     # 값이 하나여도 튜플 (쉼표!)
    return cur.fetchall()


def extract(con):
    """[E] 추출: 필요한 데이터만 SQL 로 줄여서 가져옵니다 (완료 주문만)."""
    sql = """
        SELECT SUBSTR(o.ordered_at, 1, 7) AS month,
               p.category,
               c.name AS customer,
               oi.quantity * p.price AS amount
        FROM orders AS o
        JOIN customers AS c   ON c.customer_id = o.customer_id
        JOIN order_items AS oi ON oi.order_id = o.order_id
        JOIN products AS p     ON p.product_id = oi.product_id
        WHERE o.status = ?
    """
    return pd.read_sql(sql, con, params=("완료",))


def transform(df):
    """[T] 가공: pandas 로 월×카테고리 피벗을 만들고 월 합계를 붙입니다."""
    pivot = pd.pivot_table(df, values="amount", index="month",
                           columns="category", aggfunc="sum", fill_value=0)
    pivot["월합계"] = pivot.sum(axis=1)
    return pivot


def load_and_report(df, pivot, out_dir):
    """[L] 적재: 요약표를 CSV 로 저장하고, 경영진용 텍스트 리포트를 출력합니다."""
    out_path = out_dir / "monthly_category_revenue.csv"
    pivot.to_csv(out_path, encoding="utf-8-sig")   # 엑셀에서 한글이 깨지지 않는 인코딩
    print(f"  저장 완료 → {out_path}")

    total = df["amount"].sum()
    best_month = pivot["월합계"].idxmax()
    best_cat = pivot.drop(columns="월합계").sum().idxmax()
    top3 = df.groupby("customer")["amount"].sum().nlargest(3)
    print("\n  ---- 자동 요약 리포트 (완료 주문 기준) ----")
    print(f"  · 연간 총매출        : {total:,}원")
    print(f"  · 최고 매출 달       : {best_month} ({pivot.loc[best_month, '월합계']:,}원)")
    print(f"  · 매출 1위 카테고리  : {best_cat}")
    print("  · 구매액 톱3 고객    :")
    for name, amt in top3.items():
        print(f"      {name}: {amt:,}원")


def main():
    db_path = BASE / "hjh_shop.db"
    hjh_data.build_sqlite(str(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(f"실습 DB 준비 완료: {db_path.name}\n")

    # ------------------------------------------------------------------
    print("[1] 파이썬에서 SQL 실행 네 단계 — 연결→커서→실행(바인딩)→결과")
    cur.execute("SELECT COUNT(*) FROM customers WHERE grade = ?", ("VIP",))
    print(f"  VIP 고객 수: {cur.fetchone()[0]}명 (값 'VIP' 는 ? 빈칸으로 전달)\n")

    # ------------------------------------------------------------------
    print("[2] SQL 인젝션 시연 — 같은 악성 입력, 두 가지 운명 (로컬 실습 DB 안에서만)")
    evil = "서울' OR '1'='1"      # 조건을 '항상 참'으로 바꾸려는 고전적 입력
    print(f"  악성 입력: {evil!r}")
    print("  (a) 문자열 이어붙이기(f-string) 방식:")
    rows = search_city_unsafe(cur, evil)
    print(f"      → {len(rows)}건 유출! 조건이 '항상 참'인 문장으로 변조되었습니다.")
    print("  (b) ? 바인딩 방식:")
    rows = search_city_safe(cur, evil)
    print(f"      → {len(rows)}건. 입력 전체가 '도시 이름이라는 값'으로 취급되어")
    print("        그런 도시는 없으므로 0건 — 이것이 정답입니다.")
    print("  수칙: SQL 에 사용자 값이 끼면 무조건 ? 바인딩!\n")

    # ------------------------------------------------------------------
    print("[3] Extract — pandas.read_sql 로 조인 결과를 DataFrame 으로")
    df = extract(con)
    print(f"  받아온 데이터: {df.shape[0]}행 × {df.shape[1]}열 (완료 주문의 품목별 금액)")
    print(df.head(3).to_string(index=False))
    print("  → 줄이는 일(WHERE/JOIN)은 SQL 이, 다듬는 일은 pandas 가 합니다.\n")

    # ------------------------------------------------------------------
    print("[4] Transform — 월×카테고리 매출 피벗 (일부만 표시)")
    pivot = transform(df)
    print(pivot.head(4).to_string())
    print()

    # ------------------------------------------------------------------
    print("[5] Load + 리포트 — CSV 저장과 자동 요약")
    out_dir = BASE / "outputs"
    out_dir.mkdir(exist_ok=True)
    load_and_report(df, pivot, out_dir)

    con.close()
    print("\n[정리] 이 스크립트가 곧 파이프라인입니다: 예약 실행(cron 등)에 걸면")
    print("  '매주 월요일 아침 자동 리포트'가 됩니다. 회사 DB로 바꿀 때는")
    print("  connect 부분만 해당 DB 드라이버로 교체하면 SQL 은 그대로 통합니다.")


if __name__ == "__main__":
    main()
