"""
hjh_data.py — 커리큘럼 전용 합성 데이터 생성기

이 모듈은 hjh_ai_curriculum 강의 전체에서 쓰는 연습용 데이터를 '직접 만들어' 냅니다.
외부 데이터셋을 내려받지 않으므로 저작권·라이선스 문제가 전혀 없고,
인터넷이 끊긴 교실에서도 똑같이 동작합니다.

모든 함수는 seed 를 받아 항상 같은 결과를 재현합니다.
강의 노트에 적힌 숫자와 실행 결과가 달라지지 않도록 하기 위함입니다.

작성: 하준호 (hajunho) · MIT License
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# numpy 는 있으면 쓰고 없으면 순수 파이썬으로 동작합니다.
# (lecture01~02 는 아직 numpy 를 배우기 전이라 의존성이 없어야 합니다.)
try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# 0. 공통 유틸
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    """독립적인 난수 발생기. 전역 random 상태를 건드리지 않습니다."""
    return random.Random(seed)


def to_csv(rows: list[dict], path: str) -> str:
    """딕셔너리 리스트를 CSV 파일로 저장하고 경로를 돌려줍니다."""
    import csv
    if not rows:
        raise ValueError("빈 데이터는 저장할 수 없습니다.")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def head(rows: list[dict], n: int = 5) -> None:
    """표 형태로 앞부분만 예쁘게 출력합니다 (pandas 없이도 동작)."""
    if not rows:
        print("(데이터 없음)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r[c])) for r in rows[:n])) for c in cols}
    line = " | ".join(str(c).ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows[:n]:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print(f"... 전체 {len(rows)}행")


# ---------------------------------------------------------------------------
# 1. 매출 데이터 — lecture03, 05, 07
# ---------------------------------------------------------------------------

STORES = ["강남점", "홍대점", "부산점", "대전점", "수원점"]
CATEGORIES = ["커피", "베이커리", "샌드위치", "디저트", "음료"]


def sales_table(n_days: int = 365, seed: int = 42) -> list[dict]:
    """
    가상의 카페 체인 일별 매출 데이터.

    일부러 다음 성질을 심어 두었습니다 (강의에서 하나씩 발견하게 됩니다):
      - 주말 매출이 평일보다 높다          -> groupby / 요일 효과
      - 여름에 음료 매출이 뛴다             -> 계절성 / 시계열
      - 결측치와 음수 이상치가 섞여 있다     -> 데이터 정제
      - 광고비와 매출이 상관관계를 가진다    -> 회귀 / 상관 vs 인과
    """
    rng = _rng(seed)
    rows: list[dict] = []
    for day in range(n_days):
        weekday = day % 7                      # 0=월 ... 6=일
        is_weekend = weekday >= 5
        season = math.sin(2 * math.pi * day / 365.0)   # -1(겨울) ~ +1(여름)

        for store in STORES:
            store_power = 1.0 + 0.15 * STORES.index(store)
            ad_cost = round(rng.uniform(50_000, 400_000), -3)

            for cat in CATEGORIES:
                base = 300_000 * store_power
                if cat == "음료":
                    base *= 1.0 + 0.45 * season          # 여름에 급증
                if cat == "커피":
                    base *= 1.3
                if is_weekend:
                    base *= 1.25
                noise = rng.gauss(1.0, 0.18)
                revenue = base * noise + ad_cost * 0.35

                # 데이터 정제 연습용 오염: 1% 결측, 0.5% 음수
                u = rng.random()
                if u < 0.010:
                    revenue_out = None
                elif u < 0.015:
                    revenue_out = -abs(round(revenue))
                else:
                    revenue_out = round(revenue)

                rows.append({
                    "date": f"2025-{day // 31 + 1:02d}-{day % 31 + 1:02d}",
                    "day_index": day,
                    "weekday": ["월", "화", "수", "목", "금", "토", "일"][weekday],
                    "store": store,
                    "category": cat,
                    "ad_cost": int(ad_cost),
                    "revenue": revenue_out,
                })
    return rows


# ---------------------------------------------------------------------------
# 2. 고객 이탈 데이터 — lecture06, 07
# ---------------------------------------------------------------------------

def churn_table(n: int = 2000, seed: int = 7) -> list[dict]:
    """
    구독 서비스 고객 이탈 데이터 (이진 분류용).

    진짜 신호:  이용일수↓, 고객센터 문의↑, 요금제 변경 이력↑  -> 이탈 확률↑
    가짜 신호:  customer_id 는 아무 의미 없음 (누설 변수 연습용)
    이탈률은 약 18% 로, 불균형 데이터 실습에 적합합니다.
    """
    rng = _rng(seed)
    rows = []
    for i in range(n):
        tenure = rng.randint(1, 60)                       # 가입 개월
        monthly_fee = rng.choice([9900, 14900, 19900, 29900])
        usage_days = max(0, min(30, int(rng.gauss(18, 7))))
        support_calls = max(0, int(rng.expovariate(1 / 1.3)))
        plan_changes = max(0, int(rng.expovariate(1 / 0.6)))
        is_auto_pay = rng.random() < 0.7

        # 로그오즈를 직접 설계 -> 정답이 있는 데이터
        z = (-1.2
             - 0.05 * usage_days
             + 0.45 * support_calls
             + 0.40 * plan_changes
             - 0.020 * tenure
             + 0.00004 * monthly_fee
             - (0.7 if is_auto_pay else 0.0))
        p = 1 / (1 + math.exp(-z))
        churned = 1 if rng.random() < p else 0

        rows.append({
            "customer_id": f"C{100000 + i}",
            "tenure_months": tenure,
            "monthly_fee": monthly_fee,
            "usage_days_30d": usage_days,
            "support_calls_30d": support_calls,
            "plan_changes": plan_changes,
            "auto_pay": int(is_auto_pay),
            "churned": churned,
        })
    return rows


# ---------------------------------------------------------------------------
# 3. 이상거래 데이터 — lecture07
# ---------------------------------------------------------------------------

def fraud_table(n: int = 5000, seed: int = 11) -> list[dict]:
    """카드 이상거래 데이터. 사기 비율 약 1.5% 로 심한 불균형을 만듭니다."""
    rng = _rng(seed)
    rows = []
    for i in range(n):
        is_fraud = rng.random() < 0.015
        if is_fraud:
            amount = rng.lognormvariate(12.5, 1.1)     # 비정상적으로 큰 금액
            hour = rng.choice([0, 1, 2, 3, 4, 23])     # 새벽 시간대
            foreign = rng.random() < 0.55
            n_recent = rng.randint(5, 20)              # 짧은 시간 연속 결제
        else:
            amount = rng.lognormvariate(10.2, 0.9)
            hour = int(max(0, min(23, rng.gauss(14, 4))))
            foreign = rng.random() < 0.05
            n_recent = rng.randint(0, 4)
        rows.append({
            "tx_id": f"T{i:06d}",
            "amount": round(amount),
            "hour": hour,
            "is_foreign": int(foreign),
            "tx_count_1h": n_recent,
            "is_fraud": int(is_fraud),
        })
    return rows


# ---------------------------------------------------------------------------
# 4. 한국어 텍스트 코퍼스 — lecture10, 11, 12
# ---------------------------------------------------------------------------

REVIEW_POSITIVE = [
    "배송이 정말 빨라서 좋았습니다", "가격 대비 품질이 훌륭하네요",
    "재구매 의사 100% 있습니다", "포장이 꼼꼼해서 만족합니다",
    "생각보다 훨씬 튼튼하고 좋아요", "직원분이 친절하게 응대해 주셨어요",
    "사진이랑 똑같아서 마음에 듭니다", "매장 분위기가 아주 쾌적했습니다",
    "성능이 기대 이상이라 놀랐습니다", "설치가 간단해서 편했어요",
]
REVIEW_NEGATIVE = [
    "배송이 일주일이나 걸렸습니다", "가격에 비해 품질이 너무 떨어져요",
    "두 번은 사지 않을 것 같습니다", "포장이 찢어진 채로 도착했어요",
    "생각보다 약해서 금방 망가졌습니다", "문의를 남겼는데 답이 없네요",
    "사진과 색상이 전혀 다릅니다", "매장이 좁고 시끄러웠습니다",
    "성능이 설명과 달라 실망했습니다", "설명서가 불친절해서 헤맸어요",
]


def review_corpus(n: int = 600, seed: int = 3) -> list[dict]:
    """감성 분류용 한국어 리뷰 데이터. label 1=긍정, 0=부정."""
    rng = _rng(seed)
    fillers = ["", " 다만 아쉬운 점도 조금 있어요.", " 다음에도 이용할게요.",
               " 참고하시면 좋겠습니다.", " 별점은 솔직하게 드립니다."]
    rows = []
    for i in range(n):
        label = i % 2
        pool = REVIEW_POSITIVE if label == 1 else REVIEW_NEGATIVE
        text = rng.choice(pool) + rng.choice(fillers)
        rows.append({"id": i, "text": text, "label": label})
    rng.shuffle(rows)
    return rows


SAMPLE_DOCS = {
    "사내규정_휴가.txt": (
        "제1조 연차휴가는 입사일 기준 1년 근속 시 15일이 부여된다. "
        "3년 이상 근속자는 2년마다 1일씩 가산되며 최대 25일을 넘지 않는다. "
        "연차 사용은 최소 3영업일 전에 결재 시스템으로 신청해야 한다. "
        "미사용 연차는 회계연도 종료 후 수당으로 정산한다."
    ),
    "사내규정_경비.txt": (
        "제2조 출장 경비는 사전 승인을 받은 건에 한하여 정산한다. "
        "국내 출장 일비는 3만원, 해외 출장 일비는 8만원을 기준으로 한다. "
        "영수증은 출장 종료 후 7일 이내에 제출해야 하며, "
        "제출 기한을 넘긴 경비는 원칙적으로 정산되지 않는다."
    ),
    "사내규정_재택.txt": (
        "제3조 재택근무는 주 2회까지 허용되며 팀장 승인이 필요하다. "
        "재택근무일에도 코어타임인 오전 10시부터 오후 4시까지는 연락이 가능해야 한다. "
        "회사 자산을 사외로 반출할 경우 정보보안팀에 사전 신고한다."
    ),
    "제품매뉴얼_설치.txt": (
        "본 제품을 설치하기 전에 전원을 반드시 차단하십시오. "
        "벽면과 최소 10센티미터 이상 간격을 두어 통풍을 확보해야 합니다. "
        "초기 설정은 전원을 켠 뒤 화면의 안내를 따라 5분 이내에 완료됩니다. "
        "설치 후 이상 소음이 들리면 즉시 사용을 중단하고 고객센터에 문의하십시오."
    ),
    "제품매뉴얼_보증.txt": (
        "제품 보증 기간은 구매일로부터 2년입니다. "
        "소비자 과실로 인한 파손, 침수, 낙하는 보증 대상에서 제외됩니다. "
        "보증 수리를 받으려면 구매 영수증 또는 주문번호가 필요합니다. "
        "소모품은 보증 기간과 무관하게 유상 교체됩니다."
    ),
}


def tiny_corpus(seed: int = 5) -> str:
    """
    미니 언어모델 사전학습용 한국어 텍스트 (lecture12).
    문법 패턴이 반복되어 작은 모델도 학습 신호를 잡을 수 있게 설계했습니다.
    """
    rng = _rng(seed)
    subjects = ["학생이", "회사원이", "요리사가", "개발자가", "선생님이"]
    objects = ["보고서를", "김치찌개를", "프로그램을", "편지를", "계획서를"]
    verbs = ["만들었다", "고쳤다", "확인했다", "정리했다", "준비했다"]
    times = ["어제", "오늘", "아침에", "저녁에", "주말에"]
    lines = []
    for _ in range(2000):
        lines.append(f"{rng.choice(times)} {rng.choice(subjects)} "
                     f"{rng.choice(objects)} {rng.choice(verbs)}.")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# 5. 이미지 데이터 — lecture09
# ---------------------------------------------------------------------------

def shape_images(n: int = 800, size: int = 16, seed: int = 13):
    """
    도형 분류용 흑백 이미지. 0=사각형, 1=원, 2=삼각형.
    MNIST 를 내려받지 않고도 CNN 실습을 할 수 있게 직접 그립니다.
    numpy 가 필요합니다. 반환: (X, y) — X 는 (n, size, size) float32 배열.
    """
    if _np is None:
        raise ImportError("shape_images() 는 numpy 가 필요합니다. pip install numpy")
    rng = _np.random.default_rng(seed)
    X = _np.zeros((n, size, size), dtype="float32")
    y = _np.zeros(n, dtype="int64")
    for i in range(n):
        label = i % 3
        y[i] = label
        img = _np.zeros((size, size), dtype="float32")
        cy, cx = rng.integers(5, size - 5, size=2)
        r = int(rng.integers(3, 5))
        yy, xx = _np.mgrid[0:size, 0:size]
        if label == 0:                                    # 사각형
            img[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 1.0
        elif label == 1:                                  # 원
            img[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 1.0
        else:                                             # 삼각형
            mask = (yy >= cy - r) & (yy <= cy + r) & (_np.abs(xx - cx) <= (yy - cy + r) / 2)
            img[mask] = 1.0
        img += rng.normal(0, 0.08, img.shape).astype("float32")   # 잡음
        X[i] = _np.clip(img, 0.0, 1.0)
    idx = rng.permutation(n)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# 6. 관계형 DB 스키마 — lecture04
# ---------------------------------------------------------------------------

def build_sqlite(path: str = "hjh_shop.db", seed: int = 21) -> str:
    """
    SQL 실습용 SQLite 데이터베이스를 만듭니다.
    테이블: customers, products, orders, order_items, employees
    파이썬 표준 라이브러리만 사용하므로 DB 설치가 필요 없습니다.
    """
    import os
    import sqlite3

    if os.path.exists(path):
        os.remove(path)
    rng = _rng(seed)
    con = sqlite3.connect(path)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            city TEXT, grade TEXT, joined_at TEXT);
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            category TEXT, price INTEGER, cost INTEGER);
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            dept TEXT, salary INTEGER, manager_id INTEGER);
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY, customer_id INTEGER,
            employee_id INTEGER, ordered_at TEXT, status TEXT);
        CREATE TABLE order_items (
            order_id INTEGER, product_id INTEGER, quantity INTEGER,
            PRIMARY KEY (order_id, product_id));
    """)

    cities = ["서울", "부산", "대구", "인천", "광주", "대전"]
    grades = ["VIP", "GOLD", "SILVER", "BASIC"]
    surnames = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
    givens = ["민준", "서연", "도윤", "지우", "하준", "서윤", "지호", "예린", "현우", "수아"]

    customers = [(i, rng.choice(surnames) + rng.choice(givens), rng.choice(cities),
                  rng.choice(grades), f"202{rng.randint(0, 5)}-{rng.randint(1, 12):02d}-15")
                 for i in range(1, 201)]
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    pnames = [("노트북", "전자"), ("무선마우스", "전자"), ("기계식키보드", "전자"),
              ("모니터", "전자"), ("원두커피", "식품"), ("초콜릿", "식품"),
              ("텀블러", "생활"), ("노트", "문구"), ("만년필", "문구"), ("의자", "가구")]
    products = []
    for pid, (nm, cat) in enumerate(pnames, start=1):
        price = rng.choice([4900, 12900, 29000, 89000, 350000, 1200000])
        products.append((pid, nm, cat, price, int(price * rng.uniform(0.45, 0.75))))
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    depts = ["영업", "마케팅", "개발", "CS"]
    employees = [(1, "한지민", "영업", 9000, None)]
    for eid in range(2, 21):
        employees.append((eid, rng.choice(surnames) + rng.choice(givens),
                          rng.choice(depts), rng.randint(3200, 8500),
                          1 if eid <= 5 else rng.randint(2, 5)))
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)

    orders, items = [], []
    for oid in range(1, 1001):
        orders.append((oid, rng.randint(1, 200), rng.randint(1, 20),
                       f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                       rng.choices(["완료", "취소", "배송중"], weights=[8, 1, 2])[0]))
        for pid in rng.sample(range(1, 11), rng.randint(1, 3)):
            items.append((oid, pid, rng.randint(1, 5)))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    cur.executemany("INSERT INTO order_items VALUES (?,?,?)", items)

    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    print("=== hjh_data 자가 점검 ===\n")
    print("[1] 매출 데이터"); head(sales_table(n_days=10), 3)
    print("\n[2] 이탈 데이터"); head(churn_table(200), 3)
    print("\n[3] 이상거래 데이터"); head(fraud_table(500), 3)
    print("\n[4] 리뷰 코퍼스"); head(review_corpus(20), 3)
    print(f"\n[5] 문서 {len(SAMPLE_DOCS)}건, 미니 코퍼스 {len(tiny_corpus()):,}자")
    print(f"\n[6] SQLite 생성 -> {build_sqlite('/tmp/hjh_check.db')}")
    print("\n모든 생성기 정상 동작.")
