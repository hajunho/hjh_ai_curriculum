"""모듈과 패키지 — 내 모듈(utils.py) import 와 표준 라이브러리 투어.

같은 폴더의 utils.py(공용 함수 모듈)를 import 해 쓰는 법과,
파이썬 기본 제공 '전문 부서' 4곳 — datetime / math / random / collections —
의 대표 기능을 실무 예제로 둘러봅니다.
"""

import datetime
import math
import random
from collections import Counter, defaultdict

import utils                     # 같은 폴더의 utils.py — 파일 하나가 모듈 하나


def main():
    print("=" * 56)
    print(" 모듈과 패키지 — import 와 표준 라이브러리 투어")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 내 모듈 import: utils.py 의 공용 함수 사용
    # ---------------------------------------------------------
    print("\n[1] 내 모듈 utils 사용 (모듈.함수 = 부서.담당자)")

    monthly_sales = [1200000, 980000, 1450000, 730000, 2100000]
    total = sum(monthly_sales)
    vat = utils.calc_vat(total)

    print(f"  5개 지점 매출 합계 : {utils.format_krw(total)}")
    print(f"  부가세(utils.calc_vat): {utils.format_krw(vat)}")
    print(f"  모듈 상수 접근      : utils.VAT_RATE = {utils.VAT_RATE}")
    print("  -> utils.py 의 자체 데모는 실행되지 않았습니다 (__name__ 구분 덕분).")

    # ---------------------------------------------------------
    # [2] datetime 부서: 날짜 계산
    # ---------------------------------------------------------
    print("\n[2] datetime — 날짜·시간 계산")

    base_day = datetime.date(2026, 9, 26)          # 기준일 고정(출력 재현성)
    deadline = datetime.date(2026, 12, 31)
    d_day = (deadline - base_day).days             # 날짜끼리 빼면 기간이 나온다

    print(f"  기준일        : {base_day} ({['월','화','수','목','금','토','일'][base_day.weekday()]}요일)")
    print(f"  연말 마감까지 : D-{d_day}")
    print(f"  서식화        : {base_day.strftime('%Y년 %m월 %d일')}")
    ship_day = utils.add_business_days(base_day, 3)
    print(f"  영업일 3일 뒤 납기(주말 제외, utils 함수): {ship_day}")

    # ---------------------------------------------------------
    # [3] math 부서: 올림이 필요한 순간
    # ---------------------------------------------------------
    print("\n[3] math — 배차 계획에 올림(ceil)이 필요한 이유")

    people = 17
    van_capacity = 5
    exact = people / van_capacity
    vans = math.ceil(exact)                        # 3.4대는 없다. 4대가 필요하다.
    print(f"  17명 / 5인승 = {exact}  ->  math.ceil() = {vans}대 배차")
    print(f"  math.floor(3.4) = {math.floor(3.4)} / math.sqrt(2) = {math.sqrt(2):.4f}"
          f" / math.pi = {math.pi:.4f}")

    # ---------------------------------------------------------
    # [4] random 부서: 추첨과 셔플 (seed 고정으로 재현성 확보)
    # ---------------------------------------------------------
    print("\n[4] random — 경품 추첨과 근무 순번 (seed=42 고정)")

    random.seed(42)                                # seed 고정: 실행할 때마다 같은 결과
    staff = ["김주임", "이과장", "박대리", "최부장", "정사원", "한대리"]

    winners = random.sample(staff, 2)              # 중복 없이 2명 추첨
    print(f"  경품 당첨자 2명 : {winners}")

    rotation = staff.copy()
    random.shuffle(rotation)                       # 원본 보존을 위해 copy 후 셔플
    print(f"  당직 순번       : {rotation}")
    print(f"  주사위 한 번    : {random.randint(1, 6)}")

    # ---------------------------------------------------------
    # [5] collections 부서: Counter 와 defaultdict
    # ---------------------------------------------------------
    print("\n[5] collections — 집계 전문 도구")

    orders = ["아메리카노", "라떼", "아메리카노", "티", "라떼", "아메리카노", "티", "아메리카노"]
    counter = Counter(orders)                      # 빈도 집계가 한 줄
    print(f"  주문 빈도 Counter : {dict(counter)}")
    print(f"  최다 주문 1위     : {counter.most_common(1)[0][0]} ({counter.most_common(1)[0][1]}건)")

    branch_sales = [("강남점", 120), ("서초점", 80), ("강남점", 200), ("판교점", 150), ("서초점", 90)]
    by_branch = defaultdict(int)                   # 없는 키는 자동으로 0에서 시작
    for branch, amount in branch_sales:
        by_branch[branch] += amount                # get(키, 0) 없이 바로 누적
    print(f"  지점별 매출 defaultdict: {dict(by_branch)}")

    print("\n[끝] 필요한 기능은 먼저 표준 라이브러리에서 찾고,")
    print("     우리 팀 공용 절차는 utils 같은 모듈로 묶어 재사용하세요.")


if __name__ == "__main__":
    main()
