"""우리 팀 공용 유틸리티 모듈.

여러 스크립트에서 반복해 쓰는 절차(통화 서식, 부가세, 영업일 계산)를
한 파일에 모아 둔 '업무 매뉴얼 바인더'입니다.
main.py 에서 `import utils` 로 불러 씁니다.
"""

import datetime

VAT_RATE = 0.1          # 모듈 수준 상수: import 한 쪽에서 utils.VAT_RATE 로 접근 가능

# import 되는 순간 모듈 코드는 위에서 아래로 '한 번' 실행된다.
# 이 print 는 그 사실을 눈으로 확인하기 위한 교육용 표시이다.
print("  (utils 모듈이 로딩되었습니다 — 이 줄은 import 시 딱 한 번 실행)")


def format_krw(amount):
    """금액을 '1,234,567원' 형태의 한국 원화 문자열로 서식화한다."""
    return f"{amount:,}원"


def calc_vat(amount):
    """부가세(10%)를 원 단위 정수로 계산한다."""
    return int(amount * VAT_RATE)


def add_business_days(start_date, days):
    """주말(토·일)을 건너뛰고 영업일 기준으로 days 일 뒤 날짜를 반환한다."""
    current = start_date
    remaining = days
    while remaining > 0:
        current += datetime.timedelta(days=1)
        if current.weekday() < 5:        # 0=월 ... 4=금 / 5=토, 6=일은 제외
            remaining -= 1
    return current


if __name__ == "__main__":
    # 이 블록은 `python3 utils.py` 로 '직접 실행'할 때만 동작한다.
    # main.py 가 import 할 때는 실행되지 않는다 — 정의와 실행의 분리.
    print("[utils 자체 데모] 직접 실행할 때만 보이는 출력입니다")
    print(f"  format_krw(1234567) = {format_krw(1234567)}")
    print(f"  calc_vat(50000)     = {calc_vat(50000)}")
    demo_day = datetime.date(2026, 9, 25)   # 금요일
    print(f"  {demo_day}(금) + 영업일 3일 = {add_business_days(demo_day, 3)}")
