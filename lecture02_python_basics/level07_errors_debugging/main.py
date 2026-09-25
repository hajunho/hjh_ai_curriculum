"""예외 처리와 디버깅 — 일부러 깨진 매출 데이터로 전/후 비교.

오염된 데이터(문자 금액, 빈 값, 숨은 공백, 열 이름 누락)를
[2] 보호 없이 처리해 죽는 모습과 [3] try/except 로 살아남는 모습을 비교합니다.
대표 예외 5종 관찰, print 디버깅(!r), raise 로 조기 경보까지 실습합니다.
"""


def validate_amount(amount):
    """[5]용 검증 함수: 음수 금액은 계약 위반으로 즉시 raise."""
    if amount < 0:
        raise ValueError(f"금액은 음수일 수 없습니다: {amount}")
    return amount


def main():
    print("=" * 56)
    print(" 예외 처리와 디버깅 — 깨진 데이터에서 살아남기")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 대표 예외 5종을 일부러 일으켜 '마지막 줄'을 읽어 본다
    # ---------------------------------------------------------
    print("\n[1] 대표 예외 관찰 (유형: 원인 메시지)")

    demos = [
        ("int('협의중')", lambda: int("협의중")),
        ("'100' + 5", lambda: "100" + 5),
        ("{'a':1}['금액']", lambda: {"a": 1}["금액"]),
        ("[][0]", lambda: [][0]),
        ("10 / 0", lambda: 10 / 0),
    ]
    for code_text, run in demos:
        try:
            run()
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as e:
            # traceback 의 마지막 줄에 해당하는 정보: 유형 이름 + 원인
            print(f"  {code_text:18s} -> {type(e).__name__}: {e}")
    print("  -> 에러 메시지는 아래에서 위로. 마지막 줄이 '유형: 원인'입니다.")

    # 오염된 월간 매출 데이터 (실무 파일에서 흔한 사고 유형을 모아 둠)
    dirty_rows = [
        {"store": "강남점", "amount": "1200000"},
        {"store": "서초점", "amount": "980000"},
        {"store": "판교점", "amount": "협의중"},      # 문자 금액 -> ValueError
        {"store": "분당점", "amount": "1450000"},
        {"store": "일산점", "amount": ""},            # 빈 값 -> ValueError
        {"store": "수원점", "amount": " 730000 "},    # 숨은 공백 (int 는 되지만 [4]에서 관찰)
        {"store": "인천점"},                           # amount 키 자체가 없음 -> KeyError
        {"store": "부산점", "amount": "2100000"},
        {"store": "대구점", "amount": "880000"},
        {"store": "광주점", "amount": "1010000"},
    ]

    # ---------------------------------------------------------
    # [2] 보호 없는 처리: 3번째 행에서 즉사한다
    # ---------------------------------------------------------
    print("\n[2] 예외 처리 없이 집계하면?")
    try:
        total = 0
        for i, row in enumerate(dirty_rows):
            total += int(row["amount"])           # 보호 없음!
        print(f"  합계: {total}")                  # 여기까지 못 온다
    except (ValueError, KeyError) as e:
        print(f"  {i}번째 행({dirty_rows[i]['store']})에서 사망 -> {type(e).__name__}: {e}")
        print(f"  -> 그때까지의 부분합 {total:,}원도 버려지고, 나머지 행도 처리 못 함.")

    # ---------------------------------------------------------
    # [3] 보호된 처리: 불량은 기록하고 건너뛰며 끝까지 간다
    # ---------------------------------------------------------
    print("\n[3] try/except 로 보호된 집계")

    total = 0
    ok_count = 0
    failures = []                                  # (지점, 사유) 기록
    for row in dirty_rows:
        try:
            amount = int(row["amount"].strip())    # 사고 가능 최소 구간만 try 안에
        except KeyError:
            failures.append((row["store"], "amount 열 없음"))
            continue
        except ValueError as e:
            failures.append((row["store"], f"금액 형식 오류({row['amount']!r})"))
            continue
        total += amount
        ok_count += 1

    print(f"  성공 {ok_count}건 / 실패 {len(failures)}건 / 합계 {total:,}원")
    print("  실패 내역 (조용히 pass 하지 않고 반드시 기록):")
    for store, reason in failures:
        print(f"    - {store}: {reason}")
    print("  -> 같은 데이터인데 [2]는 죽고 [3]은 운영 리포트까지 냅니다.")

    # ---------------------------------------------------------
    # [4] print 디버깅: 에러 없이 값만 이상할 때
    # ---------------------------------------------------------
    print("\n[4] print 디버깅 — 숨은 공백 잡기")

    raw = " 730000 "
    print(f"  print(raw)      -> {raw}    (겉보기엔 멀쩡)")
    print(f"  print(f'{{raw!r}}') -> {raw!r}  <- !r 로 찍으니 공백이 보인다!")
    print(f"  int(raw.strip()) = {int(raw.strip()):,}  (strip 후 변환이 안전)")
    print("  요령: 변수명과 함께, !r 로, 의심 구간에만 찍고, 해결 후 지운다.")

    # ---------------------------------------------------------
    # [5] raise: 잘못된 값은 일찍, 크게 알린다
    # ---------------------------------------------------------
    print("\n[5] raise — 음수 금액 조기 경보")
    try:
        validate_amount(150000)
        print("  validate_amount(150000)  -> 통과")
        validate_amount(-50000)
        print("  이 줄은 실행되지 않습니다")
    except ValueError as e:
        print(f"  validate_amount(-50000) -> ValueError: {e}")
        print("  -> 이상값을 조용히 통과시키는 것보다 초기에 거부하는 편이 안전합니다.")

    # ---------------------------------------------------------
    # finally 데모: 사고 여부와 무관한 뒷정리
    # ---------------------------------------------------------
    print("\n[6] finally — 무슨 일이 있어도 하는 뒷정리")
    try:
        risky = int("불량")
    except ValueError:
        print("  except: 불량 값 대처 완료")
    finally:
        print("  finally: (에러가 났어도) 파일 닫기·연결 정리 같은 문단속은 실행됩니다")

    print("\n[끝] 실무 데이터는 반드시 깨져 있습니다.")
    print("     불량 1건은 기록하고 건너뛰되, 전체 업무는 멈추지 않는 것이 예외 처리입니다.")


if __name__ == "__main__":
    main()
