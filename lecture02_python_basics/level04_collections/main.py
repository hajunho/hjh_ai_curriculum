"""리스트·튜플·딕셔너리·집합 — 문구 도매상 재고 관리 미니 예제.

네 가지 컬렉션(collection)을 재고 관리 업무에 대입해 봅니다.
리스트 = 입고 기록 대장 / 튜플 = 변경 불가 규격표 /
딕셔너리 = 상품명->수량 서랍장 / 집합 = 중복 없는 품목 명단.
마지막에 실무 표준 형태인 '딕셔너리들의 리스트'로 재고 표를 다룹니다.
"""


def main():
    print("=" * 56)
    print(" 컬렉션 — 문구 도매상 재고 관리")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 리스트: 순서 있는 입고 기록 대장
    # ---------------------------------------------------------
    print("\n[1] 리스트 — 오늘의 입고 수량 기록")

    inbound = [30, 12, 45]          # 오전 입고 3건
    inbound.append(20)              # 오후에 1건 추가
    print(f"  입고 기록      : {inbound}")
    print(f"  첫 건 inbound[0]  = {inbound[0]}  (인덱스는 0부터)")
    print(f"  마지막 inbound[-1] = {inbound[-1]}")
    print(f"  슬라이스 [1:3]  = {inbound[1:3]}  (3번 직전까지)")
    print(f"  건수 {len(inbound)} / 합계 {sum(inbound)} / 정렬 {sorted(inbound)}")

    # 복사 착각 주의: b = a 는 같은 대장에 이름표만 하나 더 붙인 것
    alias = inbound
    real_copy = inbound.copy()
    alias.append(99)
    print(f"  alias.append(99) 후 원본 -> {inbound}  (원본도 바뀜!)")
    print(f"  copy() 본은 안전       -> {real_copy}")

    # ---------------------------------------------------------
    # [2] 튜플: 바꾸면 안 되는 상품 규격표
    # ---------------------------------------------------------
    print("\n[2] 튜플 — 상품 규격표(코드, 이름, 단가)")

    product = ("P001", "볼펜", 1200)
    code, name, unit_price = product          # 언패킹
    print(f"  규격표: {product}")
    print(f"  언패킹 -> 코드 {code} / 이름 {name} / 단가 {unit_price:,}원")

    try:
        product[2] = 1500                     # 단가 몰래 수정 시도!
    except TypeError as e:
        print(f"  단가 수정 시도 -> TypeError: {e}")
        print("  -> '실수로도 못 바꾼다'는 것이 튜플의 안전장치입니다.")

    # ---------------------------------------------------------
    # [3] 딕셔너리: 상품명 -> 재고 수량 서랍장
    # ---------------------------------------------------------
    print("\n[3] 딕셔너리 — 재고 서랍장")

    stock = {"볼펜": 37, "A4용지": 12, "스테이플러": 4}
    print(f"  현재 재고        : {stock}")
    print(f"  stock['볼펜']    = {stock['볼펜']}개  (키로 즉시 조회)")

    stock["형광펜"] = 20                      # 신규 입고
    stock["A4용지"] += 30                     # 추가 입고
    stock["스테이플러"] -= 2                  # 출고
    print(f"  입출고 반영 후   : {stock}")

    # 없는 키를 안전하게 조회: get(키, 기본값)
    print(f"  stock.get('지우개', 0) = {stock.get('지우개', 0)}  (KeyError 없이 기본값)")
    print(f"  '지우개' in stock      = {'지우개' in stock}")

    print("  서랍장 전체 순회:")
    for item, qty in stock.items():
        print(f"    - {item:6s}: {qty:>3}개")

    # ---------------------------------------------------------
    # [4] 집합: 발주/입고 명단 대조
    # ---------------------------------------------------------
    print("\n[4] 집합 — 발주 vs 입고 대조")

    ordered = {"볼펜", "형광펜", "지우개", "테이프", "볼펜"}   # 중복은 자동 제거
    arrived = {"볼펜", "형광펜"}
    print(f"  발주 품목(중복 넣어도 하나만): {sorted(ordered)}")
    print(f"  입고 품목                    : {sorted(arrived)}")
    print(f"  미입고 = 발주 - 입고 (차집합): {sorted(ordered - arrived)}")
    print(f"  발주하고 입고도 됨 (교집합)  : {sorted(ordered & arrived)}")
    print(f"  이번 주 언급된 전 품목(합집합): {sorted(ordered | arrived)}")

    # ---------------------------------------------------------
    # [5] 종합: '딕셔너리들의 리스트' = 실무 표준 데이터 형태
    # ---------------------------------------------------------
    print("\n[5] 재고 표(딕셔너리들의 리스트)와 발주 리포트")

    SAFETY = 10       # 안전 재고 기준
    inventory = [
        {"code": "P001", "name": "볼펜",       "qty": 37, "unit_price": 1200},
        {"code": "P002", "name": "A4용지",     "qty": 42, "unit_price": 25000},
        {"code": "P003", "name": "스테이플러", "qty": 2,  "unit_price": 8900},
        {"code": "P004", "name": "형광펜",     "qty": 20, "unit_price": 1500},
        {"code": "P005", "name": "테이프",     "qty": 6,  "unit_price": 2300},
    ]

    total_value = 0
    shortage = []                             # 재고 부족 품목을 모을 리스트
    for row in inventory:                     # 표의 각 행(딕셔너리)에 대해
        value = row["qty"] * row["unit_price"]
        total_value += value
        flag = " <- 발주 필요!" if row["qty"] < SAFETY else ""
        print(f"  {row['code']} {row['name']:6s} {row['qty']:>3}개 x {row['unit_price']:>6,}원"
              f" = {value:>9,}원{flag}")
        if row["qty"] < SAFETY:
            shortage.append(row["name"])

    print(f"\n  재고 자산 총액: {total_value:,}원")
    print(f"  발주 필요 품목(안전재고 {SAFETY}개 미만): {shortage}")
    print("\n[끝] 그릇 선택 기준 — 기록은 리스트, 불변 묶음은 튜플,")
    print("     키 조회는 딕셔너리, 중복 제거·명단 대조는 집합.")


if __name__ == "__main__":
    main()
