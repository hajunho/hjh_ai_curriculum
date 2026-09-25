"""
Lecture 01 / Level 00 — 컴퓨터는 무엇을 하는 기계인가

컴퓨터의 일 처리 방식을 '입력 -> 계산 -> 출력' 흐름으로 체험합니다.
CPU(실무 담당자), 메모리(책상), 저장장치(문서 창고)의 역할 분담을
카페 본사의 매출 집계 업무에 빗대어 확인합니다.
표준 라이브러리만 사용하며, 인터넷 연결이 필요 없습니다.
"""

import json
import tempfile
import time
from pathlib import Path

# 반복 계산 횟수 (직접 해보기 1번 과제에서 이 값을 바꿔 보세요)
LOOP_COUNT = 1_000_000


def receive_orders():
    """[입력] 접수창구: 지점별 주문 데이터가 들어옵니다."""
    # (지점명, 주문 건수, 총 매출액[원]) — 코드 안에서 만든 가상 데이터입니다.
    orders = [
        ("강남점", 182, 1_512_000),
        ("서초점", 141, 1_098_000),
        ("판교점", 210, 1_745_000),
        ("부산점", 95, 702_000),
    ]
    return orders


def save_to_storage(orders, path):
    """[저장장치] 문서 창고에 보관: 파일로 저장하면 전원이 꺼져도 남습니다."""
    rows = [{"store": s, "count": c, "revenue": r} for s, c, r in orders]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.stat().st_size  # 창고에 보관된 서류의 크기(바이트)


def load_from_storage(path):
    """[메모리] 창고의 서류를 책상 위로: 파일을 읽어 메모리에 올립니다."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(row["store"], row["count"], row["revenue"]) for row in rows]


def process_orders(orders):
    """[계산] CPU 담당자: 입력을 받아 계산하고 결과를 돌려줍니다. (IPO 모델)"""
    total_revenue = sum(revenue for _, _, revenue in orders)
    total_count = sum(count for _, count, _ in orders)
    avg_per_order = total_revenue / total_count
    best_store = max(orders, key=lambda row: row[2])
    return {
        "total_revenue": total_revenue,
        "total_count": total_count,
        "avg_per_order": avg_per_order,
        "best_store": best_store[0],
        "best_revenue": best_store[2],
    }


def measure_cpu_speed():
    """CPU 가 단순 계산을 얼마나 빨리 반복하는지 시간을 재봅니다."""
    started = time.perf_counter()
    acc = 0
    for i in range(LOOP_COUNT):
        acc += i  # 아주 작은 덧셈을 계속 반복
    elapsed = time.perf_counter() - started
    return elapsed, acc


def show_binary(text):
    """글자가 컴퓨터 내부에서 0과 1로 어떻게 표현되는지 보여줍니다."""
    for ch in text:
        code = ord(ch)  # 글자마다 약속된 번호(유니코드)
        print(f"    글자 '{ch}' -> 번호 {code} -> 이진법 {code:08b}")


def main():
    print("=" * 60)
    print("컴퓨터 회사(주) 업무 흐름 체험 — 입력 -> 계산 -> 출력")
    print("=" * 60)

    # [1] 입력: 접수창구로 주문 데이터가 들어옵니다.
    orders = receive_orders()
    print(f"\n[1] 입력(Input): 접수창구에 {len(orders)}개 지점의 주문이 도착")
    for store, count, revenue in orders:
        print(f"    - {store}: {count}건, {revenue:,}원")

    # [2] 저장장치 <-> 메모리: 창고에 보관했다가 다시 책상 위로 꺼냅니다.
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "orders_archive.json"
        size = save_to_storage(orders, archive)
        print(f"\n[2] 저장장치(창고): '{archive.name}' 파일로 보관 ({size}바이트)")
        print("    - 파일로 저장한 내용은 전원을 꺼도 남습니다 (비휘발성)")
        orders_on_desk = load_from_storage(archive)
        print(f"    - 창고에서 다시 꺼내 책상(메모리)에 올림: {len(orders_on_desk)}건 복원")
        print("    - 메모리 위 데이터는 프로그램이 끝나면 사라집니다 (휘발성)")

    # [3] 계산: CPU 담당자가 합계와 평균을 냅니다.
    report = process_orders(orders_on_desk)
    elapsed, _ = measure_cpu_speed()
    print(f"\n[3] 계산(Process): CPU 담당자의 결재 속도 체험")
    print(f"    - 단순 덧셈 {LOOP_COUNT:,}번 반복에 걸린 시간: {elapsed:.3f}초")
    print(f"    - 초당 약 {LOOP_COUNT / elapsed:,.0f}번 꼴 — 사람은 흉내도 못 낼 속도입니다")

    # [4] 출력: 사람이 읽기 좋은 보고서로 발송합니다.
    print(f"\n[4] 출력(Output): 매출 집계 보고서")
    print(f"    - 총 매출          : {report['total_revenue']:,}원")
    print(f"    - 총 주문 건수      : {report['total_count']:,}건")
    print(f"    - 주문당 평균 매출  : {report['avg_per_order']:,.0f}원")
    print(f"    - 최고 매출 지점    : {report['best_store']} ({report['best_revenue']:,}원)")

    # [5] 이진법 맛보기: 모든 정보는 결국 0과 1입니다.
    print(f"\n[5] 이진법 맛보기: 'AI' 라는 글자의 실제 저장 모습")
    show_binary("AI")

    print("\n정리: 어떤 프로그램이든 '입력 -> 계산 -> 출력' 세 조각으로 나눠 보세요.")
    print("      CPU=담당자, 메모리=책상(빠름·휘발), 저장장치=창고(느림·영구)입니다.")


if __name__ == "__main__":
    main()
