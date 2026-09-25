"""
Lecture 01 / Level 08 — 셸 스크립트로 반복 작업 자동화

어질러진 공유 폴더(임시 폴더에 재현)를 자동으로 정리하는 실습입니다.
자동화의 표준 패턴 '수집 -> 판단 -> (드라이런) -> 실행 -> 보고'를 따라,
정규표현식으로 파일명을 해석해 지점·연월별 폴더로 일괄 이동하고
파일명을 통일된 규칙으로 바꿉니다.
"""

import random
import re
import tempfile
from pathlib import Path

# 어질러진 공유 폴더를 재현할 파일들 — 이름 규칙이 제각각!
MESSY_FILES = [
    "매출보고_강남_2024-03.csv", "매출보고_강남_2024-04.csv",
    "매출보고_서초_2024-03.csv", "sales_busan_2024-03.csv",
    "sales_busan_2024-04.csv", "매출보고_판교_2024-04.csv",
    "sales_pangyo_2024-03.csv", "매출보고_서초_2024-04.csv",
    "메모.txt", "점심투표.txt", "발표자료.pptx", "옛날백업.zip",
    "매출보고_강남_2024-05.csv", "sales_busan_2024-05.csv",
]

# 파일명 해석 규칙: '지점'과 '연-월'을 뽑아내는 정규표현식 두 가지
PATTERNS = [
    re.compile(r"매출보고_(?P<store>.+)_(?P<ym>\d{4}-\d{2})\.csv"),
    re.compile(r"sales_(?P<store>.+)_(?P<ym>\d{4}-\d{2})\.csv"),
]
# 영문 지점명 -> 한글 통일
STORE_KO = {"busan": "부산", "pangyo": "판교"}


def create_mess(folder: Path):
    """[1] 난장판 생성: 제각각인 파일들을 seed 고정 난수 내용으로 만듭니다."""
    rng = random.Random(42)  # seed 고정 — 몇 번을 실행해도 같은 내용
    for name in MESSY_FILES:
        revenue = rng.randint(500, 2000)
        (folder / name).write_text(f"revenue,{revenue}000\n", encoding="utf-8")


def plan_moves(folder: Path):
    """[2] 수집·판단: 파일마다 '어디로, 무슨 이름으로' 보낼지 계획만 세웁니다.

    실행과 분리해 두면 드라이런(예행연습)이 공짜로 생깁니다."""
    plans = []  # (원본 Path, 목적지 Path, 분류 사유)
    for path in sorted(folder.glob("*")):        # 수집: 대상 목록
        if path.is_dir():
            continue
        for pattern in PATTERNS:                 # 판단: 규칙 적용
            matched = pattern.match(path.name)
            if matched:
                store = STORE_KO.get(matched["store"], matched["store"])
                ym = matched["ym"]
                # 목적지: 지점폴더/지점_연-월.csv 로 이름까지 통일
                dest = folder / store / f"{store}_{ym}.csv"
                plans.append((path, dest, f"매출 파일 -> {store}/{ym}"))
                break
        else:  # 어떤 규칙에도 안 맞으면 사람이 볼 검토함으로
            dest = folder / "_검토필요" / path.name
            plans.append((path, dest, "규칙 불일치 -> 검토함"))
    return plans


def execute_moves(plans):
    """[4] 실행: 계획표대로 폴더를 만들고 일괄 이동합니다."""
    moved = 0
    for src, dest, _ in plans:
        dest.parent.mkdir(parents=True, exist_ok=True)  # 셸의 mkdir -p
        if dest.exists():                                # 덮어쓰기 사고 방지!
            dest = dest.with_name(dest.stem + "_중복" + dest.suffix)
        src.rename(dest)                                 # 셸의 mv
        moved += 1
    return moved


def draw_tree(folder: Path):
    """정리 결과를 트리로 출력합니다."""
    for p in sorted(folder.rglob("*")):
        depth = len(p.relative_to(folder).parts) - 1
        tag = "/" if p.is_dir() else ""
        print("      " + "  " * depth + f"- {p.name}{tag}")


def main():
    print("=" * 60)
    print("파일 정리 자동화 — 수집 -> 판단 -> 드라이런 -> 실행 -> 보고")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        shared = Path(tmp) / "shared_folder"
        shared.mkdir()

        create_mess(shared)
        print(f"\n[1] 난장판 재현: 이름 규칙이 제각각인 파일 {len(MESSY_FILES)}개 생성")
        print("    " + ", ".join(MESSY_FILES[:6]) + " ...")

        plans = plan_moves(shared)
        print(f"\n[2] 수집·판단: 정규표현식으로 파일명을 해석해 계획 {len(plans)}건 수립")

        # [3] 드라이런: 실행 전에 계획표만 눈으로 검증 — 자동화의 제1 안전장치
        print("\n[3] 드라이런(예행연습) — 아직 아무것도 옮기지 않았습니다")
        for src, dest, reason in plans:
            print(f"    {src.name:<28} -> {dest.relative_to(shared)}  ({reason})")

        moved = execute_moves(plans)
        print(f"\n[4] 실제 실행: {moved}건 이동·이름 통일 완료")

        print("\n[5] 결과 보고 — 정리 후 폴더 구조:")
        draw_tree(shared)
        review = sum(1 for _, dest, _ in plans if "_검토필요" in str(dest))
        print(f"\n    요약: 자동 분류 {moved - review}건 / 사람 검토 필요 {review}건")
        print("    손으로 했다면 파일당 30초 x 14개 = 약 7분,")
        print("    스크립트는 0.1초 — 그리고 매주 다시 시켜도 공짜입니다.")

    print("\n정리: 계획(plan)과 실행(execute)을 분리하면 드라이런이 공짜로 생깁니다.")
    print("      일괄 작업은 반드시 '계획표 검증 -> 실행' 순서로!")


if __name__ == "__main__":
    main()
