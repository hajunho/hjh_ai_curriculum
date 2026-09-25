"""
Lecture 01 / Level 01 — 파일·폴더·경로의 개념

pathlib 으로 가상 회사 '한빛상사'의 문서 폴더 트리를 만들고 탐색합니다.
절대경로/상대경로의 차이, 확장자별 파일 집계, 경로 분해(parent/stem/suffix)를
임시 폴더 안에서 안전하게 실습합니다. 종료 시 임시 폴더는 자동 삭제됩니다.
"""

import os
import tempfile
from collections import Counter
from pathlib import Path


def build_company_tree(root: Path):
    """[트리 생성] 부서 폴더와 샘플 문서 파일들을 만듭니다."""
    # 폴더 구조: (상대경로) — mkdir(parents=True)로 중간 폴더까지 한 번에 생성
    folders = [
        "총무팀",
        "영업팀/계약서",
        "영업팀/보고서",
        "개발팀/코드",
    ]
    for name in folders:
        (root / name).mkdir(parents=True, exist_ok=True)

    # 파일: (상대경로, 내용) — 확장자는 '문서 종류 도장'입니다.
    files = [
        ("총무팀/비품신청서.txt", "볼펜 10자루, A4 용지 2박스"),
        ("총무팀/주차규정.txt", "지하 2층은 방문객 전용입니다."),
        ("영업팀/계약서/A사_계약서.txt", "갑: A사 / 을: 한빛상사"),
        ("영업팀/계약서/B사_계약서.txt", "갑: B사 / 을: 한빛상사"),
        ("영업팀/보고서/3월_실적.csv", "store,revenue\n강남,1512000\n서초,1098000"),
        ("영업팀/보고서/4월_실적.csv", "store,revenue\n강남,1620000\n서초,1150000"),
        ("개발팀/코드/hello.py", "print('hello')"),
    ]
    for rel_path, content in files:
        (root / rel_path).write_text(content, encoding="utf-8")
    return len(folders), len(files)


def draw_tree(folder: Path, indent: int = 0):
    """[트리 그리기] 재귀 호출로 폴더 구조를 들여쓰기 그림으로 출력합니다."""
    marker = "📁" if indent == 0 else "└─"
    print("    " + "   " * indent + f"{marker} {folder.name}/")
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            draw_tree(child, indent + 1)  # 폴더면 자기 자신을 다시 호출 (재귀)
        else:
            print("    " + "   " * (indent + 1) + f"└─ {child.name}")


def main():
    print("=" * 60)
    print("파일·폴더·경로 실습 — 한빛상사 문서고 탐험")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "hanbit_corp"
        root.mkdir()

        # [1] 폴더 트리 생성
        n_folders, n_files = build_company_tree(root)
        print(f"\n[1] 트리 생성: 폴더 {n_folders}개(+중간 폴더), 파일 {n_files}개를 만들었습니다")
        print(f"    위치(임시 폴더): {root}")

        # [2] 트리 그리기 — 터미널의 tree 명령을 파이썬으로 흉내
        print("\n[2] 문서고 전체 구조:")
        draw_tree(root)

        # [3] 절대경로 vs 상대경로
        target = root / "영업팀" / "보고서" / "3월_실적.csv"
        print("\n[3] 같은 파일을 가리키는 두 가지 주소:")
        print(f"    절대경로: {target.resolve()}")
        print(f"    상대경로(회사 정문 기준): {target.relative_to(root)}")
        # '..' 은 한 층 위(부모 폴더)를 뜻합니다.
        sibling = target.parent.parent / "계약서" / "A사_계약서.txt"
        print(f"    보고서 폴더에서 '../계약서/A사_계약서.txt' 로 가면:")
        print(f"      -> {sibling.relative_to(root)} (존재? {sibling.exists()})")
        print(f"    현재 작업 디렉터리(CWD): {os.getcwd()}")
        print("      -> 상대경로는 항상 이 CWD(또는 명시한 기준)에서 해석됩니다")

        # [4] 확장자별 파일 집계 — rglob 은 하위 폴더까지 전부 뒤집니다
        counts = Counter(p.suffix for p in root.rglob("*") if p.is_file())
        print("\n[4] 확장자(문서 종류 도장)별 집계:")
        for ext, count in sorted(counts.items()):
            print(f"    {ext:<6} {count}개")

        # [5] 경로 해부 — 경로 하나를 부위별로 분해
        print("\n[5] 경로 해부: 영업팀/보고서/3월_실적.csv")
        print(f"    parent (들어있는 폴더) : {target.parent.name}/")
        print(f"    name   (파일 전체 이름): {target.name}")
        print(f"    stem   (확장자 뺀 이름): {target.stem}")
        print(f"    suffix (확장자)        : {target.suffix}")

    print("\n[6] 실습 종료: 임시 폴더는 자동 삭제되었습니다 — 컴퓨터에 흔적이 없습니다")
    print("정리: 절대경로는 정문부터 전체 주소, 상대경로는 현재 위치 기준 약식 주소입니다.")


if __name__ == "__main__":
    main()
