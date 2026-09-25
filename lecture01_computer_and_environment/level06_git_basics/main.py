"""
Lecture 01 / Level 06 — Git: 버전 관리의 개념

임시 폴더에 진짜 Git 저장소를 만들어 init -> add -> commit -> diff ->
log -> branch 흐름을 자동 실습합니다. 커밋=결재 도장, 브랜치=평행우주라는
비유를 실제 명령 출력으로 확인합니다.
git 이 설치되어 있지 않으면 설치 안내 후 개념 시뮬레이션으로 대체합니다.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path


def run_git(args, cwd, note=""):
    """git 명령을 실행하고 '$ 명령 -> 출력' 형태로 보여줍니다."""
    print(f"\n  $ git {' '.join(args)}")
    if note:
        print(f"    해설: {note}")
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines()[:12]:  # 너무 길면 앞부분만
        print(f"    | {line}")
    return output


def real_git_demo():
    """git 이 설치된 경우: 임시 폴더에서 실제 명령으로 전체 흐름 실습."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "cafe_project"
        repo.mkdir()
        price_file = repo / "가격표.txt"

        # [1] 서류함 설치: init + 실습용 신원 등록(이 저장소 안에서만 유효)
        print("\n[1] 서류함 설치 — git init")
        run_git(["init", "-b", "main"], repo, "이 폴더의 이력 관리를 시작(.git 서류함 생성)")
        run_git(["config", "user.name", "실습생"], repo, "도장에 새길 이름(실습용, 이 저장소 한정)")
        run_git(["config", "user.email", "student@example.com"], repo, "도장에 새길 메일")

        # [2] 첫 결재: 파일 생성 -> add(결재판) -> commit(도장)
        print("\n[2] 첫 결재 — add 와 commit")
        price_file.write_text("아메리카노 4000원\n라떼 4500원\n", encoding="utf-8")
        print(f"    (파일 생성: {price_file.name})")
        run_git(["status", "--short"], repo, "?? = 아직 서류함이 모르는 새 파일")
        run_git(["add", "가격표.txt"], repo, "결재판(스테이징)에 올림")
        run_git(["commit", "-m", "가격표 초안 작성"], repo, "도장 쾅! 이 순간이 영구 보존됨")

        # [3] 수정과 diff: 무엇이 달라졌나 확인 후 두 번째 커밋
        print("\n[3] 수정과 diff — 달라진 부분 확인")
        price_file.write_text("아메리카노 4200원\n라떼 4500원\n", encoding="utf-8")
        print("    (아메리카노 가격을 4000 -> 4200 으로 수정)")
        run_git(["diff"], repo, "- 가 옛 내용, + 가 새 내용")
        run_git(["add", "."], repo)
        run_git(["commit", "-m", "아메리카노 가격 인상(4000->4200)"], repo, "메시지에 '왜'를 적는 것이 요령")

        # [4] 이력 조회: 도장이 쌓인 결재 기록
        print("\n[4] 이력 조회 — git log")
        run_git(["log", "--oneline"], repo, "앞의 짧은 코드가 커밋 일련번호(해시)")

        # [5] 브랜치: 평행우주에서 실험 후 본편으로 복귀
        print("\n[5] 브랜치 — 평행우주 실험")
        run_git(["switch", "-c", "experiment"], repo, "experiment 우주를 만들어 이동")
        price_file.write_text("아메리카노 9900원 (실험!)\n라떼 4500원\n", encoding="utf-8")
        run_git(["add", "."], repo)
        run_git(["commit", "-m", "실험: 프리미엄 가격 정책"], repo)
        print(f"    experiment 우주의 가격표: {price_file.read_text(encoding='utf-8').splitlines()[0]}")
        run_git(["switch", "main"], repo, "본편(main)으로 복귀 — 시간여행!")
        print(f"    main 우주의 가격표    : {price_file.read_text(encoding='utf-8').splitlines()[0]}")
        print("    -> 같은 파일인데 브랜치를 바꾸니 내용이 되돌아왔습니다.")
        print("       실험이 실패하면 experiment 우주만 버리면 본편은 무사합니다.")


def fallback_simulation():
    """git 이 없는 경우: 설치 안내 + 커밋 개념 시뮬레이션."""
    print("\n[안내] git 명령을 찾지 못했습니다.")
    print("  설치: 맥은 'xcode-select --install' 또는 'brew install git',")
    print("        윈도우는 git-scm.com 의 Git for Windows 설치를 권합니다.")
    print("  설치 후 이 스크립트를 다시 실행하면 실제 명령으로 실습됩니다.")
    print("\n[개념 시뮬레이션] 커밋 = 폴더 상태의 스냅샷 + 결재 도장")
    history = [
        ("a1f9c02", "가격표 초안 작성", {"가격표.txt": "아메리카노 4000원"}),
        ("b7e3d11", "아메리카노 가격 인상(4000->4200)", {"가격표.txt": "아메리카노 4200원"}),
    ]
    for i, (commit_hash, message, snapshot) in enumerate(history, start=1):
        print(f"\n  커밋 {i}: [{commit_hash}] \"{message}\"")
        for fname, content in snapshot.items():
            print(f"    보관된 스냅샷: {fname} -> '{content}'")
    print("\n  어느 도장 시점으로든 폴더를 되돌릴 수 있는 것이 Git 의 힘입니다.")


def main():
    print("=" * 60)
    print("Git 실습 — 결재 도장(커밋)과 평행우주(브랜치)")
    print("=" * 60)
    if shutil.which("git"):
        print("\n(git 발견 — 임시 폴더에 진짜 저장소를 만들어 실습합니다.")
        print(" 여러분의 다른 폴더·저장소는 전혀 건드리지 않습니다.)")
        real_git_demo()
    else:
        fallback_simulation()

    print("\n정리: 작게, 자주 커밋하고, 메시지에는 '왜'를 적으세요.")
    print("      커밋해 둔 것은 어떤 실수를 해도 되찾을 수 있습니다.")


if __name__ == "__main__":
    main()
