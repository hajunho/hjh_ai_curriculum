"""
Lecture 01 / Level 07 — GitHub: 원격 저장소와 협업

인터넷 없이 원격 협업을 실습합니다. 임시 폴더에 bare 저장소(본사 문서고)를
만들어 '가짜 GitHub' 로 삼고, 개발자 A·B 가 clone -> push -> pull 로
서류를 주고받는 협업 시나리오를 자동 실행합니다.
git 이 없으면 설치 안내 후 개념 시뮬레이션으로 대체합니다.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path


def run(args, cwd, actor="", note=""):
    """git 명령을 실행하고 '누가($ 앞 이름) 무엇을 했는지' 출력합니다."""
    label = f"[{actor}] " if actor else ""
    print(f"\n  {label}$ git {' '.join(args)}")
    if note:
        print(f"      해설: {note}")
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines()[:8]:
        print(f"      | {line}")
    return output


def setup_identity(repo, name):
    """실습용 커밋 신원 등록 (이 저장소 안에서만 유효)."""
    run(["config", "user.name", name], repo)
    run(["config", "user.email", f"{name}@example.com"], repo)


def list_files(repo, actor):
    """지사 서류함(작업 폴더)의 파일 목록을 보여줍니다."""
    files = sorted(p.name for p in Path(repo).iterdir() if p.is_file())
    print(f"      [{actor}] 지사 폴더 내용: {files if files else '(비어 있음)'}")


def real_remote_demo():
    """bare 원격 + 두 지사(A·B)로 push/pull 협업 흐름 실습."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        hq = base / "headquarters.git"   # 본사 중앙 문서고 (bare)
        dev_a = base / "dev_a"           # 개발자 A 의 지사
        dev_b = base / "dev_b"           # 개발자 B 의 지사

        # [1] 본사 설립: bare 저장소 = 작업 책상 없이 이력만 보관 (GitHub 서버의 실제 형태)
        print("\n[1] 본사 설립 — bare 원격 저장소 만들기")
        run(["init", "--bare", "-b", "main", str(hq)], base,
            note="--bare: 작업 폴더 없는 중앙 문서고. 이것이 우리의 '가짜 GitHub'")

        # [2] 지사 개설: A 와 B 가 각각 clone
        print("\n[2] 지사 개설 — 개발자 A·B 가 clone")
        run(["clone", str(hq), str(dev_a)], base, "A", "본사 문서고를 통째로 복제(아직 비어 있음)")
        run(["clone", str(hq), str(dev_b)], base, "B", "B 도 자기 지사를 개설")
        setup_identity(dev_a, "dev-a")
        setup_identity(dev_b, "dev-b")

        # [3] A 의 push: 파일 생성 -> 커밋 -> 본사로 올려보내기
        print("\n[3] A 의 작업과 push")
        (dev_a / "menu.txt").write_text("아메리카노 4000원\n", encoding="utf-8")
        run(["add", "menu.txt"], dev_a, "A")
        run(["commit", "-m", "메뉴판 초안"], dev_a, "A", "지사 결재(커밋)는 아직 A 컴퓨터 안에만 존재")
        run(["push", "origin", "main"], dev_a, "A", "push = 결재 서류를 본사 문서고로 송부")

        # [4] B 의 pull: 본사의 최신 서류 받아오기 — 협업의 핵심 순간
        print("\n[4] B 의 pull — A 의 작업이 B 에게 도착")
        list_files(dev_b, "B(pull 전)")
        run(["pull", "origin", "main"], dev_b, "B", "pull = 본사의 새 서류를 받아와 합침")
        list_files(dev_b, "B(pull 후)")
        print("      -> A 가 만든 menu.txt 가 B 지사에 나타났습니다!")

        # [5] 주고받기: B 가 추가 -> push, A 가 pull 로 최신화
        print("\n[5] 반대 방향 — B 가 추가하고 A 가 받기")
        menu_b = dev_b / "menu.txt"
        menu_b.write_text(menu_b.read_text(encoding="utf-8") + "라떼 4500원\n", encoding="utf-8")
        run(["add", "."], dev_b, "B")
        run(["commit", "-m", "라떼 추가"], dev_b, "B")
        run(["push", "origin", "main"], dev_b, "B")
        run(["pull", "origin", "main"], dev_a, "A", "pull 을 먼저, push 는 나중에 — 협업 예절")
        print(f"      [A] 최종 메뉴판: {(dev_a / 'menu.txt').read_text(encoding='utf-8').splitlines()}")
        print("      -> 양쪽 지사가 같은 최신 상태가 되었습니다.")

        # [6] 실제 GitHub 라면: PR 흐름 해설
        print("\n[6] 실제 GitHub 협업이라면 이 흐름에 '검토'가 끼어듭니다:")
        for step in [
            "1) 브랜치를 만들어 작업 후 push",
            "2) GitHub 웹에서 Pull Request(품의서) 작성",
            "3) 동료가 리뷰 — 댓글·수정 요청이 오감",
            "4) 승인되면 main 에 merge (본편 합류)",
            "5) 팀원 전원 git pull 로 최신화",
        ]:
            print(f"      {step}")


def fallback_simulation():
    """git 이 없는 경우: 개념만 텍스트로 시뮬레이션."""
    print("\n[안내] git 명령을 찾지 못했습니다.")
    print("  맥: 'xcode-select --install' / 윈도우: git-scm.com 설치 후 재실행하세요.")
    print("\n[개념 시뮬레이션] 원격 협업의 하루:")
    events = [
        ("본사", "중앙 문서고 개설 (bare 저장소)"),
        ("A", "clone — 문서고 전체 복제로 지사 개설"),
        ("A", "커밋 후 push — 결재 서류를 본사로 송부"),
        ("B", "pull — 본사의 새 서류를 받아 지사 최신화"),
        ("B", "추가 작업 후 push, A 는 다시 pull"),
    ]
    for actor, action in events:
        print(f"  [{actor:^4}] {action}")
    print("\n  주소가 인터넷 URL(GitHub)이든 로컬 폴더든 원리는 동일합니다.")


def main():
    print("=" * 60)
    print("원격 저장소 협업 실습 — 본사(원격)와 두 지사(A·B)")
    print("=" * 60)
    if shutil.which("git"):
        real_remote_demo()
    else:
        fallback_simulation()

    print("\n정리: clone 으로 시작, push 로 올리고, pull 로 받습니다.")
    print("      pull 먼저·작게 자주 push 가 충돌을 줄이는 협업 예절입니다.")


if __name__ == "__main__":
    main()
