"""
Lecture 01 / Level 02 — 터미널 첫걸음

파이썬으로 만든 '미니 셸 시뮬레이터'로 pwd/ls/cd/mkdir/cp/mv/rm 명령의
동작 원리를 안전한 임시 폴더 안에서 체험합니다.
셸이란 결국 '현재 위치(cwd)를 기억하며 파일 작업을 대신해 주는 통역사'임을
코드 구조로 확인합니다.
"""

import shutil
import tempfile
from pathlib import Path


class MiniShell:
    """진짜 셸의 핵심 동작(현재 위치 관리 + 파일 명령)을 흉내 낸 클래스."""

    def __init__(self, root: Path):
        # resolve(): 심볼릭 링크까지 푼 절대경로 (맥의 /var -> /private/var 대비)
        self.root = root.resolve()   # 연습장의 최상위 (여기 밖으로는 못 나감)
        self.cwd = self.root      # 현재 작업 디렉터리 — 셸의 핵심 상태!

    def run(self, line: str) -> str:
        """'cp a.txt backup' 같은 한 줄을 해석해 해당 메서드를 호출합니다."""
        parts = line.split()
        command, args = parts[0], parts[1:]
        handler = getattr(self, f"cmd_{command}", None)  # cmd_ls, cmd_cd ...
        if handler is None:
            return f"(에러) '{command}': 명령을 찾을 수 없습니다"
        return handler(*args)

    # ---- 각 명령의 구현: 결국 pathlib 함수 호출입니다 ----
    def cmd_pwd(self):
        rel = self.cwd.relative_to(self.root)
        return "/" + str(rel) if str(rel) != "." else "/ (연습장 최상위)"

    def cmd_ls(self):
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in self.cwd.iterdir())
        return "  ".join(names) if names else "(비어 있음)"

    def cmd_cd(self, name):
        target = (self.cwd / name).resolve()
        if not target.is_dir():
            return f"(에러) '{name}': 그런 폴더가 없습니다"
        self.cwd = target  # '이동'이란 그저 현재 위치 변수를 바꾸는 것!
        return f"이동 완료 -> {self.cmd_pwd()}"

    def cmd_mkdir(self, name):
        (self.cwd / name).mkdir()
        return f"폴더 '{name}' 생성"

    def cmd_cp(self, src, dst):
        dst_path = self.cwd / dst
        if dst_path.is_dir():
            dst_path = dst_path / src  # 폴더로 복사하면 같은 이름으로 들어감
        shutil.copy(self.cwd / src, dst_path)
        return f"'{src}' -> '{dst}' 복사 (원본은 그대로)"

    def cmd_mv(self, src, dst):
        dst_path = self.cwd / dst
        if dst_path.is_dir():
            dst_path = dst_path / src
        (self.cwd / src).rename(dst_path)
        return f"'{src}' -> '{dst}' 이동 (같은 폴더면 이름 변경)"

    def cmd_rm(self, name):
        (self.cwd / name).unlink()
        return f"'{name}' 삭제 — 휴지통 없이 즉시 지워졌습니다!"


# 연습 시나리오: (명령, 이 명령이 하는 일 해설)
SCENARIO = [
    ("pwd", "지금 내가 어디에 서 있는지 확인 — 터미널 작업의 제1습관"),
    ("ls", "이 방(폴더)에 무엇이 있는지 확인"),
    ("mkdir backup", "백업용 캐비닛(폴더)을 새로 설치"),
    ("cp 주간보고.txt backup", "보고서 사본을 backup 폴더에 보관"),
    ("cd backup", "backup 폴더 안으로 이동"),
    ("ls", "사본이 잘 들어왔는지 확인"),
    ("mv 주간보고.txt 주간보고_백업본.txt", "같은 폴더 안 mv = 이름 변경"),
    ("cd ..", "'..' = 한 층 위(부모 폴더)로"),
    ("rm 임시메모.txt", "필요 없어진 메모를 삭제 (즉시·영구!)"),
    ("ls", "정리가 끝난 방의 최종 상태 확인"),
]


def main():
    print("=" * 60)
    print("터미널 첫걸음 — 미니 셸 시뮬레이터")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        office = Path(tmp) / "practice_office"
        office.mkdir()

        # [1] 연습장 준비: 샘플 파일을 깔아 둡니다 (임시 폴더라 실수해도 안전)
        for name, content in [
            ("주간보고.txt", "이번 주 매출 요약"),
            ("회의록.txt", "9월 정례 회의"),
            ("임시메모.txt", "점심 메뉴 후보"),
        ]:
            (office / name).write_text(content, encoding="utf-8")
        print(f"\n[1] 연습장 준비 완료: 임시 폴더에 샘플 파일 3개 생성")
        print("    (실제 여러분 컴퓨터의 파일은 전혀 건드리지 않습니다)")

        # [2] 시나리오 실행: 명령 한 줄씩 '입력 -> 실행 -> 해설'
        shell = MiniShell(office)
        print("\n[2] 명령 시나리오 실행 ($ 표시가 '입력한 명령'입니다)")
        for step, (line, note) in enumerate(SCENARIO, start=1):
            print(f"\n  ({step}) $ {line}")
            print(f"      -> {shell.run(line)}")
            print(f"      해설: {note}")

        # [3] 최종 상태와 정리
        n_ops = len(SCENARIO)
        print("\n[3] 최종 폴더 구조:")
        for p in sorted(office.rglob("*")):
            depth = len(p.relative_to(office).parts) - 1
            tag = "/" if p.is_dir() else ""
            print("      " + "  " * depth + f"- {p.name}{tag}")
        print(f"\n    명령 {n_ops}줄로 끝냈습니다. 마우스였다면 창 열기·드래그·이름 클릭…")
        print("    그리고 이 {n}줄은 저장해 두면 내일도, 후임자도 그대로 재실행할 수 있습니다.".format(n=n_ops))

    print("\n정리: 셸 = '현재 위치'를 기억하며 파일 작업을 대신하는 통역사.")
    print("      이제 진짜 터미널을 열고 pwd, ls, cd 부터 직접 쳐 보세요!")


if __name__ == "__main__":
    main()
