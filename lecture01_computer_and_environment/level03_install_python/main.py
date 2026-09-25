"""
Lecture 01 / Level 03 — 파이썬 설치와 첫 실행

내 컴퓨터의 파이썬 환경을 스스로 진단하는 '건강검진' 스크립트입니다.
버전·실행 파일 위치·운영체제·PATH 를 확인하고,
REPL(대화형)과 스크립트 실행의 차이를 출력으로 비교합니다.
"""

import os
import platform
import sys

# 이 커리큘럼이 요구하는 최소 파이썬 버전 (직접 해보기 2번에서 바꿔 보세요)
REQUIRED = (3, 10)


def check_python_identity():
    """[1] 통역사 신원 확인: 버전과 실행 파일 위치."""
    version = sys.version_info  # (major, minor, micro, ...) 숫자로 비교 가능
    ok = version >= REQUIRED
    print("[1] 파이썬 신원 확인")
    print(f"    버전            : {platform.python_version()}")
    print(f"    실행 파일 위치   : {sys.executable}")
    print(f"    요구 버전       : {REQUIRED[0]}.{REQUIRED[1]} 이상")
    print(f"    판정            : {'통과 — 커리큘럼 진행 가능' if ok else '미달 — 업그레이드 필요'}")
    return ok


def check_workplace():
    """[2] 근무지 확인: 운영체제와 현재 작업 디렉터리."""
    os_name = platform.system()  # 'Darwin'(맥) / 'Windows' / 'Linux'
    friendly = {"Darwin": "macOS(맥)", "Windows": "윈도우", "Linux": "리눅스"}.get(os_name, os_name)
    print("\n[2] 근무 환경 확인")
    print(f"    운영체제        : {friendly} ({platform.release()})")
    print(f"    프로세서        : {platform.machine()}")
    print(f"    현재 작업 폴더  : {os.getcwd()}")
    print("    -> 상대경로는 모두 이 폴더를 기준으로 해석됩니다 (level01 복습)")
    return True


def check_path():
    """[3] PATH 훑어보기: 셸이 명령을 찾는 폴더 목록."""
    raw = os.environ.get("PATH", "")
    entries = [e for e in raw.split(os.pathsep) if e]
    print("\n[3] PATH — 셸이 'python3' 같은 명령을 찾아다니는 폴더 목록")
    print(f"    등록된 폴더 수  : {len(entries)}개 (앞에서부터 순서대로 탐색)")
    for i, entry in enumerate(entries[:5], start=1):
        print(f"    {i}순위: {entry}")
    if len(entries) > 5:
        print(f"    ... 외 {len(entries) - 5}개")
    print("    -> 'command not found' = 이 목록 어디에도 그 이름이 없다는 뜻")
    return len(entries) > 0


def compare_repl_vs_script():
    """[4] REPL(즉석 대화) vs 스크립트(문서 지시) 비교."""
    print("\n[4] 두 가지 실행 방식 비교")
    print("    (a) REPL — 터미널에 python3 만 치면 나오는 '>>>' 에서 즉석 대화:")
    print("        >>> 120 * 12")
    print(f"        {120 * 12}")
    print("        >>> '보고서' + '_최종.xlsx'")
    print(f"        '{'보고서' + '_최종.xlsx'}'")
    print("    (b) 스크립트 — 바로 지금! 이 출력 전체가 main.py 라는 문서를")
    print("        python3 에게 통째로 넘겨 위에서부터 실행한 결과입니다.")
    print("    -> 실험은 REPL, 정식 업무는 스크립트(보관·공유·재실행 가능)")
    return True


def main():
    print("=" * 60)
    print("파이썬 환경 자가진단 — 나의 통역사는 준비되었는가")
    print("=" * 60 + "\n")

    results = {
        "파이썬 버전": check_python_identity(),
        "운영체제 확인": check_workplace(),
        "PATH 설정": check_path(),
        "실행 방식 이해": compare_repl_vs_script(),
    }

    # [5] 종합 판정표
    print("\n[5] 종합 진단 요약")
    for item, ok in results.items():
        mark = "[OK] " if ok else "[주의]"
        print(f"    {mark} {item}")
    if all(results.values()):
        print("\n    축하합니다! 이 컴퓨터는 커리큘럼을 진행할 준비가 되었습니다.")
    else:
        print("\n    [주의] 항목을 해결한 뒤 다시 실행해 보세요.")

    print("\n정리: 환경이 이상할 땐 언제나 sys.executable(어느 파이썬인가)과")
    print("      python3 --version(몇 버전인가)부터 확인하는 습관을 들이세요.")


if __name__ == "__main__":
    main()
