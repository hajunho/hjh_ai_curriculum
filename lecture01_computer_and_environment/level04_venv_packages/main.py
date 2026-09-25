"""
Lecture 01 / Level 04 — 가상환경과 패키지 관리

지금 이 코드를 실행 중인 파이썬 환경을 진단합니다.
sys.prefix 와 sys.base_prefix 비교로 가상환경 여부를 판정하고,
importlib.metadata 로 설치된 패키지(도구상자 내용물)를 나열하며,
pip freeze 가 만드는 requirements.txt 의 정체를 보여줍니다.
"""

import importlib.metadata
import importlib.util
import sys


def check_which_toolbox():
    """[1] 어느 도구상자인가: 가상환경 안인지 시스템 파이썬인지 판정."""
    prefix = sys.prefix            # 지금 파이썬이 도구를 찾는 공간
    base = sys.base_prefix         # 원본(시스템) 파이썬의 공간
    in_venv = prefix != base       # 둘이 다르면 = 전용 도구상자를 차고 있음
    print("[1] 어느 도구상자를 차고 있는가")
    print(f"    sys.prefix      (지금 쓰는 공간): {prefix}")
    print(f"    sys.base_prefix (원본 공간)     : {base}")
    if in_venv:
        print("    판정: 가상환경 안입니다 — 프로젝트 전용 도구상자를 착용 중")
    else:
        print("    판정: 시스템 파이썬입니다 — 회사 공용 공구함을 쓰는 중")
        print("    (저장소의 .venv/bin/python 으로 실행하면 판정이 달라집니다)")
    print(f"    실행 파일: {sys.executable}")
    return in_venv


def list_installed_packages(limit=15):
    """[2] 도구상자 내용물: 설치된 패키지 이름과 버전 나열."""
    dists = sorted(
        ((d.metadata["Name"] or "?", d.version) for d in importlib.metadata.distributions()),
        key=lambda pair: pair[0].lower(),
    )
    print(f"\n[2] 이 환경에 설치된 패키지: 총 {len(dists)}개 (pip list 와 같은 정보)")
    for name, version in dists[:limit]:
        print(f"    - {name:<28} {version}")
    if len(dists) > limit:
        print(f"    ... 외 {len(dists) - limit}개")
    return dists


def check_standard_tools():
    """[3] 기본 공구(표준 라이브러리) 확인: 설치 없이 이미 있는 도구들."""
    basics = ["json", "csv", "datetime", "sqlite3", "pathlib", "random"]
    print("\n[3] 기본 공구 확인 — 표준 라이브러리는 설치 없이 바로 사용")
    for name in basics:
        found = importlib.util.find_spec(name) is not None
        print(f"    {'[OK]' if found else '[없음]'} {name}")
    print("    -> 이 커리큘럼의 lecture01~02 는 전부 기본 공구만으로 진행합니다")


def preview_requirements(dists, limit=8):
    """[4] requirements.txt 미리보기: pip freeze 가 만드는 물품 목록표."""
    print("\n[4] requirements.txt 미리보기 — '이름==버전' 형식의 물품 목록표")
    if not dists:
        print("    (설치된 패키지가 없어 목록이 비어 있습니다)")
        return
    for name, version in dists[:limit]:
        print(f"    {name}=={version}")
    if len(dists) > limit:
        print(f"    ... 외 {len(dists) - limit}줄")
    print("    -> 이 파일만 넘기면 동료가 'pip install -r requirements.txt' 로")
    print("       같은 구성의 도구상자를 자기 컴퓨터에 그대로 재현합니다")


def main():
    print("=" * 60)
    print("가상환경·패키지 진단 — 나는 지금 어떤 도구상자를 차고 있나")
    print("=" * 60 + "\n")

    in_venv = check_which_toolbox()
    dists = list_installed_packages()
    check_standard_tools()
    preview_requirements(dists)

    # [5] 요약
    print("\n[5] 진단 요약")
    print(f"    가상환경 여부 : {'예 (.venv 계열)' if in_venv else '아니오 (시스템 파이썬)'}")
    print(f"    설치 패키지   : {len(dists)}개")
    print("\n정리: 프로젝트마다 전용 도구상자(venv)를 만들고,")
    print("      pip freeze 목록표(requirements.txt)로 구성을 공유하세요.")
    print("      '내 컴퓨터에선 되는데요' 사고의 절반이 여기서 예방됩니다.")


if __name__ == "__main__":
    main()
