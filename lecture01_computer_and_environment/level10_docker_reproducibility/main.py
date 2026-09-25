"""
Lecture 01 / Level 10 — 도커: 환경을 통째로 재현하기

도커 없이도 도커를 배우는 실습입니다.
1) 도커 설치 여부를 진단하고, 2) 프로젝트 사양(딕셔너리)으로부터
Dockerfile 텍스트를 자동 생성해 줄별 해설과 함께 출력하며,
3) 생성한 Dockerfile 을 outputs/ 에 저장하고, 4) 빌드 시 층(layer)이
쌓이는 과정을 순서도로 보여줍니다.
"""

import os
import shutil
import subprocess
from pathlib import Path

# Dockerfile 을 생성할 프로젝트 사양 두 가지 (직접 해보기에서 바꿔 보세요)
SPECS = {
    "데이터 분석 배치": {
        "python_version": "3.12",
        "packages": ["pandas", "matplotlib"],
        "entry": "analyze.py",
    },
    "웹 API 서버": {
        "python_version": "3.12",
        "packages": ["fastapi", "uvicorn"],
        "entry": "server.py",
    },
}

# 각 명령의 한 줄 해설 (규격 화물 비유)
LINE_NOTES = {
    "FROM": "바탕 화물 선택 — 파이썬 깔린 미니 리눅스에서 시작",
    "WORKDIR": "상자 안 작업대 위치 지정",
    "COPY": "내 컴퓨터의 파일을 상자 안으로 싣기",
    "RUN": "포장(빌드) 중 실행할 조립 작업",
    "CMD": "개봉(실행) 시 자동으로 켤 스위치",
}


def diagnose_docker():
    """[1] 도커 설치 진단: 있으면 버전 확인, 없으면 설치 안내."""
    print("[1] 도커 설치 진단")
    docker_path = shutil.which("docker")
    if docker_path:
        print(f"    docker 명령 발견: {docker_path}")
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=False)
        version = (result.stdout or result.stderr).strip()
        print(f"    버전: {version if version else '(확인 실패 — 도커 데스크톱이 꺼져 있을 수 있음)'}")
        print("    -> [3]에서 저장하는 Dockerfile 로 'docker build' 를 시도해 볼 수 있습니다")
        return True
    print("    docker 명령이 없습니다 — 괜찮습니다, 오늘 실습은 도커 없이 진행됩니다.")
    print("    설치를 원하면: docker.com 의 Docker Desktop (맥/윈도우 공통)")
    print("    (회사 컴퓨터라면 설치 정책을 먼저 확인하세요)")
    return False


def generate_dockerfile(spec):
    """[2] 포장 지시서 생성: 사양 딕셔너리 -> Dockerfile 텍스트.

    Dockerfile 은 결국 '규칙 있는 텍스트'라서 문자열 조립으로 만들 수 있습니다."""
    lines = [
        f"FROM python:{spec['python_version']}-slim",
        "WORKDIR /app",
        # requirements 를 먼저 복사하는 이유: 층(layer) 캐시 활용!
        # 코드만 바뀌면 패키지 설치 층은 재사용되어 빌드가 빨라집니다.
        "COPY requirements.txt .",
        "RUN pip install --no-cache-dir -r requirements.txt",
        "COPY . .",
        f'CMD ["python", "{spec["entry"]}"]',
    ]
    return "\n".join(lines) + "\n"


def explain_dockerfile(name, spec):
    """생성한 Dockerfile 을 줄별 해설과 함께 출력합니다."""
    text = generate_dockerfile(spec)
    print(f"\n    ── {name} 용 Dockerfile ──")
    print(f"    (requirements.txt: {', '.join(spec['packages'])})")
    for line in text.splitlines():
        keyword = line.split()[0] if line.strip() else ""
        note = LINE_NOTES.get(keyword, "")
        print(f"    {line:<52} # {note}" if note else f"    {line}")
    return text


def main():
    print("=" * 60)
    print("도커 개념 실습 — 포장 지시서(Dockerfile)를 읽고 쓰기")
    print("=" * 60 + "\n")

    diagnose_docker()

    # [2] 사양별 Dockerfile 생성 + 해설
    print("\n[2] Dockerfile 생성기 — 프로젝트 사양에 따라 지시서가 달라집니다")
    generated = {name: explain_dockerfile(name, spec) for name, spec in SPECS.items()}
    print("\n    비교 포인트: 두 지시서의 차이는 CMD(실행 파일)와 패키지 목록뿐 —")
    print("    구조(FROM->WORKDIR->COPY->RUN->COPY->CMD)는 관례처럼 동일합니다.")

    # [3] outputs/ 폴더에 저장 — 도커 설치 후 실제 빌드해 볼 수 있게
    out_dir = Path(__file__).resolve().parent / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    print("\n[3] 생성한 지시서를 파일로 저장")
    for name, text in generated.items():
        slug = "analysis" if "분석" in name else "webapi"
        dockerfile = out_dir / f"Dockerfile.{slug}"
        dockerfile.write_text(text, encoding="utf-8")
        print(f"    저장: {dockerfile}")
    print("    -> 도커 설치 후 'docker build -f Dockerfile.analysis -t myapp .' 로 포장해 보세요")

    # [4] 빌드가 일어난다면: 층(layer)이 쌓이는 과정
    print("\n[4] 빌드 흐름 해설 — 층(layer)이 쌓여 이미지가 됩니다")
    steps = [
        ("1/6 FROM", "바탕 이미지 내려받기 (최초 1회만, 이후 캐시)"),
        ("2/6 WORKDIR", "작업 폴더 층 생성"),
        ("3/6 COPY req...", "requirements.txt 층 — 파일이 안 바뀌면 캐시 재사용"),
        ("4/6 RUN pip...", "패키지 설치 층 — 가장 오래 걸리지만 캐시되면 0초"),
        ("5/6 COPY . .", "코드 층 — 코드는 자주 바뀌므로 맨 뒤에 두는 것"),
        ("6/6 CMD", "실행 스위치 기록 -> 이미지 완성(포장 끝)"),
    ]
    for step, note in steps:
        print(f"    [{step:<14}] {note}")
    print("    이미지 완성 후: 'docker run' = 화물 개봉·가동(컨테이너 시작)")

    print("\n정리: Dockerfile(지시서) -> build -> 이미지(포장 화물) -> run -> 컨테이너(가동).")
    print("      환경 전체를 화물로 만들면 어느 항구(컴퓨터)에서든 똑같이 돌아갑니다.")


if __name__ == "__main__":
    main()
