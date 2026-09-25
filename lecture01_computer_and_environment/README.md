# Lecture 01 — 컴퓨터와 개발 환경

> 코딩을 시작하기 전에 반드시 갖춰야 할 "작업 환경"을 이해하고 세팅하는 강의입니다.
> 컴퓨터가 일하는 원리부터 터미널, 파이썬 설치, Git, 도커, 클라우드 GPU까지 —
> AI 를 배우기 위한 모든 기초 체력을 여기서 만듭니다.

## 이 강의에서 배우는 것

- 컴퓨터(CPU·메모리·저장장치)가 실제로 무슨 일을 하는지 감 잡기
- 파일·폴더·경로, 터미널 명령 같은 개발자의 기본 언어
- 파이썬 설치, 가상환경, 에디터·Jupyter 등 실습 도구 세팅
- Git/GitHub 로 작업물을 안전하게 관리하고 협업하는 방법
- 환경변수·비밀키·도커·클라우드 GPU 등 실무 환경 감각

## 선행 강의

없음. 이 커리큘럼의 첫 강의입니다. 엑셀을 다뤄 본 정도의 컴퓨터 경험이면 충분합니다.

## 레벨 목차

| 레벨 | 제목 | 난이도 |
|---|---|---|
| [level00](level00_what_computers_do/README.md) | 컴퓨터는 무엇을 하는 기계인가 | ⭐ |
| [level01](level01_files_folders_paths/README.md) | 파일·폴더·경로의 개념 | ⭐ |
| [level02](level02_terminal_basics/README.md) | 터미널 첫걸음 | ⭐ |
| [level03](level03_install_python/README.md) | 파이썬 설치와 첫 실행 | ⭐ |
| [level04](level04_venv_packages/README.md) | 가상환경과 패키지 관리 | ⭐⭐ |
| [level05](level05_editors_jupyter/README.md) | 코드 에디터와 Jupyter | ⭐⭐ |
| [level06](level06_git_basics/README.md) | Git — 버전 관리의 개념 | ⭐⭐ |
| [level07](level07_github_collaboration/README.md) | GitHub — 원격 저장소와 협업 | ⭐⭐ |
| [level08](level08_shell_automation/README.md) | 셸 스크립트로 반복 작업 자동화 | ⭐⭐⭐ |
| [level09](level09_env_vars_secrets/README.md) | 환경변수와 비밀키 안전하게 다루기 | ⭐⭐⭐ |
| [level10](level10_docker_reproducibility/README.md) | 도커 — 환경을 통째로 재현하기 | ⭐⭐⭐⭐ |
| [level11](level11_cloud_gpu_servers/README.md) | 원격 서버·클라우드·GPU 환경 | ⭐⭐⭐⭐ |

## 빠른 경로 (시간이 없다면 이것만)

1. **level02 터미널 첫걸음** — 이후 모든 강의의 실습이 터미널에서 시작됩니다.
2. **level03 파이썬 설치와 첫 실행** — `python3 main.py` 를 직접 쳐 보는 순간이 출발점입니다.
3. **level04 가상환경과 패키지 관리** — 실무에서 "내 컴퓨터에선 되는데요" 사고의 절반이 여기서 예방됩니다.
4. **level06 Git 기초** — 작업물을 잃어버리지 않는 최소한의 안전장치입니다.
5. **level09 환경변수와 비밀키** — API 키 유출 사고를 막는 습관은 처음부터 들여야 합니다.

## 이 강의가 실무에서 쓰이는 장면

- **보고서 자동화의 첫 단추**: 매주 반복하는 파일 정리·이름 변경을 스크립트 한 번으로 끝내는 팀원은 이 강의의 level08 을 이해한 사람입니다.
- **협업 사고 예방**: "최종_진짜최종_수정2.xlsx" 대신 Git 커밋 이력으로 문서를 관리하면 누가 언제 무엇을 바꿨는지 1분 안에 찾습니다.
- **보안 감사 통과**: 코드에 비밀키를 하드코딩하지 않는 습관(level09)은 사내 보안 점검과 외부 감사에서 가장 먼저 확인하는 항목입니다.
- **AI 프로젝트 견적 회의**: GPU 클라우드 비용이 왜 시간당으로 청구되는지(level11)를 알면 개발팀과의 예산 회의에서 대화가 통합니다.

## 실습 방법

각 레벨 폴더에서 아래 한 줄이면 됩니다. 인터넷 연결이나 API 키는 전혀 필요 없습니다.

```bash
python3 main.py
```
