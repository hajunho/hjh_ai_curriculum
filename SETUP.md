# 환경 준비 가이드 (SETUP)

이 커리큘럼은 **GPU 없는 보통 노트북**(맥/윈도우)에서 전 과정이 돌아갑니다.

## 1. 파이썬 설치 확인

터미널(맥: 터미널 앱, 윈도우: PowerShell)에서:

```bash
python3 --version   # 3.10 이상이면 OK (윈도우는 python --version)
```

없다면 https://www.python.org/downloads/ 에서 설치하세요.
자세한 과정은 lecture01/level03 에서 그림처럼 따라갑니다.

## 2. 저장소 받기와 가상환경

```bash
git clone https://github.com/hajunho/hjh_ai_curriculum.git
cd hjh_ai_curriculum

python3 -m venv .venv
source .venv/bin/activate        # 윈도우: .venv\Scripts\activate
pip install -r requirements.txt
```

`(.venv)` 가 프롬프트 앞에 붙으면 성공입니다. 가상환경이 뭔지는
lecture01/level04 에서 배웁니다. 지금은 "이 프로젝트 전용 도구상자"라고만
생각하세요.

## 3. 동작 확인

```bash
python3 common/hjh_data.py
```

"모든 생성기 정상 동작." 이 출력되면 준비 끝입니다.

## 4. 강의 실행 방법

모든 레벨은 같은 방식입니다.

```bash
cd lecture06_machine_learning_basics/level03_linear_regression
cat README.md      # 강의 노트 읽기 (GitHub 에서 읽어도 됩니다)
python3 main.py    # 실습 실행
```

## 5. 자주 묻는 질문

- **인터넷이 필요한가요?** 설치 후에는 필요 없습니다. 실습 데이터는 코드가 직접 생성합니다.
- **GPU 가 필요한가요?** 아니요. 딥러닝 강의도 작게 설계해 CPU 로 수 분 내에 끝납니다.
- **LLM API 키가 필요한가요?** lecture11 일부 레벨은 키가 있으면 더 좋지만,
  키 없이도 동작하는 오프라인 모드를 모든 레벨에 넣어 두었습니다.
- **에러가 나요.** 각 레벨 README 하단 "자주 하는 실수" 를 먼저 확인하세요.
