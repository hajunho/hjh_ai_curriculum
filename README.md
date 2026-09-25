# hjh AI Curriculum — 비전공 직장인을 위한 AI 완주 코스

> **엑셀만 쓰던 사람이 머신러닝·딥러닝·LLM 엔지니어링까지.**
> IT 를 전공하지 않은 비즈니스 직장인이 "AI 를 이해하고 직접 다루는 사람"이
> 되도록 설계한 12개 강의, 144개 레벨의 실습 중심 커리큘럼입니다.

- **작성**: 하준호 (hajunho) · ZeliDesk / 에듀테크학과
- **라이선스**: MIT — 모든 강의 노트와 코드는 이 커리큘럼을 위해 새로 작성한 순수 창작물입니다.
  외부 교재·데이터셋을 복제하지 않았으며, 실습 데이터도 전부 코드로 직접 생성합니다.
- **언어**: 강의는 한국어, 코드 식별자는 영어

---

## 이 커리큘럼이 다른 점

1. **비유 먼저, 수식은 나중** — 모든 개념을 업무·일상 비유로 먼저 설명합니다.
2. **설치 지옥 없음** — 실습 데이터를 코드가 직접 만들어내므로 다운로드가 필요 없고,
   GPU 없는 노트북에서 전 과정이 돌아갑니다.
3. **비즈니스 문제로 배움** — 매출 예측, 고객 이탈, 이상거래 탐지, 사내 문서 챗봇 등
   회사에서 실제로 마주치는 문제로 실습합니다.
4. **밑바닥 구현 포함** — 퍼셉트론, 역전파, 토크나이저, 미니 GPT 까지 직접 만들어
   "블랙박스"를 열어 봅니다.

## 전체 구조

12개 강의(lecture01~12), 각 강의는 12개 레벨(level00~11)로 구성됩니다.
level00 은 완전 입문(비유와 개념), level11 은 실무 심화입니다.

| 강의 | 주제 | 한 줄 소개 |
|---|---|---|
| [lecture01](lecture01_computer_and_environment/) | 컴퓨터와 개발 환경 | 터미널, 파이썬 설치, Git, 클라우드까지 |
| [lecture02](lecture02_python_basics/) | 파이썬 프로그래밍 기초 | 변수부터 객체지향·테스트까지 |
| [lecture03](lecture03_data_handling/) | 데이터 다루기 | 엑셀 탈출 — NumPy · Pandas |
| [lecture04](lecture04_database_sql/) | 데이터베이스와 SQL | SELECT 부터 윈도우 함수·파이썬 연동까지 |
| [lecture05](lecture05_statistics_visualization/) | 통계와 시각화 | 평균의 함정부터 A/B 테스트까지 |
| [lecture06](lecture06_machine_learning_basics/) | 머신러닝 입문 | 회귀·분류·앙상블·모델 해석 |
| [lecture07](lecture07_business_ml_practice/) | 비즈니스 ML 실전 | 매출 예측·고객 이탈·이상거래 탐지 |
| [lecture08](lecture08_deep_learning_foundations/) | 딥러닝 기초 | 퍼셉트론 밑바닥 구현부터 PyTorch 까지 |
| [lecture09](lecture09_computer_vision/) | 컴퓨터 비전 | CNN·전이학습·산업 응용 |
| [lecture10](lecture10_nlp_text/) | 자연어처리 | 한국어 전처리부터 트랜스포머 구현까지 |
| [lecture11](lecture11_llm_and_rag/) | LLM 활용과 RAG | 프롬프트, 임베딩, 사내 문서 챗봇 |
| [lecture12](lecture12_llm_engineering_mlops/) | LLM 엔지니어링과 MLOps | 토크나이저·미니 GPT·정렬·양자화·서빙 |

상세 목차는 [CURRICULUM.md](CURRICULUM.md), 환경 준비는 [SETUP.md](SETUP.md),
용어가 막히면 [GLOSSARY.md](GLOSSARY.md) 를 보세요.

## 어떻게 공부하나

1. `SETUP.md` 대로 파이썬 환경을 준비합니다 (30분).
2. 각 레벨 폴더의 `README.md` 를 읽고 → `main.py` 를 실행하고 → 과제를 풉니다.
3. 한 레벨은 30~90분 분량입니다. 주 5레벨이면 **약 7개월**에 완주합니다.
4. 순서대로가 정석이지만, 각 강의 README 에 "빠른 경로"를 안내해 두었습니다.

```bash
git clone https://github.com/hajunho/hjh_ai_curriculum.git
cd hjh_ai_curriculum
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 lecture01_computer_and_environment/level00_what_computers_do/main.py
```

## 저작권과 라이선스

이 저장소의 모든 문서·코드·데이터 생성기는 하준호(hajunho)가 이 커리큘럼을 위해
새로 작성했습니다. 특정 상용 교육과정이나 서적의 내용을 옮기지 않았고,
외부 데이터셋 파일을 포함하지 않습니다. MIT 라이선스로 누구나 자유롭게
사용·수정·재배포할 수 있습니다. 강의에 활용하실 때 출처를 남겨 주시면 감사하겠습니다.
