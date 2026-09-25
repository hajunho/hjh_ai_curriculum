# Lecture 11 — LLM 활용과 RAG

> ChatGPT·Claude 같은 대규모 언어모델(LLM)의 원리를 이해하고, 프롬프트·구조화 출력·API 운영 패턴을 익힌 뒤,
> 회사 문서에 근거해 답하는 RAG 챗봇과 도구를 쓰는 미니 에이전트, 그리고 이를 "믿고 배포할" 평가·가드레일 체계까지 만듭니다.

## 이 강의에서 배우는 것

- LLM이 실제로 하는 일("다음 단어 예측")과 두 가지 태생적 한계 — 지식 시점, 환각
- 회사에서 LLM을 어디에 쓰는지(요약·초안·분류·추출)와 도입 리스크 체크리스트
- 프롬프트 엔지니어링: 역할·맥락·형식·예시(few-shot)로 결과 품질을 끌어올리는 법
- LLM 출력을 프로그램에 연결하는 기술 — JSON 구조화 출력, 스키마 검증, 재시도 루프
- API 운영 기본기 — 키 관리, 토큰 과금, 재시도·백오프·타임아웃·비용 추적
- RAG의 부품과 조립 — 임베딩·의미 검색, 청킹, 벡터 DB, 근거 인용 챗봇, "모른다" 설계
- 검색 품질 개선 — BM25 직접 구현, 하이브리드 결합, MMR 다양성, 리랭킹 개념
- 도구 호출과 에이전트 — 계산기·문서검색을 쥐여주는 ReAct 루프와 안전장치
- 평가와 가드레일 — 근거 일치율 자동 채점, 개인정보 마스킹, 금칙어 필터

**이 강의의 특징**: 전 레벨이 API 키·인터넷 없이 동작합니다. 공용 모듈 `mock_llm.py`의 규칙 기반
MockLLM(패턴 응답 + 문맥 반영 답변 조립)과 MockEmbedding(문자 n-gram 해시 + 동시출현)으로
LLM/임베딩을 오프라인 재현하고, 실제 API(Claude 등) 호출 코드는 각 레벨의 주석에
"키가 있다면 이렇게" 형태로 제공합니다. 부품의 원리를 직접 만들어 보므로,
나중에 LangChain·Chroma 같은 실전 도구를 만나도 내부가 훤히 보이게 됩니다.

## 선행 강의

- **lecture02 — 파이썬 프로그래밍 기초** (함수·클래스·딕셔너리 읽기)
- **lecture03 — 데이터 다루기** (NumPy 배열과 행렬곱, level05부터 사용)
- lecture10(NLP)을 먼저 들으면 임베딩 이해가 쉬워지지만 필수는 아닙니다.

## 레벨 목차

| 레벨 | 제목 | 난이도 |
|---|---|---|
| [level00](level00_what_is_llm/README.md) | LLM이란 — 초대형 다음 단어 예측기 | ⭐ |
| [level01](level01_llm_business_map/README.md) | LLM 업무 활용 지도 | ⭐ |
| [level02](level02_prompt_engineering/README.md) | 프롬프트 엔지니어링 | ⭐⭐ |
| [level03](level03_structured_output/README.md) | 구조화된 출력과 파싱 — JSON·검증·재시도 | ⭐⭐⭐ |
| [level04](level04_api_patterns_keys/README.md) | API 호출 패턴과 키 관리 | ⭐⭐⭐ |
| [level05](level05_embeddings_semantic_search/README.md) | 임베딩과 의미 검색 | ⭐⭐⭐ |
| [level06](level06_loading_chunking/README.md) | 문서 로딩과 청킹 | ⭐⭐⭐ |
| [level07](level07_vector_databases/README.md) | 벡터 데이터베이스 | ⭐⭐⭐⭐ |
| [level08](level08_rag_chatbot/README.md) | RAG 챗봇 — 검색·증강·생성 조립 | ⭐⭐⭐⭐ |
| [level09](level09_retrieval_quality/README.md) | 검색 품질 개선 — 하이브리드·MMR·리랭킹 | ⭐⭐⭐⭐ |
| [level10](level10_tools_agents/README.md) | 도구 호출과 에이전트 | ⭐⭐⭐⭐⭐ |
| [level11](level11_evaluation_guardrails/README.md) | 평가·가드레일·환각 억제 | ⭐⭐⭐⭐⭐ |

## 빠른 경로 (시간이 없다면 이 5개만)

1. **level00** — LLM의 원리와 한계 (모든 판단의 기초)
2. **level02** — 프롬프트 엔지니어링 (비용 0으로 품질을 올리는 기술)
3. **level05** — 임베딩과 의미 검색 (RAG의 핵심 부품)
4. **level08** — RAG 챗봇 완결 조립 (기업 LLM 활용의 표준형)
5. **level11** — 평가·가드레일 (배포 가능 여부를 가르는 기술)

## 이 강의가 실무에서 쓰이는 장면

- **사내 규정/매뉴얼 챗봇**: "연차 며칠 전에 신청해요?"에 규정 조항을 인용해 답하고, 모르는 건 담당 부서를 안내하는 봇 — level05~09의 직접 조합입니다.
- **문서 자동 처리**: 계약서·이력서·민원에서 금액·날짜·유형을 JSON으로 뽑아 ERP/CRM에 넣기 — level03의 검증·재시도 패턴이 그대로 들어갑니다.
- **고객센터 보조**: 민원 자동 분류·응대 초안 생성(level01~02) + 개인정보 마스킹과 금칙어 필터(level11)로 안전하게 운영.
- **LLM 도입 의사결정**: "어떤 업무부터, 어떤 리스크를 점검하며, 품질을 어떻게 숫자로 증명할지"를 말할 수 있게 됩니다 — level01의 활용 지도와 level11의 평가 체계가 회의 자료의 뼈대가 됩니다.
- **AI 에이전트 제품 평가**: 유행하는 "에이전트" 제품의 내부(ReAct 루프, 도구 호출, 안전장치)를 알고 도입 검토 질문을 던질 수 있습니다 — level10.

## 실행 방법

```bash
cd lecture11_llm_and_rag/level00_what_is_llm
python3 main.py        # 모든 레벨 동일. API 키·인터넷 불필요
```

공용 데이터는 `common/hjh_data.py`(사내규정·매뉴얼 `SAMPLE_DOCS`, 미니 코퍼스), 공용 모의 모델은 이 폴더의 `mock_llm.py`를 사용합니다.
