# Lecture 10 — 자연어처리(NLP)

> 컴퓨터가 사람의 말을 다루는 기술을 밑바닥부터 쌓아 올립니다.
> 문자열 정제와 정규표현식에서 시작해 TF-IDF·감성 분석·임베딩을 거쳐,
> 어텐션과 미니 트랜스포머를 직접 구현하고 GPT/BERT 사전학습 패러다임까지 도달합니다.

## 이 강의에서 배우는 것

- 왜 언어가 컴퓨터에게 유독 어려운지 — 중의성, 문맥, 그리고 한국어 교착어의 특성
- 지저분한 텍스트를 분석 가능한 형태로 만드는 전처리 파이프라인과 정규표현식
- 텍스트를 숫자로 바꾸는 세 가지 세대: BoW/TF-IDF → 워드 임베딩 → 문맥 임베딩
- 리뷰 감성 분석기를 직접 만들고, 모델이 어떤 단어를 근거로 판단했는지 해석하기
- RNN·어텐션·트랜스포머의 작동 원리를 numpy 와 torch 로 부품 단위로 조립하기
- GPT(이어쓰기)와 BERT(빈칸 채우기)가 세상을 바꾼 사전학습 패러다임의 핵심

이 강의는 konlpy, nltk, transformers 같은 NLP 전용 패키지를 **일부러 쓰지 않습니다**.
형태소 분석기, TF-IDF, 어텐션을 전부 직접 구현해 봐야 "라이브러리가 안에서 무엇을
하는지"를 알게 되고, 다음 강의(lecture11 — LLM 활용과 RAG)에서 도구를 쓸 때
블랙박스가 아니라 유리상자가 됩니다.

## 선행 강의

- **lecture02 — 파이썬 프로그래밍 기초** (문자열, 리스트, 딕셔너리, 함수)
- **lecture03 — 데이터 다루기** (NumPy 배열 연산)
- **lecture06 — 머신러닝 입문** (분류, 로지스틱 회귀, 학습/평가 분리)
- **lecture08 — 딥러닝 기초** (level07 이후의 torch 실습에 필요. level00~06 은 없어도 됩니다)

## 레벨 목차

| 레벨 | 제목 | 난이도 |
|---|---|---|
| [level00](level00_why_language_is_hard/README.md) | 컴퓨터가 언어를 다루는 어려움 | ⭐ |
| [level01](level01_text_preprocessing/README.md) | 텍스트 전처리 기초 | ⭐⭐ |
| [level02](level02_regex/README.md) | 정규표현식 | ⭐⭐ |
| [level03](level03_tokenization_korean/README.md) | 토큰화와 한국어의 특수성 | ⭐⭐⭐ |
| [level04](level04_bow_tfidf/README.md) | BoW 와 TF-IDF | ⭐⭐⭐ |
| [level05](level05_text_classification/README.md) | 텍스트 분류 실습 — 리뷰 감성 분석 | ⭐⭐⭐ |
| [level06](level06_word_embeddings/README.md) | 워드 임베딩 | ⭐⭐⭐ |
| [level07](level07_rnn_sequences/README.md) | RNN 과 시퀀스 모델 | ⭐⭐⭐⭐ |
| [level08](level08_attention/README.md) | 어텐션 메커니즘 | ⭐⭐⭐⭐ |
| [level09](level09_transformer_anatomy/README.md) | 트랜스포머 구조 해부 | ⭐⭐⭐⭐ |
| [level10](level10_mini_transformer/README.md) | 미니 트랜스포머 직접 구현 | ⭐⭐⭐⭐⭐ |
| [level11](level11_pretraining_paradigm/README.md) | 사전학습 패러다임 — BERT 와 GPT | ⭐⭐⭐⭐ |

## 빠른 경로 (시간이 없다면 이 5개만)

1. **level01 텍스트 전처리** — 모든 텍스트 업무의 출발점. 실무 활용 빈도 1위.
2. **level04 BoW·TF-IDF** — "텍스트를 숫자로"의 표준. 검색·분류·키워드 추출의 기반.
3. **level05 텍스트 분류** — 리뷰 감성 분석 전체 파이프라인을 한 번에 경험.
4. **level08 어텐션** — 요즘 AI 를 이해하는 단 하나의 열쇠를 골라야 한다면 이것.
5. **level11 사전학습 패러다임** — GPT/BERT 가 왜 대단한지, LLM 시대의 큰 그림.

## 이 강의가 실무에서 쓰이는 장면

- **VOC(고객의 소리) 분석**: 하루 수천 건 쌓이는 리뷰·문의를 긍정/부정으로 자동 분류하고,
  불만 키워드를 뽑아 주간 리포트로 만듭니다. (level01, 04, 05)
- **문서에서 정보 추출**: 계약서·견적서 더미에서 금액, 날짜, 연락처를 정규표현식으로
  일괄 추출해 엑셀로 정리합니다. (level02)
- **사내 문서 검색·요약의 기반**: "휴가 규정이 어디 있더라?"에 답하는 사내 검색과
  RAG 챗봇은 전부 이 강의의 토큰화·TF-IDF·임베딩 위에 서 있습니다. (level03, 04, 06)
- **LLM 을 제대로 쓰는 눈**: 토큰 단위 과금이 왜 그런지, 컨텍스트 길이 제한이 왜 있는지,
  프롬프트가 왜 그렇게 동작하는지 — 트랜스포머 구조를 알면 전부 설명됩니다. (level08~11)

## 실행 방법

```bash
cd lecture10_nlp_text/level00_why_language_is_hard
python3 main.py
```

모든 실습은 인터넷 연결 없이 동작하며, 데이터는 `common/hjh_data.py` 가 즉석에서
생성합니다. torch 를 쓰는 레벨(07, 10, 11)도 CPU 에서 90초 안에 끝나도록 아주 작게
설계했습니다.
