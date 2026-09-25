# Lecture 05 — 통계와 데이터 시각화

숫자를 "요약"하고, 요약이 숨기는 것을 "그림"으로 드러내고, 마지막에는
"이 차이가 진짜인가, 우연인가"를 판정하는 강의입니다. 엑셀에서 AVERAGE 를
눌러 본 적이 있다면 이미 절반은 시작한 셈입니다. 나머지 절반 — 그 평균을
언제 믿으면 안 되는지 — 를 여기서 배웁니다.

## 무엇을 배우나

- 대표값(평균·중앙값·분산)이 데이터를 요약하는 방식과 그 함정
- 분포·히스토그램·Matplotlib 로 데이터를 그림으로 만드는 법
- 사람을 속이는 그래프와 정직한 그래프를 구분하는 눈
- 상관과 인과, 확률, 정규분포와 중심극한정리
- 표본으로 전체를 추정하는 법: 신뢰구간, 가설검정, p-value
- 실무의 꽃: A/B 테스트 설계와 해석, 베이지안 사고로 불확실성 보고하기

## 선행 강의

- **lecture02 — 파이썬 프로그래밍 기초** (필수)
- **lecture03 — 데이터 다루기 (NumPy · Pandas)** (필수: 배열·데이터프레임을 다룹니다)
- **lecture04** 까지 마쳤다면 더 수월합니다.

## 레벨 목차

| 레벨 | 제목 | 난이도 |
|---|---|---|
| [level00](level00_summarizing_reality/README.md) | 숫자로 현실을 요약한다는 것 | ⭐ |
| [level01](level01_mean_median_variance/README.md) | 평균·중앙값·분산 | ⭐ |
| [level02](level02_distributions_histograms/README.md) | 분포와 히스토그램 | ⭐⭐ |
| [level03](level03_matplotlib_basics/README.md) | Matplotlib 기본 그래프 | ⭐⭐ |
| [level04](level04_good_vs_bad_charts/README.md) | 좋은 그래프 vs 나쁜 그래프 | ⭐⭐ |
| [level05](level05_correlation_causation/README.md) | 상관관계와 인과관계 | ⭐⭐⭐ |
| [level06](level06_probability_basics/README.md) | 확률의 기초 | ⭐⭐⭐ |
| [level07](level07_normal_clt/README.md) | 정규분포와 중심극한정리 | ⭐⭐⭐ |
| [level08](level08_sampling_confidence/README.md) | 표본과 신뢰구간 | ⭐⭐⭐ |
| [level09](level09_hypothesis_pvalue/README.md) | 가설검정과 p-value | ⭐⭐⭐⭐ |
| [level10](level10_ab_testing/README.md) | A/B 테스트 설계와 해석 | ⭐⭐⭐⭐ |
| [level11](level11_bayesian_thinking/README.md) | 베이지안 사고와 불확실성 소통 | ⭐⭐⭐⭐⭐ |

## 빠른 경로 (시간이 없다면 이 5개만)

1. **level01** — 평균·중앙값·분산: 모든 숫자 보고의 기초
2. **level04** — 좋은 그래프 vs 나쁜 그래프: 속지 않고, 속이지 않기
3. **level05** — 상관과 인과: 회의실에서 가장 자주 틀리는 부분
4. **level09** — 가설검정과 p-value: "이 차이 진짜예요?"에 답하는 법
5. **level10** — A/B 테스트: 데이터로 의사결정하는 조직의 표준 도구

## 이 강의가 실무에서 쓰이는 장면

- **월간 보고**: "평균 매출 12% 상승"이라는 문장 뒤에 극단값 하나가 숨어
  있는지 중앙값과 분포로 확인합니다 (level01~02).
- **경영진 보고 자료**: 축이 잘린 막대그래프로 성과를 부풀리는 실수를
  피하고, 남이 만든 왜곡 차트를 간파합니다 (level03~04).
- **마케팅 성과 분석**: "광고비를 늘렸더니 매출이 올랐다"가 인과인지,
  성수기라는 교란변수 때문인지 따져봅니다 (level05).
- **신제품·UI 개편 결정**: A/B 테스트를 설계하고, 중간에 몰래 결과를
  훔쳐보는 것(peeking)이 왜 위험한지 이해합니다 (level09~10).
- **불확실성 보고**: "된다/안 된다" 대신 "B안이 우월할 확률 92%"처럼
  경영진이 결정할 수 있는 언어로 말합니다 (level08, 11).

## 실행 방법

각 레벨 폴더에서 저장소 가상환경으로 실행합니다.

```bash
cd lecture05_statistics_visualization/level00_summarizing_reality
/Users/junhoha/Documents/hajunho/hjh_ai_curriculum/.venv/bin/python main.py
```

그래프를 그리는 레벨은 실행 후 해당 폴더의 `outputs/` 안에 PNG 파일이
생기고, 경로가 화면에 출력됩니다. 데이터는 전부 `common/hjh_data.py` 또는
코드 안에서 생성하므로 인터넷 연결이 필요 없습니다.
