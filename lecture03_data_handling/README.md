# Lecture 03 — 데이터 다루기 (NumPy · Pandas)

> 엑셀로 하던 모든 표 작업을 코드로 옮기고, 엑셀이 못 하는 일까지 해내는 강의입니다.

## 이 강의에서 배우는 것

- 데이터가 왜 "행과 열의 표"로 정리되는지, 그 표를 코드로 다루면 무엇이 달라지는지
- NumPy 배열(ndarray)로 수십만 개 숫자를 한 번에 계산하는 벡터화 사고방식
- Pandas DataFrame으로 불러오기 → 정제 → 필터 → 집계 → 병합 → 시계열 → 피벗까지, 실무 데이터 분석의 전체 파이프라인
- 데이터가 커졌을 때 메모리를 아끼고 나눠 처리하는 대용량 전략, 그리고 언제 DB/Spark로 넘어가야 하는지

모든 실습은 가상의 카페 체인(강남·홍대·부산·대전·수원 5개 지점) 매출 데이터를 사용합니다.
데이터는 `common/hjh_data.py`가 코드로 직접 생성하므로 인터넷 연결이 필요 없고, 결측치·음수 오염까지
실무처럼 심어져 있습니다.

## 선행 강의

- **lecture01 — 컴퓨터와 프로그래밍의 기초**, **lecture02 — 파이썬 기초** (변수, 리스트, 딕셔너리, 반복문, 함수, 파일 읽기·쓰기)
- 엑셀에서 SUM/필터/피벗 테이블을 써 본 경험이 있으면 비유가 더 잘 와닿습니다.

## 레벨 구성

| 레벨 | 제목 | 난이도 |
|---|---|---|
| [level00](level00_what_is_data/README.md) | 데이터란 무엇인가 — 표의 구조 | ⭐ |
| [level01](level01_excel_to_python/README.md) | 엑셀에서 파이썬으로 | ⭐ |
| [level02](level02_numpy_basics/README.md) | NumPy 배열 기초 | ⭐⭐ |
| [level03](level03_pandas_dataframe/README.md) | Pandas — Series 와 DataFrame | ⭐⭐ |
| [level04](level04_loading_data/README.md) | 데이터 불러오기 (CSV·Excel·JSON) | ⭐⭐ |
| [level05](level05_filter_sort_select/README.md) | 필터링·정렬·선택 | ⭐⭐ |
| [level06](level06_missing_outliers/README.md) | 결측치와 이상치 처리 | ⭐⭐⭐ |
| [level07](level07_groupby_aggregation/README.md) | 그룹화와 집계 (groupby) | ⭐⭐⭐ |
| [level08](level08_merge_join/README.md) | 병합과 조인 (merge·concat) | ⭐⭐⭐ |
| [level09](level09_time_series/README.md) | 시계열 데이터 다루기 | ⭐⭐⭐⭐ |
| [level10](level10_pivot_reshape_window/README.md) | 피벗·리셰이프·윈도우 연산 | ⭐⭐⭐⭐ |
| [level11](level11_large_data_strategies/README.md) | 대용량 데이터 처리 전략 | ⭐⭐⭐⭐⭐ |

## 빠른 경로 (시간이 없다면 이 5개만)

1. **level03** — DataFrame 구조를 모르면 아무것도 못 합니다.
2. **level05** — 실무 질문의 80%는 "골라내고 줄 세우기"입니다.
3. **level06** — 현실 데이터는 반드시 오염되어 있습니다.
4. **level07** — "지점별 매출은?" 같은 질문의 답이 groupby입니다.
5. **level08** — 표 하나로 끝나는 분석은 없습니다. 병합이 실전입니다.

빠른 경로를 마친 뒤 시계열 업무가 있으면 level09를, 보고서 업무가 많으면 level10을 추가하세요.

## 이 강의가 실무에서 쓰이는 장면

- **월간 실적 보고**: 지점 30곳 매출 CSV를 열어 지점별·카테고리별 합계와 전월 대비 성장률을 5분 만에 뽑습니다. (level04·07·09)
- **데이터 검수**: 거래 로그에 비어 있는 값과 음수 금액이 섞여 들어왔을 때, 몇 건이 오염됐는지 진단하고 처리 방침을 정합니다. (level06)
- **목표 관리**: 매출표와 지점 정보표, 목표표를 합쳐 지점별 목표 달성률 순위를 만듭니다. (level08·10)
- **자동화**: 매주 반복하던 엑셀 수작업을 스크립트 하나로 바꿔, 같은 결과가 매번 재현되게 만듭니다. (level01·11)

## 실행 방법

각 레벨 폴더에서 아래처럼 실행합니다. 그림·CSV 같은 산출물은 각 레벨의 `outputs/` 폴더에 생성됩니다.

```bash
cd lecture03_data_handling/level03_pandas_dataframe
python3 main.py
```
