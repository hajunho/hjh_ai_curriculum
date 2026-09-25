# Lecture 04 — 데이터베이스와 SQL

> "데이터는 회사의 장부입니다. SQL은 그 장부에게 말을 거는 언어입니다."

엑셀 파일을 주고받으며 일하던 사람이, 회사 데이터베이스(Database)에 직접 질문을
던질 수 있는 사람으로 성장하는 강의입니다. 모든 실습은 파이썬에 내장된
SQLite(에스큐라이트)를 사용하므로 **아무것도 설치할 필요가 없습니다.**
각 레벨 폴더에서 `main.py` 를 실행하면 실습용 쇼핑몰 데이터베이스
`hjh_shop.db` 가 자동으로 만들어지고, SQL 문장과 실행 결과가 나란히 출력됩니다.

이 강의의 주인공은 SQL 입니다. 파이썬은 SQL 을 실행해 주는 "실행기" 역할만
합니다. 코드보다 **SQL 문장 자체**를 읽고 따라 쓰는 데 집중해 보세요.

## 이 강의에서 배우는 것

- 데이터베이스가 엑셀 공유보다 나은 이유 (동시 수정, 정합성, 권한)
- 테이블·기본키·외래키로 데이터가 어떻게 연결되는지
- SELECT / WHERE / ORDER BY / GROUP BY / JOIN / 서브쿼리 — 실무 질문을 SQL 로 바꾸는 법
- INSERT / UPDATE / DELETE 와 트랜잭션 — 데이터를 안전하게 바꾸는 법
- 인덱스와 실행 계획 — 느린 쿼리를 빠르게 만드는 원리
- 윈도우 함수 — 순위·누적·이동합계 같은 분석 쿼리
- 파이썬 + pandas 로 SQL 결과를 리포트까지 자동화하는 미니 파이프라인

## 선행 강의

- **lecture02 (파이썬 기초)** — main.py 를 읽고 실행할 수 있을 정도면 충분합니다.
- **lecture03 (NumPy·Pandas)** — level11 에서 pandas 를 잠깐 쓰지만, 몰라도 진행 가능합니다.

## 레벨 목차

| 레벨 | 제목 | 난이도 |
|---|---|---|
| [level00](level00_why_databases/README.md) | 데이터베이스가 왜 필요한가 | ⭐ |
| [level01](level01_tables_keys/README.md) | 테이블·행·열·기본키·외래키 | ⭐ |
| [level02](level02_select_basics/README.md) | SELECT 기초 | ⭐ |
| [level03](level03_where_filtering/README.md) | WHERE — 조건 검색 | ⭐⭐ |
| [level04](level04_order_limit_distinct/README.md) | 정렬·중복 제거·상위 N | ⭐⭐ |
| [level05](level05_aggregate_groupby/README.md) | 집계 함수와 GROUP BY | ⭐⭐ |
| [level06](level06_joins/README.md) | JOIN — 여러 테이블 연결 | ⭐⭐⭐ |
| [level07](level07_subqueries/README.md) | 서브쿼리와 CTE | ⭐⭐⭐ |
| [level08](level08_dml_transactions/README.md) | 데이터 변경과 트랜잭션 | ⭐⭐⭐ |
| [level09](level09_indexes_performance/README.md) | 인덱스와 쿼리 성능 | ⭐⭐⭐⭐ |
| [level10](level10_window_functions/README.md) | 윈도우 함수와 분석 쿼리 | ⭐⭐⭐⭐ |
| [level11](level11_python_db_pipeline/README.md) | 파이썬 연동과 데이터 파이프라인 | ⭐⭐⭐⭐ |

## 빠른 경로 (시간이 없다면 이 5개만)

1. **level02 — SELECT 기초**: 모든 SQL 의 출발점.
2. **level03 — WHERE**: "조건에 맞는 것만" 골라내기. 실무 질문의 80%.
3. **level05 — GROUP BY**: "도시별", "카테고리별" — 보고서의 언어.
4. **level06 — JOIN**: 여러 장부를 연결해야 진짜 분석이 됩니다.
5. **level11 — 파이썬 연동**: SQL 결과를 자동 리포트로 만드는 마무리.

## 실습 방법

```bash
cd lecture04_database_sql/level02_select_basics
python3 main.py
```

각 main.py 는 실행할 때마다 레벨 폴더 안에 `hjh_shop.db` 를 새로 만듭니다.
실수로 데이터를 망가뜨려도 다시 실행하면 원상복구되니 마음껏 실험하세요.
출력에는 항상 **SQL 문장이 먼저**, 그 아래에 결과 표가 나옵니다.
SQL 문장을 소리 내어 읽어 보는 것이 가장 좋은 복습입니다.

## 이 강의가 실무에서 쓰이는 장면

- **마케팅**: "지난 분기 VIP 고객 중 재구매가 없는 사람 명단 주세요" → WHERE + 서브쿼리
- **영업 관리**: "지점별·월별 매출 순위와 누적 달성률" → GROUP BY + 윈도우 함수
- **재고/운영**: "주문이 들어오면 재고를 차감하되, 부족하면 전체 취소" → 트랜잭션
- **데이터 분석**: 사내 DB에서 SQL 로 뽑은 데이터를 pandas 로 정리해 주간 리포트 자동 발송
- **개발 협업**: 개발자에게 "이 화면 왜 느려요?" 대신 "이 쿼리에 인덱스가 없는 것 같아요"라고 말하기

엑셀만 쓰던 시절에는 "데이터 좀 뽑아 주세요"라고 부탁하는 쪽이었다면,
이 강의를 마치면 **직접 질문하고 직접 답을 꺼내는 쪽**이 됩니다.
