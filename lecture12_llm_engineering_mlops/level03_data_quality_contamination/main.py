"""
데이터 품질·중복 제거·오염 검사 실습.
- 문자 n-gram(슁글) 집합과 자카드 유사도로 완전/근사 중복 문서를 탐지·제거합니다.
- 벤치마크 문항이 학습 코퍼스에 섞인 '오염(시험문제 유출)'을 n-gram 겹침으로 검사하고,
  오염이 벤치마크 점수를 얼마나 부풀리는지 시뮬레이션으로 확인합니다.
"""
import random
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

DUP_THRESHOLD = 0.7    # 이 값 이상이면 근사 중복으로 판정
NGRAM_N = 5            # 문자 n-gram 길이
CONTAM_N = 10          # 오염 검사용 n-gram 길이 (길수록 '통째 유출'만 잡음)
CONTAM_THRESHOLD = 0.3 # 문항 n-gram 중 이 비율 이상이 코퍼스에 있으면 유출 의심


def char_ngrams(text: str, n: int) -> set:
    """문자열을 길이 n의 연속 조각(문자 n-gram) 집합으로 자른다."""
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """자카드 유사도 = |교집합| / |합집합|. 완전 동일 1.0, 무관 0.0."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def contamination_ratio(question: str, corpus_grams: set, n: int) -> float:
    """문항의 n-gram 중 학습 코퍼스에 그대로 존재하는 비율."""
    q_grams = char_ngrams(question, n)
    if not q_grams:
        return 0.0
    return sum(1 for g in q_grams if g in corpus_grams) / len(q_grams)


def build_documents(rng: random.Random):
    """tiny_corpus 문장으로 문서 24개를 만들고 중복/유출 사례를 일부러 심는다."""
    sentences = [s.strip() + "." for s in hjh_data.tiny_corpus().split(". ") if s.strip()]
    docs = []
    for i in range(24):
        picked = rng.sample(sentences, 3)          # 문장 3개 = 문서 1개
        docs.append(" ".join(picked))
    docs[15] = docs[2]                             # 완전 복사본 심기
    docs[19] = docs[5].replace("만들었다", "완성했다", 1)  # 한 단어만 바꾼 근사 복사본
    return docs


BENCHMARK = [  # (문항, 정답) — 4지선다 상식 시험이라고 가정
    ("대한민국의 수도는 어디인가? 보기: 부산, 서울, 대전, 광주", "서울"),
    ("물이 끓는 섭씨 온도는 몇 도인가? 보기: 50, 80, 100, 120", "100"),
    ("일주일은 며칠인가? 보기: 5일, 6일, 7일, 8일", "7일"),
    ("삼각형의 내각의 합은 몇 도인가? 보기: 90, 180, 270, 360", "180"),
    ("빛과 소리 중 더 빠른 것은? 보기: 빛, 소리, 같다, 알수없다", "빛"),
    ("1년은 대략 며칠인가? 보기: 300일, 330일, 365일, 400일", "365일"),
]
LEAKED_IDX = 3  # 이 문항을 학습 문서에 유출시킨다


def memorizer_score(corpus_text: str, items, rng: random.Random):
    """'외운 것만 맞히는' 가상 모델: 문항이 코퍼스에 통째로 있으면 정답,
    아니면 4지선다 찍기(정답률 25%). 반환: 문항별 정오 리스트."""
    results = []
    for question, _answer in items:
        seen = question in corpus_text          # 족보에서 본 문제인가?
        correct = True if seen else (rng.random() < 0.25)
        results.append((seen, correct))
    return results


def main():
    rng = random.Random(42)  # 재현성을 위한 seed 고정

    # [1] 코퍼스 구성 -------------------------------------------------------
    docs = build_documents(rng)
    q_leak, a_leak = BENCHMARK[LEAKED_IDX]
    docs[9] = docs[9] + f" 오늘의 상식 퀴즈. {q_leak} 정답은 {a_leak}." # 유출 문서
    print("[1] 학습 코퍼스 구성: 문서", len(docs), "개")
    print("    - 심어 둔 문제: 완전 복사본(2↔15), 근사 복사본(5↔19), 벤치마크 유출(문서 9)")
    print("    - 문서 예시:", docs[0][:44], "...")

    # [2] 근사 중복 탐지 ----------------------------------------------------
    grams = [char_ngrams(d, NGRAM_N) for d in docs]
    dup_pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = jaccard(grams[i], grams[j])
            if sim >= DUP_THRESHOLD:
                dup_pairs.append((i, j, sim))
    print(f"\n[2] 중복 탐지 (문자 {NGRAM_N}-gram 자카드 >= {DUP_THRESHOLD})")
    for i, j, sim in dup_pairs:
        kind = "완전 중복" if sim > 0.999 else "근사 중복"
        print(f"    - 문서 {i:2d} ↔ 문서 {j:2d} : 자카드 {sim:.3f}  → {kind}")
    if not dup_pairs:
        print("    - 발견된 중복 없음")

    # [3] 중복 제거 ---------------------------------------------------------
    drop = {j for _i, j, _s in dup_pairs}       # 뒤에 등장한 쪽을 버린다
    kept = [d for k, d in enumerate(docs) if k not in drop]
    print(f"\n[3] 중복 제거: {len(docs)}개 → {len(kept)}개 (버린 문서: {sorted(drop)})")
    print("    낭비될 뻔한 학습 토큰을 아끼고, 특정 문서 암기 위험을 줄였습니다.")

    # [4] 벤치마크 오염 검사 ------------------------------------------------
    corpus_text = " ".join(kept)
    corpus_grams = char_ngrams(corpus_text, CONTAM_N)
    print(f"\n[4] 오염 검사 (문항 {CONTAM_N}-gram이 코퍼스에 존재하는 비율 >= {CONTAM_THRESHOLD:.0%})")
    contaminated = []
    for idx, (question, _a) in enumerate(BENCHMARK):
        ratio = contamination_ratio(question, corpus_grams, CONTAM_N)
        flag = ratio >= CONTAM_THRESHOLD
        if flag:
            contaminated.append(idx)
        mark = "★유출 의심" if flag else "깨끗"
        print(f"    - 문항 {idx}: 겹침 {ratio:5.1%}  [{mark}]  {question[:26]}...")

    # [5] 오염이 점수를 부풀리는 시연 ----------------------------------------
    print("\n[5] '외운 것만 맞히는' 가상 모델의 벤치마크 점수 비교")
    results = memorizer_score(corpus_text, BENCHMARK, random.Random(6))  # 찍기 전용 seed
    total = len(BENCHMARK)
    score_all = sum(c for _s, c in results) / total
    clean_items = [r for k, r in enumerate(results) if k not in contaminated]
    score_clean = sum(c for _s, c in clean_items) / max(1, len(clean_items))
    for k, (seen, correct) in enumerate(results):
        note = "족보에서 봄 → 자동 정답" if seen else ("찍어서 정답" if correct else "오답")
        print(f"    - 문항 {k}: {'O' if correct else 'X'}  ({note})")
    print(f"    오염 포함 점수 : {score_all:.1%}  ← 발표 자료에 실리기 쉬운 숫자")
    print(f"    클린 점수     : {score_clean:.1%}  ← 실제 실력에 가까운 숫자")
    print("    → 유출 문항 하나가 점수를 부풀립니다. '이 점수는 오염 검사를 거쳤나요?'라고 물어보세요.")


if __name__ == "__main__":
    main()
