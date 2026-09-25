"""
검색 품질 올리기 3종 세트를 직접 구현해 전후를 비교합니다.
1) BM25: 고전 키워드 검색의 표준 공식을 밑바닥부터 구현
2) 하이브리드: BM25(정확한 단어) + 임베딩(의미)을 점수 결합
3) MMR: 비슷한 조각만 잔뜩 뽑히는 중복 문제를 다양성으로 해결
RAG 의 답변 품질은 결국 '무엇을 검색해 왔는가'가 좌우합니다.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, split_sentences, tokenize

# 중복 문제 재현용: 규정과 거의 같은 내용의 사내 공지가 함께 색인된 상황
EXTRA_DOCS = {
    "사내공지_연차안내.txt": (
        "연차휴가는 1년 근속 시 15일이 부여됩니다. "
        "연차 신청은 최소 3영업일 전에 결재 시스템으로 해 주세요. "
        "연차휴가 신청 방법은 인트라넷 공지를 참고하세요."
    ),
}


class BM25:
    """BM25 키워드 검색기 (직접 구현).
    아이디어: 질문 단어가 (1) 그 문서에 자주 나올수록(TF, 포화 있음)
    (2) 전체에서 희귀한 단어일수록(IDF) (3) 짧은 문서일수록 가산점."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = np.array([len(t) for t in self.doc_tokens])
        self.avg_len = float(self.doc_len.mean())
        self.n_docs = len(docs)
        self.df: dict[str, int] = {}                 # 단어 -> 등장 문서 수
        for tokens in self.doc_tokens:
            for w in set(tokens):
                self.df[w] = self.df.get(w, 0) + 1

    def _idf(self, word: str) -> float:
        n = self.df.get(word, 0)
        return math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs)
        for w in tokenize(query):
            idf = self._idf(w)
            for i, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(w)
                if tf == 0:
                    continue
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out


def minmax(x: np.ndarray) -> np.ndarray:
    """점수 눈금 통일: 0~1 로 정규화 (결합 전 필수)."""
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def mmr_select(qv: np.ndarray, vectors: np.ndarray, base_scores: np.ndarray,
               k: int = 4, lam: float = 0.6) -> list[int]:
    """MMR(Maximal Marginal Relevance):
    관련성(lam)과 '이미 뽑은 것과 안 겹침'(1-lam)을 저울질하며 하나씩 선택."""
    selected: list[int] = []
    candidates = list(np.argsort(-base_scores)[: k * 3])   # 상위 후보만
    while candidates and len(selected) < k:
        best_i, best_val = None, -1e9
        for i in candidates:
            redundancy = max((float(vectors[i] @ vectors[j]) for j in selected),
                             default=0.0)
            val = lam * base_scores[i] - (1 - lam) * redundancy
            if val > best_val:
                best_i, best_val = i, val
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


def show(title: str, idx_scores, corpus) -> None:
    print(f"    {title}")
    for rank, (i, s) in enumerate(idx_scores, 1):
        src, text = corpus[i]
        print(f"      {rank}위 {s:5.3f} {text[:32]}... ({src})")


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 09 | 검색 품질 개선 — BM25·하이브리드·MMR")
    print("=" * 62)

    # [1] 코퍼스: 사내 문서 + 내용이 겹치는 공지 (중복 상황 재현)
    corpus = []
    for name, doc in {**hjh_data.SAMPLE_DOCS, **EXTRA_DOCS}.items():
        corpus += [(name, s) for s in split_sentences(doc)]
    texts = [t for _, t in corpus]
    print(f"\n[1] 코퍼스: 문서 {len(hjh_data.SAMPLE_DOCS) + 1}건 -> 조각 {len(corpus)}개")
    print("    (연차 규정과 내용이 겹치는 '사내공지'가 섞여 있는 실전형 상황)")

    # [2] 두 검색기 준비
    bm25 = BM25(texts)
    emb = MockEmbedding(dim=512).fit(texts)
    vectors = emb.embed_batch(texts)

    # [3] BM25 작동 확인: 명사 나열형 검색 질의에 강하다
    q1 = "미사용 연차 수당 정산"
    print(f"\n[2] BM25 작동 확인: \"{q1}\" (검색창에 치는 명사 나열형)")
    b1 = bm25.scores(q1)
    show("BM25 top-2", [(i, float(b1[i])) for i in np.argsort(-b1)[:2]], corpus)
    print("    -> 질문 단어가 문서에 그대로 있으면 BM25 는 정확하고 빠릅니다.")

    # [4] 두 검색기의 서로 다른 실패 모드
    q2 = "코어타임 시간을 알려주세요"          # 문서엔 '코어타임인' -> 단어 불일치
    q3 = "출장 일비 기준"                      # 흔한 단어 '기준'이 BM25 를 유인
    print(f"\n[3] 실패 모드 비교")
    for q, note in ((q2, "문서 표기는 '코어타임인' — 조사가 붙어 단어 일치 실패"),
                    (q3, "'기준'은 여러 문서에 흔한 단어 — 키워드가 엉뚱한 곳에 낚임")):
        b = bm25.scores(q)
        s = vectors @ emb.embed(q)
        print(f"  질문: \"{q}\"  ({note})")
        show("(a) BM25 키워드", [(i, float(b[i])) for i in np.argsort(-b)[:1]], corpus)
        show("(b) 임베딩 의미", [(i, float(s[i])) for i in np.argsort(-s)[:1]], corpus)

    # [5] 하이브리드: 두 점수를 0~1 정규화 후 가중 합
    alpha = 0.5
    answers = {q1: "수당으로 정산", q2: "코어타임", q3: "국내 출장 일비"}
    print(f"\n[4] 하이브리드 = {alpha} x BM25 + {1 - alpha} x 의미 (각각 0~1 정규화 후 결합)")
    print(f"    {'질문':<22}{'BM25':>6}{'의미':>6}{'하이브리드':>9}   (1위가 정답이면 O)")
    for q, span in answers.items():
        b, s = bm25.scores(q), vectors @ emb.embed(q)
        hybrid = alpha * minmax(b) + (1 - alpha) * minmax(s)
        marks = ["O" if span in corpus[int(np.argmax(x))][1] else "X"
                 for x in (b, s, hybrid)]
        print(f"    {q:<24}{marks[0]:>4}{marks[1]:>6}{marks[2]:>9}")
    print("    -> 실패 모드가 서로 달라, 합치면 세 질문 모두에서 정답이 1위가 됩니다.")

    # [6] MMR: 중복 줄이기 전후 비교
    q3 = "연차 휴가 규정을 알려주세요"
    print(f"\n[5] MMR 전후 비교: \"{q3}\" (top-4)")
    s3 = vectors @ emb.embed(q3)
    plain = np.argsort(-s3)[:4]
    show("(a) 유사도 순 top-4 (중복 많음)", [(i, float(s3[i])) for i in plain], corpus)
    picked = mmr_select(emb.embed(q3), vectors, s3, k=4, lam=0.5)
    show("(b) MMR top-4 (다양성 확보)", [(i, float(s3[i])) for i in picked], corpus)
    print("    -> (a)의 1·2위는 사실상 같은 문장(신청 기한)이라 한 자리를 낭비하지만,")
    print("       (b)는 중복을 버리고 '15일 부여'라는 새 정보를 문맥에 확보합니다.")
    print("       LLM 에 넘길 문맥 예산(top-k)이 같다면 (b)가 더 풍부한 답을 만듭니다.")

    # [7] 리랭킹 개념 소개
    print("\n[6] 한 걸음 더: 리랭킹(re-ranking)")
    print("    1차 검색(빠른 하이브리드)으로 후보 20~50개를 추린 뒤,")
    print("    더 정밀한 모델(크로스 인코더, 또는 LLM 자체)로 질문-조각 쌍을")
    print("    다시 채점해 최종 top-k 를 고르는 2단계 구조가 실무 표준입니다.")
    print("    '서류 전형(빠름) -> 면접(정밀)' 의 채용 절차와 같은 원리입니다.")

    print("\n요약: 키워드와 의미는 경쟁자가 아니라 보완재(하이브리드),")
    print("      그리고 top-k 는 '관련 있고 + 서로 다른' 조각으로 채우세요(MMR).")


if __name__ == "__main__":
    main()
