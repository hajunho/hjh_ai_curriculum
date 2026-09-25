"""
level04 — BoW 와 TF-IDF

문서-단어 행렬(BoW)과 TF-IDF 를 numpy 로 직접 구현하고,
sklearn TfidfVectorizer 결과와 소수점 12자리까지 일치함을 검증합니다.
마지막으로 SAMPLE_DOCS(사내규정·매뉴얼)에서 문서별 핵심 단어를 추출하고
문서 간 코사인 유사도를 계산합니다.
"""

import re
import sys
import pathlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def tokenize(text: str) -> list[str]:
    """한글/영문/숫자 2글자 이상 어절만 추출 (sklearn 기본 패턴과 동일한 규칙)."""
    return re.findall(r"(?u)\b\w\w+\b", text.lower())


# ---------------------------------------------------------------------------
# [1] BoW — 문서-단어 행렬을 손으로
# ---------------------------------------------------------------------------

def build_bow(docs: list[str]) -> tuple[np.ndarray, list[str]]:
    """문서 리스트 -> (등장 횟수 행렬, 어휘 리스트). 어휘는 가나다순 정렬."""
    vocab = sorted({tok for d in docs for tok in tokenize(d)})
    index = {w: i for i, w in enumerate(vocab)}
    counts = np.zeros((len(docs), len(vocab)), dtype=float)
    for row, doc in enumerate(docs):
        for tok in tokenize(doc):
            counts[row, index[tok]] += 1
    return counts, vocab


# ---------------------------------------------------------------------------
# [2] TF-IDF — sklearn 기본형과 같은 공식을 numpy 로
#     idf(t) = ln((1+N) / (1+df(t))) + 1   (스무딩)
#     tfidf  = tf * idf 를 문서별 L2 정규화
# ---------------------------------------------------------------------------

def tfidf_numpy(counts: np.ndarray) -> np.ndarray:
    n_docs = counts.shape[0]
    df = (counts > 0).sum(axis=0)                       # 단어별 등장 문서 수
    idf = np.log((1 + n_docs) / (1 + df)) + 1           # 스무딩 IDF
    weighted = counts * idf                             # TF x IDF
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return weighted / norms                             # 문서별 L2 정규화


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

MINI_DOCS = [
    "배송 빠르고 배송 기사님 친절",
    "포장 꼼꼼 가격 만족",
    "배송 느림 가격 불만",
]


def demo_bow() -> None:
    print("[1] BoW: 문서를 갈아서 '단어 성분표'로 만들기")
    counts, vocab = build_bow(MINI_DOCS)
    header = "         " + " ".join(f"{w:>4s}" for w in vocab)
    print(header)
    for i, row in enumerate(counts):
        cells = " ".join(f"{int(v):4d}" for v in row)
        print(f"    문서{i} {cells}   <- {MINI_DOCS[i]!r}")
    sparsity = (counts == 0).mean()
    print(f"    -> 행렬 {counts.shape}, 0의 비율 {sparsity:.0%} (희소 행렬)\n")


def demo_verify() -> None:
    print("[2]-[3] TF-IDF 직접 구현 vs sklearn 대조 검증")
    docs = [r["text"] for r in hjh_data.review_corpus(80, seed=3)]
    counts, vocab = build_bow(docs)
    mine = tfidf_numpy(counts)

    vec = TfidfVectorizer()                              # 기본 설정: 스무딩+L2
    theirs = vec.fit_transform(docs).toarray()
    their_vocab = vec.get_feature_names_out().tolist()

    assert vocab == their_vocab, "어휘 사전 불일치"
    max_err = float(np.abs(mine - theirs).max())
    print(f"    문서 {len(docs)}건, 어휘 {len(vocab)}종")
    print(f"    직접 구현과 sklearn 의 최대 오차: {max_err:.2e}")
    ok = max_err < 1e-10
    print(f"    -> {'일치! 라이브러리 내부를 정확히 재현했습니다.' if ok else '불일치 — 공식 확인 필요'}\n")
    assert ok


def demo_keywords() -> np.ndarray:
    print("[4] SAMPLE_DOCS 문서별 핵심 단어 top5 (TF-IDF 기준)")
    names = list(hjh_data.SAMPLE_DOCS.keys())
    docs = list(hjh_data.SAMPLE_DOCS.values())
    counts, vocab = build_bow(docs)
    scores = tfidf_numpy(counts)
    for i, name in enumerate(names):
        top = np.argsort(scores[i])[::-1][:5]
        words = [f"{vocab[j]}({scores[i, j]:.2f})" for j in top]
        print(f"    {name:16s}: {', '.join(words)}")
    print("    -> 휴가 문서엔 연차, 경비 문서엔 출장·정산 — 문서의 '얼굴'이 뽑힙니다.\n")
    return scores


def demo_similarity(scores: np.ndarray) -> None:
    print("[5] 문서 간 코사인 유사도 (L2 정규화 벡터의 내적)")
    names = [n.replace(".txt", "") for n in hjh_data.SAMPLE_DOCS.keys()]
    sim = scores @ scores.T
    print("    " + " " * 14 + "  ".join(f"{n[:6]:>6s}" for n in names))
    for i, name in enumerate(names):
        cells = "  ".join(f"{sim[i, j]:6.2f}" for j in range(len(names)))
        print(f"    {name:14s}{cells}")
    # 자기 자신 제외 최대 유사도 쌍
    mask = sim - np.eye(len(names))
    a, b = np.unravel_index(np.argmax(mask), mask.shape)
    print(f"    -> 가장 비슷한 쌍: {names[a]} <-> {names[b]} ({mask[a, b]:.2f})")
    print("       전체적으로 유사도가 낮은 이유: 문서끼리 '같은 단어'를 거의 안 씁니다.")
    print("       TF-IDF 는 글자가 같아야만 닮음을 인정합니다 — '뜻이 비슷한 다른 단어'를")
    print("       알아보는 방법이 level06 워드 임베딩입니다.")


if __name__ == "__main__":
    print("=" * 70)
    print("BoW 와 TF-IDF — 텍스트를 숫자 벡터로, 직접 구현으로 검증까지")
    print("=" * 70 + "\n")
    np.random.seed(0)          # 이 레벨은 난수를 안 쓰지만 관례적으로 고정
    demo_bow()
    demo_verify()
    scores = demo_keywords()
    demo_similarity(scores)
