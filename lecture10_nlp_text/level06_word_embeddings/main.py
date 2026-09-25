"""
level06 — 워드 임베딩

tiny_corpus(한국어 미니 말뭉치)에서 동시출현 행렬을 만들고
PPMI 변환 + SVD 압축으로 워드 임베딩을 '직접' 학습합니다.
유사 단어 검색과 2차원 임베딩 지도 PNG 저장까지 시연합니다.
word2vec 같은 전용 라이브러리 없이 numpy 만 사용합니다.
"""

import os
import sys
import pathlib
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
JOSA = ["이", "가", "을", "를", "에"]


def set_korean_font() -> None:
    """그림에 한글이 깨지지 않도록 시스템에 있는 한글 폰트를 찾아 설정합니다."""
    candidates = ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def tokenize(corpus: str) -> list[str]:
    """마침표 제거 + 어절 끝 조사(이/가/을/를/에) 간이 분리."""
    tokens = []
    for word in corpus.replace(".", " ").split():
        for josa in JOSA:
            if word.endswith(josa) and len(word) - len(josa) >= 2:
                word = word[: -len(josa)]
                break
        tokens.append(word)
    return tokens


def build_cooccurrence(tokens: list[str], vocab: list[str], window: int = 2):
    """[2] 창 크기 window 안에서 함께 나온 횟수를 세는 '친구 관계 장부'."""
    index = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                C[index[center], index[tokens[j]]] += 1
    return C


def ppmi(C: np.ndarray) -> np.ndarray:
    """[3] 양의 점별 상호정보량 — '우연 이상의 친분'만 남기기."""
    total = C.sum()
    row = C.sum(axis=1, keepdims=True)
    col = C.sum(axis=0, keepdims=True)
    expected = row @ col / total                 # 우연히 함께 나올 기대 횟수
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(C * total / (row * col))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0)


def nearest(word: str, emb: np.ndarray, vocab: list[str], k: int = 4):
    """[5] 코사인 유사도 기준 최근접 이웃 k개."""
    index = {w: i for i, w in enumerate(vocab)}
    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = normed @ normed[index[word]]
    order = np.argsort(sims)[::-1]
    return [(vocab[i], float(sims[i])) for i in order if vocab[i] != word][:k]


if __name__ == "__main__":
    print("=" * 70)
    print("워드 임베딩 — 동시출현 + PPMI + SVD 로 단어 지도 만들기")
    print("=" * 70 + "\n")
    np.random.seed(0)
    set_korean_font()

    # [1] 말뭉치 준비 ------------------------------------------------------
    corpus = hjh_data.tiny_corpus()
    tokens = tokenize(corpus)
    freq = Counter(tokens)
    vocab = sorted(freq)
    print(f"[1] 말뭉치: {len(corpus):,}자 -> 토큰 {len(tokens):,}개, 어휘 {len(vocab)}종")
    print(f"    빈도 상위: {freq.most_common(5)}\n")

    # [2] 동시출현 행렬 ----------------------------------------------------
    C = build_cooccurrence(tokens, vocab, window=2)
    print(f"[2] 동시출현 행렬 {C.shape} (창 크기 2)")
    idx = {w: i for i, w in enumerate(vocab)}
    student_row = C[idx["학생"]]
    top = np.argsort(student_row)[::-1][:5]
    print("    '학생'의 주요 동석자:",
          ", ".join(f"{vocab[i]}({int(student_row[i])}회)" for i in top))
    print("    -> 문형이 '시간 주어 목적어 동사'라 주어 앞뒤 창에는 동사가 자주 들어옵니다.")
    print("       이 '동석 프로필'이 곧 단어의 지문이 됩니다.\n")

    # [3] PPMI -------------------------------------------------------------
    P = ppmi(C)
    common_word = freq.most_common(1)[0][0]
    print(f"[3] PPMI 변환: 우연한 동석(흔한 단어)의 점수를 깎습니다")
    print(f"    예: '학생'-'{common_word}' 동시출현 {int(C[idx['학생'], idx[common_word]])}회지만 "
          f"PPMI {P[idx['학생'], idx[common_word]]:.2f}")
    print(f"    -> 횟수가 많아도 '누구와나 함께 나오는' 조합은 정보가 아닙니다.\n")

    # [4] SVD 압축 ---------------------------------------------------------
    U, S, Vt = np.linalg.svd(P)
    dim = 8
    emb = U[:, :dim] * S[:dim]                   # 상위 dim 축만 남긴 좌표
    explained = S[:dim].sum() / S.sum()
    print(f"[4] SVD 압축: {P.shape[1]}차원 -> {dim}차원 임베딩")
    print(f"    상위 {dim}개 특이값이 전체 정보의 {explained:.0%}를 담습니다.\n")

    # [5] 유사 단어 검색 ---------------------------------------------------
    print("[5] 유사 단어 검색 (코사인 유사도)")
    for query in ["학생", "보고서", "만들었다", "어제"]:
        pairs = ", ".join(f"{w}({s:.2f})" for w, s in nearest(query, emb, vocab))
        print(f"    {query:6s} 의 이웃: {pairs}")
    print("    -> 주어끼리·목적어끼리·동사끼리 모입니다. '뜻(역할)이 비슷하면 가깝다'!\n")

    # [6] 임베딩 지도 PNG --------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    coords = U[:, :2] * S[:2]                    # 그림용 2차원 좌표
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=28, color="#3b6db4")
    for i, w in enumerate(vocab):
        ax.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=9,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title("Word embedding map (co-occurrence + PPMI + SVD)")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "embedding_map.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[6] 임베딩 지도 저장: {path}")
    print("    그림에서 명사·동사·시간 표현이 각자 영역을 이루는지 확인해 보세요.")
    print("\n결론: '단어의 뜻은 어울리는 친구가 정한다'를 통계+선형대수로 구현했습니다.")
