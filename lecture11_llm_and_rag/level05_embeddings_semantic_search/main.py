"""
임베딩(embedding): 문장을 숫자 좌표(벡터)로 바꿔 '의미'로 검색하는 원리.
1) 사내 문서 문장들을 MockEmbedding 으로 벡터화
2) 같은 주제/다른 주제 문장의 코사인 유사도 비교
3) 질문으로 의미 검색 vs 단어 일치 키워드 검색 결과 비교
4) 임베딩 공간을 2차원 지도로 그려 outputs/ 에 저장
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, cosine, split_sentences, tokenize

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def keyword_search(query: str, corpus: list[tuple[str, str]], k: int = 3):
    """고전적 키워드 검색: 질문의 '단어'가 문장에 그대로 있는 개수로 점수."""
    q_tokens = [w for w in tokenize(query) if len(w) >= 2]
    results = []
    for source, sent in corpus:
        s_tokens = set(tokenize(sent))
        hits = [w for w in q_tokens if w in s_tokens]   # 완전 일치만 인정
        results.append((len(hits), hits, source, sent))
    results.sort(key=lambda x: -x[0])
    return results[:k]


def semantic_search(query: str, emb: MockEmbedding, vectors: np.ndarray,
                    corpus: list[tuple[str, str]], k: int = 3):
    """의미 검색: 질문 벡터와 모든 문장 벡터의 코사인 유사도 순위."""
    qv = emb.embed(query)
    sims = vectors @ qv                       # 정규화된 벡터라 내적 = 코사인
    order = np.argsort(-sims)[:k]
    return [(float(sims[i]), corpus[i][0], corpus[i][1]) for i in order]


def main() -> None:
    np.random.seed(42)                        # 재현성

    print("=" * 62)
    print("Level 05 | 임베딩과 의미 검색")
    print("=" * 62)

    # [1] 사내 문서를 문장 단위로 준비
    corpus = []                               # (출처 파일명, 문장)
    for name, doc in hjh_data.SAMPLE_DOCS.items():
        corpus += [(name, s) for s in split_sentences(doc)]
    print(f"\n[1] 문서 {len(hjh_data.SAMPLE_DOCS)}건 -> 문장 {len(corpus)}개 로 분해")

    # [2] 임베딩: 문장 -> 512차원 좌표
    emb = MockEmbedding(dim=512).fit([s for _, s in corpus])
    vectors = emb.embed_batch([s for _, s in corpus])
    sample_vec = vectors[0]
    nz = np.nonzero(sample_vec)[0][:5]
    print(f"\n[2] 임베딩 완료: 문장마다 {vectors.shape[1]}차원 벡터 (모두 길이 1로 정규화)")
    print(f"    예: \"{corpus[0][1][:30]}...\"")
    print("    -> " + ", ".join(f"{i}번 칸={sample_vec[i]:.3f}" for i in nz) +
          f", ... (0이 아닌 칸 {int((sample_vec != 0).sum())}개 / {vectors.shape[1]}칸)")

    # [3] 좌표가 가깝다 = 의미가 비슷하다
    print("\n[3] 코사인 유사도: 1에 가까울수록 같은 주제")
    pairs = [
        ("연차 휴가 며칠 쓸 수 있나요", "제1조 연차휴가는 입사일 기준 1년 근속 시 15일이 부여된다."),
        ("연차 휴가 며칠 쓸 수 있나요", "제품 보증 기간은 구매일로부터 2년입니다."),
        ("보증 기간이 궁금합니다", "제품 보증 기간은 구매일로부터 2년입니다."),
    ]
    for a, b in pairs:
        sim = cosine(emb.embed(a), emb.embed(b))
        print(f"    {sim:.3f}  \"{a}\"  vs  \"{b[:28]}...\"")

    # [4] 의미 검색 vs 키워드 검색
    queries = [
        "재택으로 일할 수 있는 날은 몇 번인가요?",   # '재택근무'와 표기가 달라 키워드가 놓침
        "경비 영수증은 언제까지 내야 하나요?",
    ]
    print("\n[4] 같은 질문, 두 가지 검색")
    for q in queries:
        print(f"\n  질문: \"{q}\"")
        print("  (a) 키워드 검색 (단어 완전 일치)")
        for score, hits, source, sent in keyword_search(q, corpus):
            mark = ", ".join(hits) if hits else "일치 단어 없음"
            print(f"      일치 {score}개 [{mark}] {sent[:34]}... ({source})")
        print("  (b) 의미 검색 (임베딩 코사인)")
        for sim, source, sent in semantic_search(q, emb, vectors, corpus):
            print(f"      {sim:.3f} {sent[:34]}... ({source})")
    print("\n    -> '재택으로'처럼 표기가 조금만 달라져도 키워드 검색은 놓치지만,")
    print("       임베딩은 문자 조각과 동시출현 정보로 '재택근무' 문장을 찾아냅니다.")

    # [5] 임베딩 공간 2차원 지도 (SVD 로 차원 축소)
    centered = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    # 범례는 폰트 문제를 피해 영문으로 표기 (사내규정_휴가 -> vacation ...)
    label_en = {"사내규정_휴가.txt": "policy: vacation", "사내규정_경비.txt": "policy: expense",
                "사내규정_재택.txt": "policy: remote", "제품매뉴얼_설치.txt": "manual: install",
                "제품매뉴얼_보증.txt": "manual: warranty"}
    plt.figure(figsize=(8, 6))
    for name in hjh_data.SAMPLE_DOCS.keys():
        idx = [i for i, (src, _) in enumerate(corpus) if src == name]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=label_en.get(name, name), s=60)
    plt.legend(fontsize=8)
    plt.title("Sentence embedding map (SVD 2D)")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "embedding_map.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\n[5] 임베딩 지도 저장 -> {out_path}")
    print("    같은 문서(주제)의 문장들이 서로 뭉쳐 있는지 확인해 보세요.")

    print("\n요약: 임베딩은 '뜻이 비슷하면 좌표도 가깝다'는 지도를 만드는 기술입니다.")
    print("      단어가 달라도 의미로 찾는 검색 — RAG 의 첫 번째 부품입니다.")

    # ------------------------------------------------------------------
    # [참고] 실제 서비스라면 임베딩 전용 모델 API 를 씁니다. 예:
    # response = embeddings_client.create(model="...", input=["문장1", "문장2"])
    # vec = response.data[0].embedding        # 보통 1024~3072차원 실수 벡터
    # 사용법(코사인 유사도로 순위 매기기)은 오늘 코드와 완전히 같습니다.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
