"""
문서 로딩과 청킹(chunking): 문서를 어떤 단위로 잘라 검색할 것인가.
1) 사내 문서를 3가지 전략(고정 길이/고정+오버랩/문장 단위)으로 자름
2) 잘린 조각을 임베딩해 미니 검색기를 만들고
3) 같은 질문 5개로 '정답 구절이 든 조각을 찾았는지' 명중률을 비교
청크 크기·오버랩의 트레이드오프를 숫자로 확인합니다.
"""

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, split_sentences

# 평가셋: (질문, 정답이 반드시 포함해야 하는 구절)
EVAL_SET = [
    ("연차는 며칠 전에 신청해야 하나요?", "3영업일"),
    ("국내 출장 일비는 얼마인가요?", "3만원"),
    ("재택근무는 주 몇 회까지 가능한가요?", "주 2회"),
    ("제품 보증 기간은 얼마나 되나요?", "2년"),
    ("설치할 때 벽과 얼마나 띄워야 하나요?", "10센티미터"),
]


# ---------------------------------------------------------------------------
# 청킹 전략 3종
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, size: int = 50, overlap: int = 0) -> list[str]:
    """전략 A/B: 글자 수로 자르기 (overlap 만큼 앞 조각과 겹치게)."""
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def chunk_by_sentence(text: str, max_chars: int = 90) -> list[str]:
    """전략 C: 문장 경계를 지키며 max_chars 를 넘지 않게 묶기."""
    chunks, current = [], ""
    for sent in split_sentences(text):
        if current and len(current) + len(sent) + 1 > max_chars:
            chunks.append(current)
            current = sent
        else:
            current = (current + " " + sent).strip()
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# 평가: 청킹 전략별 미니 검색기 성능
# ---------------------------------------------------------------------------

def evaluate(chunks: list[tuple[str, str]], top_k: int = 2, verbose: bool = False):
    """조각들을 임베딩해 검색기를 만들고, 평가셋 명중률을 잽니다.
    명중 = 상위 top_k 조각 중 정답 구절을 포함한 조각이 있음."""
    texts = [c for _, c in chunks]
    emb = MockEmbedding(dim=512).fit(texts)
    vectors = emb.embed_batch(texts)
    hits = 0
    for question, answer_span in EVAL_SET:
        sims = vectors @ emb.embed(question)
        order = np.argsort(-sims)[:top_k]
        hit_rank = next((r for r, i in enumerate(order, 1) if answer_span in texts[i]), None)
        hits += hit_rank is not None
        if verbose:
            if hit_rank is None:
                print(f"      [실패] {question}")
                print(f"             1위 조각: \"{texts[order[0]][:44]}...\"")
            else:
                found_text = texts[order[hit_rank - 1]]
                print(f"      [명중/{hit_rank}위] {question}")
                print(f"             해당 조각: \"{found_text[:44]}...\"")
    return hits, len(EVAL_SET)


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 06 | 문서 로딩과 청킹 — 자르는 방법이 검색 품질을 좌우한다")
    print("=" * 62)

    # [1] 문서 로딩
    docs = hjh_data.SAMPLE_DOCS
    total_chars = sum(len(d) for d in docs.values())
    print(f"\n[1] 문서 로딩: {len(docs)}건, 총 {total_chars:,}자")
    print("    왜 자르나? (1) LLM 문맥창은 유한 (2) 통문서 임베딩은 주제가 섞여")
    print("    검색이 흐려짐 (3) 답변 근거는 '조각' 단위로 인용해야 정확함")

    # [2] 3가지 전략으로 자르기
    strategies = {
        "A. 고정 50자 (오버랩 0)": lambda t: chunk_fixed(t, size=50, overlap=0),
        "B. 고정 50자 + 오버랩 15": lambda t: chunk_fixed(t, size=50, overlap=15),
        "C. 문장 단위 (최대 90자)": lambda t: chunk_by_sentence(t, max_chars=90),
    }
    all_chunks: dict[str, list[tuple[str, str]]] = {}
    print("\n[2] 청킹 결과 요약")
    for label, fn in strategies.items():
        chunks = []
        for name, doc in docs.items():
            chunks += [(name, c) for c in fn(doc)]
        all_chunks[label] = chunks
        avg = np.mean([len(c) for _, c in chunks])
        print(f"    {label:<26} 조각 {len(chunks):>2}개, 평균 {avg:.0f}자")

    # [3] 전략 A 의 문제: 문장이 중간에서 뚝 끊긴다
    print("\n[3] 전략 A 가 만든 조각 예시 (문장이 잘려 뜻이 손상됨)")
    for _, c in all_chunks["A. 고정 50자 (오버랩 0)"][:3]:
        print(f"    | {c}")
    print("    -> '2년마다'가 '2년마'와 '다'로 쪼개지는 등 단어가 경계에서 끊깁니다.")

    # [4] 전략별 검색 명중률 비교
    print("\n[4] 검색 명중률 비교 (질문 5개, 상위 2조각 안에 정답 구절이 있으면 명중)")
    results = {}
    for label, chunks in all_chunks.items():
        hits, total = evaluate(chunks)
        results[label] = hits
        bar = "#" * hits + "." * (total - hits)
        print(f"    {label:<26} {hits}/{total}  [{bar}]")

    # [5] 가장 좋은 전략의 상세 결과
    best_label = max(results, key=results.get)
    print(f"\n[5] 최고 전략 '{best_label}' 상세")
    evaluate(all_chunks[best_label], verbose=True)

    print("\n[6] 트레이드오프 정리")
    print("    조각이 너무 작으면: 문장이 잘려 뜻 손상, 답이 여러 조각에 분산")
    print("    조각이 너무 크면 : 여러 주제가 섞여 검색이 흐려지고 토큰 비용 증가")
    print("    오버랩          : 경계에서 잘리는 정보를 보완 (저장량은 늘어남)")
    print("    실무 기본값     : 문장/문단 경계 존중 + 200~500토큰 + 10~20% 오버랩")

    print("\n요약: RAG 품질 문제의 절반은 모델이 아니라 청킹에서 생깁니다.")
    print("      자른 조각을 눈으로 읽어 보는 것이 가장 빠른 디버깅입니다.")


if __name__ == "__main__":
    main()
