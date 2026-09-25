"""
RAG(검색 증강 생성) 파이프라인 완결판: 검색 -> 증강 -> 생성.
사내 규정/매뉴얼을 색인해 두고, 질문이 오면
1) 관련 조각을 의미 검색으로 찾고 (Retrieval)
2) 찾은 조각을 프롬프트에 붙여 (Augmentation)
3) 근거를 인용한 답을 생성합니다 (Generation)
근거가 부족하면 지어내지 않고 '모른다'고 답합니다.
"""

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, MockLLM, split_sentences


class RAGChatbot:
    """검색-증강-생성 3단계를 조립한 미니 사내 챗봇."""

    def __init__(self, docs: dict[str, str], min_score: float = 0.15, top_k: int = 3):
        self.min_score = min_score      # 이 유사도 미만이면 '모른다'로 응답
        self.top_k = top_k
        self.llm = MockLLM()
        # 색인: 문장 단위 청킹 -> 임베딩 (level06~07 의 조립)
        self.chunks = []                # (출처, 문장)
        for name, doc in docs.items():
            self.chunks += [(name, s) for s in split_sentences(doc)]
        texts = [c for _, c in self.chunks]
        self.embedder = MockEmbedding(dim=512).fit(texts)
        self.vectors = self.embedder.embed_batch(texts)

    # -- 1단계: 검색 ---------------------------------------------------------
    def retrieve(self, question: str) -> list[tuple[float, str, str]]:
        sims = self.vectors @ self.embedder.embed(question)
        order = np.argsort(-sims)[: self.top_k]
        return [(float(sims[i]), self.chunks[i][0], self.chunks[i][1]) for i in order]

    # -- 2단계: 증강 (실제 LLM 에 보낼 프롬프트 조립) -------------------------
    @staticmethod
    def build_prompt(question: str, evidence: list[tuple[float, str, str]]) -> str:
        ctx = "\n".join(f"- ({src}) {text}" for _, src, text in evidence)
        return (
            "당신은 사내 규정 안내 챗봇입니다.\n"
            "아래 '근거 문서'에 있는 내용만으로 답하고, 출처를 표기하세요.\n"
            "근거에 없는 내용은 '문서에서 확인되지 않습니다'라고 답하세요.\n"
            f"[근거 문서]\n{ctx}\n"
            f"[질문] {question}"
        )

    # -- 3단계: 생성 ----------------------------------------------------------
    def answer(self, question: str, show_prompt: bool = False) -> str:
        evidence = self.retrieve(question)
        best = evidence[0][0] if evidence else 0.0
        print(f"    검색: 최고 유사도 {best:.3f} "
              f"({'근거 충분' if best >= self.min_score else '근거 부족'})")
        for sim, src, text in evidence:
            tag = "채택" if sim >= self.min_score else "탈락"
            print(f"      [{tag}] {sim:.3f} {text[:30]}... ({src})")

        if best < self.min_score:
            # 오픈북에 답이 없으면 백지 제출이 정답 (환각 방지)
            return "죄송합니다. 사내 문서에서 근거를 찾지 못해 답변드릴 수 없습니다. 담당 부서에 문의해 주세요."

        kept = [(src, text) for sim, src, text in evidence if sim >= self.min_score]
        if show_prompt:
            print("    증강: 실제 LLM 에 보낼 프롬프트 ↓")
            for line in self.build_prompt(question, evidence).splitlines():
                print(f"      | {line}")
        return self.llm.answer_with_context(question, kept)


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 08 | RAG 챗봇 — 검색·증강·생성의 완결 조립")
    print("=" * 62)

    print("\n[1] 색인: 사내 규정·매뉴얼 5건을 문장 단위로 임베딩")
    bot = RAGChatbot(hjh_data.SAMPLE_DOCS)
    print(f"    조각 {len(bot.chunks)}개, 벡터 {bot.vectors.shape}, "
          f"모른다 기준 유사도 {bot.min_score}")

    questions = [
        ("연차는 며칠 전에 신청해야 하나요?", True),    # 첫 질문은 프롬프트도 공개
        ("해외 출장 일비는 얼마인가요?", False),
        ("재택근무는 주 몇 회까지 가능한가요?", False),
        ("보증 기간이 지나면 수리는 어떻게 되나요?", False),
        ("내년도 최저임금은 얼마인가요?", False),        # 문서 밖 질문 -> 모른다
    ]

    for i, (q, show) in enumerate(questions, 2):
        print(f"\n[{i}] 질문: {q}")
        answer = bot.answer(q, show_prompt=show)
        print("    답변:")
        for line in answer.splitlines():
            print(f"      {line}")

    print("\n[7] RAG 파이프라인 정리")
    print("    질문 -> (1)검색: 벡터 DB 에서 관련 조각 top-k")
    print("         -> (2)증강: 조각을 '근거 문서'로 프롬프트에 삽입")
    print("         -> (3)생성: 근거 안에서만 답하게 지시 + 출처 인용")
    print("    근거 부족 시: 지어내지 않고 '모른다' + 담당자 안내 (환각 방지 1원칙)")

    # ------------------------------------------------------------------
    # [참고] 실제 API 라면 3단계 생성만 이렇게 바뀝니다:
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-opus-5", max_tokens=1024,
    #     messages=[{"role": "user",
    #                "content": bot.build_prompt(question, evidence)}],
    # )
    # answer = response.content[0].text
    # 검색·증강(오늘 만든 부분)은 그대로 재사용합니다.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
