"""
The complete RAG (retrieval-augmented generation) pipeline: retrieve ->
augment -> generate.
The internal policies/manuals are indexed ahead of time; when a question
arrives we
1) find the relevant pieces by semantic search (Retrieval)
2) attach the found pieces to the prompt (Augmentation)
3) generate an answer that cites its evidence (Generation)
With insufficient evidence, it does not make things up — it says
'I don't know'.
"""

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, MockLLM, split_sentences


class RAGChatbot:
    """A mini internal chatbot assembled from the retrieve-augment-generate stages."""

    def __init__(self, docs: dict[str, str], min_score: float = 0.4, top_k: int = 3):
        self.min_score = min_score      # below this similarity we answer 'I don't know'
        self.top_k = top_k
        self.llm = MockLLM()
        # Indexing: sentence-based chunking -> embedding (assembling levels 06-07)
        self.chunks = []                # (source, sentence)
        for name, doc in docs.items():
            self.chunks += [(name, s) for s in split_sentences(doc)]
        texts = [c for _, c in self.chunks]
        self.embedder = MockEmbedding(dim=512).fit(texts)
        self.vectors = self.embedder.embed_batch(texts)

    # -- stage 1: retrieval ---------------------------------------------------
    def retrieve(self, question: str) -> list[tuple[float, str, str]]:
        sims = self.vectors @ self.embedder.embed(question)
        order = np.argsort(-sims)[: self.top_k]
        return [(float(sims[i]), self.chunks[i][0], self.chunks[i][1]) for i in order]

    # -- stage 2: augmentation (assemble the prompt for a real LLM) ------------
    @staticmethod
    def build_prompt(question: str, evidence: list[tuple[float, str, str]]) -> str:
        ctx = "\n".join(f"- ({src}) {text}" for _, src, text in evidence)
        return (
            "You are the internal policy-guide chatbot.\n"
            "Answer ONLY from the 'evidence documents' below, and cite your sources.\n"
            "If something is not in the evidence, answer 'not confirmed in the documents'.\n"
            f"[Evidence documents]\n{ctx}\n"
            f"[Question] {question}"
        )

    # -- stage 3: generation ----------------------------------------------------
    def answer(self, question: str, show_prompt: bool = False) -> str:
        evidence = self.retrieve(question)
        best = evidence[0][0] if evidence else 0.0
        print(f"    retrieval: best similarity {best:.3f} "
              f"({'sufficient evidence' if best >= self.min_score else 'insufficient evidence'})")
        for sim, src, text in evidence:
            tag = "kept" if sim >= self.min_score else "dropped"
            print(f"      [{tag:>7}] {sim:.3f} {text[:30]}... ({src})")

        if best < self.min_score:
            # If the open book has no answer, a blank sheet beats an invention
            # (hallucination prevention)
            return ("I'm sorry — I could not find evidence in the internal documents, "
                    "so I cannot answer. Please contact the responsible department.")

        kept = [(src, text) for sim, src, text in evidence if sim >= self.min_score]
        if show_prompt:
            print("    augmentation: the prompt that would go to a real LLM ↓")
            for line in self.build_prompt(question, evidence).splitlines():
                print(f"      | {line}")
        return self.llm.answer_with_context(question, kept)


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 08 | RAG chatbot — the full retrieve-augment-generate assembly")
    print("=" * 62)

    print("\n[1] Indexing: 5 internal policies/manuals embedded sentence by sentence")
    bot = RAGChatbot(hjh_data.SAMPLE_DOCS)
    print(f"    {len(bot.chunks)} pieces, vectors {bot.vectors.shape}, "
          f"'I don't know' similarity threshold {bot.min_score}")

    questions = [
        ("How many days in advance must annual leave be requested?", True),  # 1st also shows the prompt
        ("What is the daily allowance for overseas business trips?", False),
        ("How many times a week is remote work allowed?", False),
        ("How are repairs handled after the warranty expires?", False),
        ("What is next year's minimum wage?", False),        # out-of-corpus -> don't know
    ]

    for i, (q, show) in enumerate(questions, 2):
        print(f"\n[{i}] Question: {q}")
        answer = bot.answer(q, show_prompt=show)
        print("    Answer:")
        for line in answer.splitlines():
            print(f"      {line}")

    print("\n[7] The RAG pipeline, recapped")
    print("    question -> (1) retrieve: top-k pieces from the vector DB")
    print("             -> (2) augment: insert the pieces as 'evidence documents'")
    print("             -> (3) generate: instruct 'answer only from evidence' + cite sources")
    print("    Insufficient evidence: don't invent — say 'I don't know' + point to the")
    print("    responsible team (rule #1 of hallucination prevention)")

    # ------------------------------------------------------------------
    # [Reference] With a real API, only stage 3 (generation) changes:
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-opus-5", max_tokens=1024,
    #     messages=[{"role": "user",
    #                "content": bot.build_prompt(question, evidence)}],
    # )
    # answer = response.content[0].text
    # The retrieval and augmentation you built today are reused as-is.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
