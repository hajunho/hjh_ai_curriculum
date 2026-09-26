"""
Implement the 3-piece retrieval-quality toolkit and compare before/after.
1) BM25: the standard formula of classic keyword search, from scratch
2) Hybrid: score fusion of BM25 (exact words) + embeddings (meaning)
3) MMR: fixing the redundancy problem (stacks of near-identical pieces)
   with diversity
RAG answer quality is ultimately decided by 'what retrieval brought back'.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, split_sentences, tokenize

# For reproducing redundancy: an internal notice nearly identical to the policy
# has been indexed alongside it
EXTRA_DOCS = {
    "notice_annual_leave.txt": (
        "Annual leave of 15 days is granted after one full year of service. "
        "Please request annual leave through the approval system at least "
        "3 business days ahead. "
        "See the intranet notice for how to request annual leave."
    ),
}


# Stopwords: in English, function words like 'the'/'for' would otherwise
# rack up spurious keyword matches everywhere. (Search engines do the same.)
STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by",
        "for", "from", "with", "as", "is", "are", "was", "be", "am", "do",
        "does", "did", "can", "could", "may", "must", "will", "would", "i",
        "we", "you", "it", "my", "our", "your", "this", "that", "what",
        "which", "how", "many", "much", "when", "where", "who", "why",
        "me", "get", "about", "tell"}


class BM25:
    """A BM25 keyword search engine (from scratch).
    The idea: a query word scores higher when (1) it appears often in that
    document (TF, with saturation) (2) it is rare across the corpus (IDF)
    (3) the document is short."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [[w for w in tokenize(d) if w not in STOP] for d in docs]
        self.doc_len = np.array([len(t) for t in self.doc_tokens])
        self.avg_len = float(self.doc_len.mean())
        self.n_docs = len(docs)
        self.df: dict[str, int] = {}                 # word -> number of docs containing it
        for tokens in self.doc_tokens:
            for w in set(tokens):
                self.df[w] = self.df.get(w, 0) + 1

    def _idf(self, word: str) -> float:
        n = self.df.get(word, 0)
        return math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs)
        for w in (t for t in tokenize(query) if t not in STOP):
            idf = self._idf(w)
            for i, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(w)
                if tf == 0:
                    continue
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out


def minmax(x: np.ndarray) -> np.ndarray:
    """Put scores on one scale: normalize to 0-1 (required before fusion)."""
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def mmr_select(qv: np.ndarray, vectors: np.ndarray, base_scores: np.ndarray,
               k: int = 4, lam: float = 0.6) -> list[int]:
    """MMR (Maximal Marginal Relevance):
    pick one at a time, weighing relevance (lam) against 'not overlapping
    what is already picked' (1-lam)."""
    selected: list[int] = []
    candidates = list(np.argsort(-base_scores)[: k * 3])   # top candidates only
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
        print(f"      #{rank} {s:5.3f} {text[:32]}... ({src})")


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 09 | Retrieval quality — BM25, hybrid, MMR")
    print("=" * 62)

    # [1] Corpus: internal documents + an overlapping notice (real-world redundancy)
    corpus = []
    for name, doc in {**hjh_data.SAMPLE_DOCS, **EXTRA_DOCS}.items():
        corpus += [(name, s) for s in split_sentences(doc)]
    texts = [t for _, t in corpus]
    print(f"\n[1] Corpus: {len(hjh_data.SAMPLE_DOCS) + 1} documents -> {len(corpus)} pieces")
    print("    (an 'internal notice' overlapping the leave policy is mixed in —")
    print("     the realistic situation)")

    # [2] Prepare the two search engines
    bm25 = BM25(texts)
    emb = MockEmbedding(dim=512).fit(texts)
    vectors = emb.embed_batch(texts)

    # [3] BM25 at work: strong on noun-pile search-box queries
    q1 = "unused annual leave allowance payout"
    print(f"\n[2] BM25 at work: \"{q1}\" (the noun pile you type into a search box)")
    b1 = bm25.scores(q1)
    show("BM25 top-2", [(i, float(b1[i])) for i in np.argsort(-b1)[:2]], corpus)
    print("    -> When the query words appear verbatim, BM25 is precise and fast.")

    # [4] The two engines' different failure modes
    q2 = "What are the rules for working remotely?"   # doc says 'remote work' -> word-form mismatch
    q3 = "What daily amount of money do I get on a work trip?"  # paraphrase drags semantics to a neighbor
    print(f"\n[3] Failure-mode comparison")
    for q, note in ((q2, "the document says 'remote work' — 'working remotely' never matches a keyword"),
                    (q3, "the paraphrase drifts semantics to a neighboring sentence — the exact word 'daily' saves BM25")):
        b = bm25.scores(q)
        s = vectors @ emb.embed(q)
        print(f"  Question: \"{q}\"  ({note})")
        show("(a) BM25 keyword", [(i, float(b[i])) for i in np.argsort(-b)[:1]], corpus)
        show("(b) embedding semantic", [(i, float(s[i])) for i in np.argsort(-s)[:1]], corpus)

    # [5] Hybrid: normalize both to 0-1, then weighted sum
    alpha = 0.5
    answers = {q1: "paid out as an allowance", q2: "Remote work is allowed", q3: "KRW 30,000"}
    print(f"\n[4] Hybrid = {alpha} x BM25 + {1 - alpha} x semantic (each min-max normalized)")
    print(f"    {'question':<48}{'BM25':>6}{'sem':>6}{'hybrid':>8}   (O = correct at #1)")
    for q, span in answers.items():
        b, s = bm25.scores(q), vectors @ emb.embed(q)
        hybrid = alpha * minmax(b) + (1 - alpha) * minmax(s)
        marks = ["O" if span in corpus[int(np.argmax(x))][1] else "X"
                 for x in (b, s, hybrid)]
        print(f"    {q:<48}{marks[0]:>4}{marks[1]:>6}{marks[2]:>8}")
    print("    -> Each engine loses one question — a different one. Fused search")
    print("       puts the right piece at #1 for all three.")

    # [6] MMR: before/after reducing redundancy
    q3 = "Tell me about the annual leave policy"
    print(f"\n[5] MMR before/after: \"{q3}\" (top-4)")
    s3 = vectors @ emb.embed(q3)
    plain = np.argsort(-s3)[:4]
    show("(a) plain similarity top-4 (redundant)", [(i, float(s3[i])) for i in plain], corpus)
    picked = mmr_select(emb.embed(q3), vectors, s3, k=4, lam=0.6)
    show("(b) MMR top-4 (diversity secured)", [(i, float(s3[i])) for i in picked], corpus)
    print("    -> In (a), two slots go to essentially the same sentence (the request")
    print("       deadline, from the notice AND the policy), while (b) drops the")
    print("       duplicate and secures new information (the unused-leave payout).")
    print("       With the same context budget (top-k), (b) makes richer answers.")

    # [7] Introducing reranking
    print("\n[6] One step further: re-ranking")
    print("    First-pass retrieval (fast hybrid) narrows to 20-50 candidates, then a")
    print("    more precise model (a cross-encoder, or the LLM itself) re-scores each")
    print("    question-piece pair to pick the final top-k — the two-stage structure")
    print("    that is standard in production.")
    print("    Same principle as hiring: resume screening (fast) -> interviews (precise).")

    print("\nSummary: keywords and meaning are complements, not rivals (hybrid),")
    print("         and fill your top-k with pieces that are 'relevant AND mutually")
    print("         different' (MMR).")


if __name__ == "__main__":
    main()
