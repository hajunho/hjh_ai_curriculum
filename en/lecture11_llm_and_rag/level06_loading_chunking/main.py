"""
Document loading and chunking: in what units should documents be cut for search?
1) Cut the internal documents with 3 strategies (fixed length / fixed+overlap /
   sentence-based)
2) Embed the pieces to build a mini search engine
3) Compare hit rates on the same 5 questions: 'did we retrieve the piece
   containing the answer?'
The chunk-size and overlap trade-offs become concrete numbers.
"""

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, split_sentences

# Evaluation set: (question, phrase the correct chunk must contain)
EVAL_SET = [
    ("How many days in advance must annual leave be requested?", "3 business days"),
    ("What is the daily allowance for a domestic business trip?", "KRW 30,000"),
    ("How many times a week is remote work allowed?", "twice a week"),
    ("How long is the product warranty?", "2 years"),
    ("How much clearance from the wall is needed during installation?", "10 centimeters"),
]


# ---------------------------------------------------------------------------
# The 3 chunking strategies
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, size: int = 125, overlap: int = 0) -> list[str]:
    """Strategy A/B: cut by character count (overlapping the previous piece by `overlap`)."""
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def chunk_by_sentence(text: str, max_chars: int = 225) -> list[str]:
    """Strategy C: group sentences without crossing max_chars, respecting boundaries."""
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
# Evaluation: mini search engine performance per chunking strategy
# ---------------------------------------------------------------------------

def evaluate(chunks: list[tuple[str, str]], top_k: int = 2, verbose: bool = False):
    """Embed the pieces into a search engine and measure hit rate on the eval set.
    Hit = one of the top_k pieces contains the answer phrase."""
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
                print(f"      [miss] {question}")
                print(f"             top piece: \"{texts[order[0]][:44]}...\"")
            else:
                found_text = texts[order[hit_rank - 1]]
                print(f"      [hit/rank {hit_rank}] {question}")
                print(f"             piece: \"{found_text[:44]}...\"")
    return hits, len(EVAL_SET)


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 06 | Loading and chunking — how you cut decides search quality")
    print("=" * 62)

    # [1] Document loading
    docs = hjh_data.SAMPLE_DOCS
    total_chars = sum(len(d) for d in docs.values())
    print(f"\n[1] Documents loaded: {len(docs)}, {total_chars:,} characters total")
    print("    Why cut at all? (1) the LLM context window is finite (2) whole-document")
    print("    embeddings blur topics together and dull the search (3) evidence must be")
    print("    cited piece by piece to be precise")

    # [2] Cutting with the three strategies
    strategies = {
        "A. fixed 125 chars (no overlap)": lambda t: chunk_fixed(t, size=125, overlap=0),
        "B. fixed 125 + overlap 40": lambda t: chunk_fixed(t, size=125, overlap=40),
        "C. sentence-based (max 225)": lambda t: chunk_by_sentence(t, max_chars=225),
    }
    all_chunks: dict[str, list[tuple[str, str]]] = {}
    print("\n[2] Chunking summary")
    for label, fn in strategies.items():
        chunks = []
        for name, doc in docs.items():
            chunks += [(name, c) for c in fn(doc)]
        all_chunks[label] = chunks
        avg = np.mean([len(c) for _, c in chunks])
        print(f"    {label:<32} {len(chunks):>2} pieces, avg {avg:.0f} chars")

    # [3] Strategy A's problem: sentences snap mid-thought
    print("\n[3] Sample pieces from strategy A (sentences cut mid-way, meaning damaged)")
    for _, c in all_chunks["A. fixed 125 chars (no overlap)"][:3]:
        print(f"    | {c}")
    print("    -> Words and numbers get severed at the boundary, splitting facts in two.")

    # [4] Hit-rate comparison per strategy
    print("\n[4] Search hit rate (5 questions; hit = answer phrase in the top 2 pieces)")
    results = {}
    for label, chunks in all_chunks.items():
        hits, total = evaluate(chunks)
        results[label] = hits
        bar = "#" * hits + "." * (total - hits)
        print(f"    {label:<32} {hits}/{total}  [{bar}]")

    # [5] Details for the best strategy
    best_label = max(results, key=results.get)
    print(f"\n[5] Details for the best strategy '{best_label}'")
    evaluate(all_chunks[best_label], verbose=True)

    print("\n[6] The trade-offs, recapped")
    print("    Pieces too small: sentences severed, answers scattered across pieces")
    print("    Pieces too large: topics mix, search blurs, token costs climb")
    print("    Overlap         : patches information cut at boundaries (more storage)")
    print("    Practical start : respect sentence/paragraph boundaries + 200-500 tokens")
    print("                      + 10-20% overlap")

    print("\nSummary: half of all RAG quality problems come from chunking, not the model.")
    print("         Reading the cut pieces with your own eyes is the fastest debugging.")


if __name__ == "__main__":
    main()
