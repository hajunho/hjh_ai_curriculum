"""
Embeddings: turning sentences into numeric coordinates (vectors) so we can
search by 'meaning'.
1) Vectorize the sentences of the internal documents with MockEmbedding
2) Compare cosine similarity between same-topic / different-topic sentences
3) Compare semantic search vs word-match keyword search for the same question
4) Draw the embedding space as a 2-D map and save it to outputs/
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

# Function words to ignore in keyword search — otherwise "the"/"for" would
# "match" nearly every sentence and drown the ranking in noise.
STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by",
        "for", "from", "with", "as", "is", "are", "was", "be", "am", "do",
        "does", "did", "can", "could", "may", "must", "will", "would", "i",
        "we", "you", "it", "my", "our", "your", "this", "that", "what",
        "which", "how", "many", "much", "when", "where", "who", "why"}


def keyword_search(query: str, corpus: list[tuple[str, str]], k: int = 3):
    """Classic keyword search: score = how many query 'words' appear verbatim."""
    q_tokens = [w for w in tokenize(query) if len(w) >= 2 and w not in STOP]
    results = []
    for source, sent in corpus:
        s_tokens = set(tokenize(sent))
        hits = [w for w in q_tokens if w in s_tokens]   # exact matches only
        results.append((len(hits), hits, source, sent))
    results.sort(key=lambda x: -x[0])
    return results[:k]


def semantic_search(query: str, emb: MockEmbedding, vectors: np.ndarray,
                    corpus: list[tuple[str, str]], k: int = 3):
    """Semantic search: rank all sentence vectors by cosine similarity to the query vector."""
    qv = emb.embed(query)
    sims = vectors @ qv                       # normalized vectors: dot = cosine
    order = np.argsort(-sims)[:k]
    return [(float(sims[i]), corpus[i][0], corpus[i][1]) for i in order]


def main() -> None:
    np.random.seed(42)                        # reproducibility

    print("=" * 62)
    print("Level 05 | Embeddings and semantic search")
    print("=" * 62)

    # [1] Prepare the internal documents, sentence by sentence
    corpus = []                               # (source filename, sentence)
    for name, doc in hjh_data.SAMPLE_DOCS.items():
        corpus += [(name, s) for s in split_sentences(doc)]
    print(f"\n[1] {len(hjh_data.SAMPLE_DOCS)} documents -> split into {len(corpus)} sentences")

    # [2] Embedding: sentence -> 512-dimensional coordinates
    emb = MockEmbedding(dim=512).fit([s for _, s in corpus])
    vectors = emb.embed_batch([s for _, s in corpus])
    sample_vec = vectors[0]
    nz = np.nonzero(sample_vec)[0][:5]
    print(f"\n[2] Embedding done: every sentence is a {vectors.shape[1]}-dim vector "
          f"(all normalized to length 1)")
    print(f"    e.g. \"{corpus[0][1][:30]}...\"")
    print("    -> " + ", ".join(f"cell {i}={sample_vec[i]:.3f}" for i in nz) +
          f", ... ({int((sample_vec != 0).sum())} nonzero cells / {vectors.shape[1]})")

    # [3] Close coordinates = similar meaning
    print("\n[3] Cosine similarity: the closer to 1, the more same-topic")
    pairs = [
        ("How many days of annual leave can I take?",
         "Employees are granted 15 days of annual leave upon completing one full year of service."),
        ("How many days of annual leave can I take?",
         "The product warranty lasts 2 years from the date of purchase."),
        ("I am wondering about the warranty period",
         "The product warranty lasts 2 years from the date of purchase."),
    ]
    for a, b in pairs:
        sim = cosine(emb.embed(a), emb.embed(b))
        print(f"    {sim:.3f}  \"{a}\"  vs  \"{b[:28]}...\"")

    # [4] Semantic search vs keyword search
    queries = [
        "What are the rules for working remotely?",   # says 'remotely', doc says 'remote work'
        "When do I need to hand in expense receipts?",
    ]
    print("\n[4] Same question, two kinds of search")
    for q in queries:
        print(f"\n  Question: \"{q}\"")
        print("  (a) keyword search (exact word match)")
        for score, hits, source, sent in keyword_search(q, corpus):
            mark = ", ".join(hits) if hits else "no matching words"
            print(f"      {score} match(es) [{mark}] {sent[:34]}... ({source})")
        print("  (b) semantic search (embedding cosine)")
        for sim, source, sent in semantic_search(q, emb, vectors, corpus):
            print(f"      {sim:.3f} {sent[:34]}... ({source})")
    print("\n    -> Change the wording just a little ('remotely' vs 'remote work') and")
    print("       keyword search goes blind, while the embedding still finds the")
    print("       remote-work policy through character n-grams and co-occurrence.")

    # [5] A 2-D map of the embedding space (dimensionality reduction via SVD)
    centered = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    # Short legend labels for the plot (policy_vacation.txt -> policy: vacation ...)
    label_en = {"policy_vacation.txt": "policy: vacation", "policy_expenses.txt": "policy: expense",
                "policy_remote_work.txt": "policy: remote", "manual_install.txt": "manual: install",
                "manual_warranty.txt": "manual: warranty"}
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
    print(f"\n[5] Embedding map saved -> {out_path}")
    print("    Check whether sentences from the same document (topic) cluster together.")

    print("\nSummary: embedding is the technique that builds a map where 'similar meaning")
    print("         = nearby coordinates'. Search by meaning even when the words differ —")
    print("         the first component of RAG.")

    # ------------------------------------------------------------------
    # [Reference] A real service uses a dedicated embedding-model API, e.g.:
    # response = embeddings_client.create(model="...", input=["sentence 1", "sentence 2"])
    # vec = response.data[0].embedding        # usually a 1024-3072-dim float vector
    # The usage (ranking by cosine similarity) is exactly the same as today's code.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
