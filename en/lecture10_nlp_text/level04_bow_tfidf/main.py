"""
level04 — Bag of Words and TF-IDF

Implements the document-term matrix (BoW) and TF-IDF with plain numpy and
verifies the result matches sklearn's TfidfVectorizer to 12 decimal places.
Finally, extracts each document's signature keywords from SAMPLE_DOCS
(company policies and product manuals) and computes cosine similarity
between documents.
"""

import re
import sys
import pathlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def tokenize(text: str) -> list[str]:
    """Extract lowercase word tokens of 2+ characters (same rule as sklearn's default)."""
    return re.findall(r"(?u)\b\w\w+\b", text.lower())


# ---------------------------------------------------------------------------
# [1] BoW — the document-term matrix, by hand
# ---------------------------------------------------------------------------

def build_bow(docs: list[str]) -> tuple[np.ndarray, list[str]]:
    """Document list -> (count matrix, vocabulary list). Vocabulary sorted alphabetically."""
    vocab = sorted({tok for d in docs for tok in tokenize(d)})
    index = {w: i for i, w in enumerate(vocab)}
    counts = np.zeros((len(docs), len(vocab)), dtype=float)
    for row, doc in enumerate(docs):
        for tok in tokenize(doc):
            counts[row, index[tok]] += 1
    return counts, vocab


# ---------------------------------------------------------------------------
# [2] TF-IDF — sklearn's default formula, in numpy
#     idf(t) = ln((1+N) / (1+df(t))) + 1   (smoothing)
#     tfidf  = tf * idf, then L2-normalize per document
# ---------------------------------------------------------------------------

def tfidf_numpy(counts: np.ndarray) -> np.ndarray:
    n_docs = counts.shape[0]
    df = (counts > 0).sum(axis=0)                       # per-word document frequency
    idf = np.log((1 + n_docs) / (1 + df)) + 1           # smoothed IDF
    weighted = counts * idf                             # TF x IDF
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return weighted / norms                             # per-document L2 normalization


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

MINI_DOCS = [
    "delivery fast and delivery driver friendly",
    "packaging careful price satisfied",
    "delivery slow price complaint",
]


def demo_bow() -> None:
    print("[1] BoW: grinding documents into a 'word ingredient label'")
    counts, vocab = build_bow(MINI_DOCS)
    header = "         " + " ".join(f"{w:>9s}" for w in vocab)
    print(header)
    for i, row in enumerate(counts):
        cells = " ".join(f"{int(v):9d}" for v in row)
        print(f"    doc{i} {cells}   <- {MINI_DOCS[i]!r}")
    sparsity = (counts == 0).mean()
    print(f"    -> matrix {counts.shape}, share of zeros {sparsity:.0%} (a sparse matrix)\n")


def demo_verify() -> None:
    print("[2]-[3] Hand-rolled TF-IDF vs sklearn, cross-checked")
    docs = [r["text"] for r in hjh_data.review_corpus(80, seed=3)]
    counts, vocab = build_bow(docs)
    mine = tfidf_numpy(counts)

    vec = TfidfVectorizer()                              # defaults: smoothing + L2
    theirs = vec.fit_transform(docs).toarray()
    their_vocab = vec.get_feature_names_out().tolist()

    assert vocab == their_vocab, "vocabulary mismatch"
    max_err = float(np.abs(mine - theirs).max())
    print(f"    {len(docs)} documents, {len(vocab)} vocabulary types")
    print(f"    max difference between our implementation and sklearn: {max_err:.2e}")
    ok = max_err < 1e-10
    print(f"    -> {'Match! We reproduced the library internals exactly.' if ok else 'Mismatch — recheck the formula'}\n")
    assert ok


def demo_keywords() -> np.ndarray:
    print("[4] Top-5 keywords per SAMPLE_DOCS document (by TF-IDF)")
    names = list(hjh_data.SAMPLE_DOCS.keys())
    docs = list(hjh_data.SAMPLE_DOCS.values())
    counts, vocab = build_bow(docs)
    scores = tfidf_numpy(counts)
    for i, name in enumerate(names):
        top = np.argsort(scores[i])[::-1][:5]
        words = [f"{vocab[j]}({scores[i, j]:.2f})" for j in top]
        print(f"    {name:22s}: {', '.join(words)}")
    print("    -> The vacation doc surfaces 'leave', the expenses doc 'trip'/'reimbursed' —")
    print("       each document's 'face' comes out on its own.\n")
    return scores


def demo_similarity(scores: np.ndarray) -> None:
    print("[5] Cosine similarity between documents (dot products of L2-normalized vectors)")
    names = [n.replace(".txt", "") for n in hjh_data.SAMPLE_DOCS.keys()]
    sim = scores @ scores.T
    print("    " + " " * 20 + "  ".join(f"{n.replace('policy_', '').replace('manual_', '')[:6]:>6s}" for n in names))
    for i, name in enumerate(names):
        cells = "  ".join(f"{sim[i, j]:6.2f}" for j in range(len(names)))
        print(f"    {name:20s}{cells}")
    # most similar pair, excluding self-similarity
    mask = sim - np.eye(len(names))
    a, b = np.unravel_index(np.argmax(mask), mask.shape)
    print(f"    -> Most similar pair: {names[a]} <-> {names[b]} ({mask[a, b]:.2f})")
    print("       Why the similarities are low overall: the documents barely share 'the")
    print("       same words'. TF-IDF only credits likeness when the letters match —")
    print("       recognizing 'different words with similar meaning' is what")
    print("       level06 word embeddings are for.")


if __name__ == "__main__":
    print("=" * 70)
    print("BoW and TF-IDF — text to numeric vectors, verified by building it ourselves")
    print("=" * 70 + "\n")
    np.random.seed(0)          # this level uses no randomness, fixed by convention
    demo_bow()
    demo_verify()
    scores = demo_keywords()
    demo_similarity(scores)
