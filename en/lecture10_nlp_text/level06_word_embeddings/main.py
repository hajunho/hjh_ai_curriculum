"""
level06 — Word Embeddings

Builds a co-occurrence matrix from tiny_corpus (a mini English corpus),
then learns word embeddings 'by hand' via PPMI transformation + SVD
compression. Demonstrates similar-word search and saves a 2-D embedding
map as a PNG. Uses only numpy — no word2vec-style libraries.
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
JOSA = ["'s"]      # in the Korean edition this strips particles; for English only 's


def set_korean_font() -> None:
    """The Korean edition needed a CJK font for the plot labels.
    English renders fine with matplotlib defaults, so we only fix the minus sign."""
    plt.rcParams["axes.unicode_minus"] = False


def tokenize(corpus: str) -> list[str]:
    """Remove periods + peel a possessive 's off word endings (crude split)."""
    tokens = []
    for word in corpus.replace(".", " ").split():
        for josa in JOSA:
            if word.endswith(josa) and len(word) - len(josa) >= 2:
                word = word[: -len(josa)]
                break
        tokens.append(word)
    return tokens


def build_cooccurrence(tokens: list[str], vocab: list[str], window: int = 2):
    """[2] The 'friendship ledger': counts how often words appear within the window."""
    index = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                C[index[center], index[tokens[j]]] += 1
    return C


def ppmi(C: np.ndarray) -> np.ndarray:
    """[3] Positive pointwise mutual information — keep only 'more-than-chance friendship'."""
    total = C.sum()
    row = C.sum(axis=1, keepdims=True)
    col = C.sum(axis=0, keepdims=True)
    expected = row @ col / total                 # co-occurrences expected by pure chance
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(C * total / (row * col))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0)


def nearest(word: str, emb: np.ndarray, vocab: list[str], k: int = 4):
    """[5] k nearest neighbors by cosine similarity."""
    index = {w: i for i, w in enumerate(vocab)}
    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = normed @ normed[index[word]]
    order = np.argsort(sims)[::-1]
    return [(vocab[i], float(sims[i])) for i in order if vocab[i] != word][:k]


if __name__ == "__main__":
    print("=" * 70)
    print("Word embeddings — a word map from co-occurrence + PPMI + SVD")
    print("=" * 70 + "\n")
    np.random.seed(0)
    set_korean_font()

    # [1] Corpus prep ------------------------------------------------------
    corpus = hjh_data.tiny_corpus()
    tokens = tokenize(corpus)
    freq = Counter(tokens)
    vocab = sorted(freq)
    print(f"[1] Corpus: {len(corpus):,} chars -> {len(tokens):,} tokens, {len(vocab)} vocabulary types")
    print(f"    top frequencies: {freq.most_common(5)}\n")

    # [2] Co-occurrence matrix ---------------------------------------------
    C = build_cooccurrence(tokens, vocab, window=2)
    print(f"[2] Co-occurrence matrix {C.shape} (window size 2)")
    idx = {w: i for i, w in enumerate(vocab)}
    student_row = C[idx["student"]]
    top = np.argsort(student_row)[::-1][:5]
    print("    'student' most often sits next to:",
          ", ".join(f"{vocab[i]}({int(student_row[i])}x)" for i in top))
    print("    -> The sentence template is 'time subject verb object', so the window")
    print("       around a subject noun fills with articles and verbs.")
    print("       This 'seating profile' becomes the word's fingerprint.\n")

    # [3] PPMI -------------------------------------------------------------
    P = ppmi(C)
    common_word = freq.most_common(1)[0][0]
    print(f"[3] PPMI transform: discounting accidental seatmates (common words)")
    print(f"    e.g. 'student'-'{common_word}' co-occurs {int(C[idx['student'], idx[common_word]])}x, yet "
          f"PPMI {P[idx['student'], idx[common_word]]:.2f}")
    print(f"    -> High counts with a word that 'sits next to everyone' carry no information.\n")

    # [4] SVD compression ----------------------------------------------------
    U, S, Vt = np.linalg.svd(P)
    dim = 8
    emb = U[:, :dim] * S[:dim]                   # keep only the top dim axes
    explained = S[:dim].sum() / S.sum()
    print(f"[4] SVD compression: {P.shape[1]} dims -> {dim}-dim embeddings")
    print(f"    The top {dim} singular values hold {explained:.0%} of the total information.\n")

    # [5] Similar-word search ------------------------------------------------
    print("[5] Similar-word search (cosine similarity)")
    for query in ["student", "report", "made", "Yesterday"]:
        pairs = ", ".join(f"{w}({s:.2f})" for w, s in nearest(query, emb, vocab))
        print(f"    neighbors of {query:9s}: {pairs}")
    print("    -> Subjects cluster with subjects, objects with objects, verbs with verbs.")
    print("       'Similar meaning (role) = nearby coordinates'!\n")

    # [6] Embedding map PNG --------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    coords = U[:, :2] * S[:2]                    # 2-D coordinates for the plot
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
    print(f"[6] Embedding map saved: {path}")
    print("    Check the picture: do nouns, verbs, and time expressions claim their own regions?")
    print("\nBottom line: 'a word's meaning is decided by the company it keeps' — "
          "implemented with statistics + linear algebra.")
