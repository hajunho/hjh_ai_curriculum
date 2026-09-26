"""
The principle of a vector database: build 'a library you search by meaning'
from scratch in NumPy.
- MiniVectorStore: add / search / save / load + metadata filters
- Index the internal document pieces, search semantically, then save and
  reload and confirm identical results
- Time exhaustive search as the data grows, and introduce approximate
  nearest-neighbor (ANN) search
This is the skeleton of what production tools like Chroma/FAISS do inside.
"""

import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, split_sentences

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


class MiniVectorStore:
    """A teaching-size mini vector store.
    It carries the 4 core functions of production vector DBs
    (Chroma, FAISS, pgvector...):
    add (index) / search (similarity + filter) / save (persist) / load (restore)"""

    def __init__(self, embedder: MockEmbedding):
        self.embedder = embedder
        self.vectors = np.zeros((0, embedder.dim))
        self.texts: list[str] = []
        self.metas: list[dict] = []

    def __len__(self) -> int:
        return len(self.texts)

    def add(self, texts: list[str], metas: list[dict]) -> None:
        """Embed and store document pieces (this is indexing)."""
        new_vecs = self.embedder.embed_batch(texts)
        self.vectors = np.vstack([self.vectors, new_vecs])
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query: str, k: int = 3, where: dict | None = None):
        """Embed the question and return the top k by cosine similarity.
        A metadata filter can be applied, e.g. where={"category": "policy"}."""
        qv = self.embedder.embed(query)
        sims = self.vectors @ qv                      # normalized vectors: dot=cosine
        if where:
            mask = np.array([all(m.get(key) == val for key, val in where.items())
                             for m in self.metas])
            sims = np.where(mask, sims, -np.inf)      # exclude out-of-filter candidates
        order = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.texts[i], self.metas[i])
                for i in order if np.isfinite(sims[i])]

    def save(self, path: str) -> None:
        """Persist to disk: vectors as .npz, source text and metadata as .json."""
        np.savez_compressed(path + ".npz", vectors=self.vectors)
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metas": self.metas},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, embedder: MockEmbedding) -> "MiniVectorStore":
        store = cls(embedder)
        store.vectors = np.load(path + ".npz")["vectors"]
        with open(path + ".json", encoding="utf-8") as f:
            payload = json.load(f)
        store.texts = payload["texts"]
        store.metas = payload["metas"]
        return store


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 07 | Vector DB — building the search-by-meaning library yourself")
    print("=" * 62)

    # [1] Prepare document pieces (level06's sentence-based chunking)
    chunks, metas = [], []
    for name, doc in hjh_data.SAMPLE_DOCS.items():
        category = "policy" if "policy" in name else "manual"
        for i, sent in enumerate(split_sentences(doc)):
            chunks.append(sent)
            metas.append({"source": name, "category": category, "chunk_id": i})
    print(f"\n[1] Pieces to index: {len(chunks)} (metadata: source/category/id)")

    # [2] Building the store (add = indexing)
    embedder = MockEmbedding(dim=512).fit(chunks)
    store = MiniVectorStore(embedder)
    store.add(chunks, metas)
    print(f"\n[2] MiniVectorStore built: {len(store)} pieces, "
          f"vector matrix {store.vectors.shape}")

    # [3] Semantic search
    query = "What do I need in order to get a repair?"
    print(f"\n[3] Search: \"{query}\"")
    for sim, text, meta in store.search(query, k=3):
        print(f"    {sim:.3f} {text[:36]}... ({meta['source']})")
    print("    -> The word 'repair' pinpoints the warranty clause in the product manual.")

    # [4] Metadata filter: same question, but only within 'policy' documents
    print("\n[4] Filtered search: same question, where={'category': 'policy'}")
    for sim, text, meta in store.search(query, k=2, where={"category": "policy"}):
        print(f"    {sim:.3f} {text[:36]}... ({meta['source']})")
    print("    -> Manuals are excluded; only policy documents remain candidates.")
    print("       Narrowing search by department/date/document type via metadata")
    print("       filters is a vector DB fundamental.")

    # [5] Persistence: save -> pretend it's a new process -> restore
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "mini_store")
    store.save(path)
    restored = MiniVectorStore.load(path, embedder)
    q = "How many days in advance do I request annual leave?"
    same = store.search(q, k=1)[0][1] == restored.search(q, k=1)[0][1]
    print(f"\n[5] save/load check: saved {path}.npz/.json, then restored")
    print(f"    Restored store returns identical search results: {same}")
    print("    -> Embedding is expensive; 'index once, reuse forever' is the point.")

    # [6] What about scale? Time exhaustive comparison + the idea of ANN
    print("\n[6] Scale experiment: timing exhaustive search over growing fake vectors")
    rng = np.random.default_rng(0)
    qv = store.vectors[0]
    for n in (1_000, 20_000, 200_000):
        big = rng.standard_normal((n, embedder.dim))
        big /= np.linalg.norm(big, axis=1, keepdims=True)
        t0 = time.perf_counter()
        np.argsort(-(big @ qv))[:3]
        ms = (time.perf_counter() - t0) * 1000
        print(f"    exhaustive comparison over {n:>7,} vectors: {ms:6.1f} ms")
    print("    -> Up to a few hundred thousand vectors, exhaustive NumPy search is")
    print("       plenty fast. At tens of millions, you bring in approximate")
    print("       nearest-neighbor search (ANN: divide the library into zones and")
    print("       inspect only the promising ones) via dedicated vector DBs like")
    print("       Chroma/FAISS/pgvector. Their usage looks almost like today's API:")
    print("       collection.add(...) / collection.query(...)  (in Chroma's case)")

    print("\nSummary: vector DB = embedding matrix + text/metadata + (at scale) an")
    print("         approximate index. Today's add/search/save/load is the shared")
    print("         skeleton of every vector DB.")


if __name__ == "__main__":
    main()
