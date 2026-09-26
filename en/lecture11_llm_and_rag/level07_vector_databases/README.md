# Lecture 11 · Level 07 — Vector Databases

> Build "the library you search by meaning" from scratch in NumPy — add/search/save/load, metadata filters, and the concept of approximate search (ANN).
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level05, level06 / **Estimated time** 50 min

## 1. Why Learn This — The Business View

In levels 05–06 we successfully embedded and searched document pieces. Real work adds three more problems. First, **embedding is expensive** — embedding 100,000 document pieces through an API costs time and money, so vectors must be stored once and reused. Second, **conditional search is required** — without filters like "only in HR policies" or "only documents after 2025," it isn't a business system. Third, at scale, **speed** becomes the issue.

The dedicated store that solves all three (persistence, filters, scale) is the vector database. You will hear names like Chroma, FAISS, and pgvector in adoption-review meetings — build the internals yourself today, and you will know exactly "what you are buying" when you choose.

## 2. Grasping It Through an Analogy

If a regular database is **a library shelved by title and author** (you must know the exact title), a vector DB is **a library where books on similar topics sit next to each other**. Say "give me books like this one," and the librarian shows you the shelf around where that book lives.

- **Indexing (add)** = when a new book arrives, read it (embed) and shelve it in the right spot.
- **Search** = pull the books nearest to the question's coordinates.
- **Filter (where)** = "but only from the company-policy shelf" — restricting scope by label (metadata).
- **Persistence (save/load)** = keep the shelf map, so tomorrow you find things instantly without re-shelving.
- **Approximate search (ANN)** = with tens of millions of volumes you cannot walk every shelf, so divide the library into zones and search **only the promising zones** — very occasionally you miss the single best book, but you are hundreds of times faster.

## 3. Core Concepts

### 3.1 What a vector DB really is = matrix + text + metadata

As today's implementation shows, the essence is three blocks: (1) the embedding-vector matrix (an N×512 NumPy array), (2) the source text for each row, (3) metadata like source, category, and date. Search is one `vectors @ query_vector` matrix product plus a sort.

### 3.2 Metadata filters

Before (or after) the similarity computation, exclude candidates that fail the condition. Today's code sets out-of-filter scores to `-inf`. In production this doubles as a **permission filter** — documents the asker may not see must be excluded from the search itself to prevent leaks.

### 3.3 Persistence — index once, reuse forever

Embedding dominates the total cost, so vectors are saved to disk (today: `.npz` + `.json`). When documents change, re-embed only the changed pieces — incremental indexing is the working pattern. Caution: **changing the embedding model means re-indexing everything** (the coordinate system changes).

### 3.4 Exhaustive vs approximate search (ANN)

Brute-force comparison over 200,000 vectors takes tens of milliseconds in NumPy — meaning today's implementation suffices for small-to-mid internal document sets. At tens of millions of vectors you need Approximate Nearest Neighbor search: partition the vector space into zones (IVF) or build a neighbor graph (HNSW) and explore only the promising parts. Chroma/FAISS are libraries with such indexes built in, and their APIs look remarkably like the `add/search` you build today.

### 3.5 The adoption menu at a glance

The choices come in roughly three tiers. (1) **Roll your own / lightweight**: today's NumPy approach — up to tens of thousands of pieces, single server, minimal dependencies. (2) **Embedded library**: Chroma (easy), FAISS (large-scale, high-performance indexes) — living inside your application. (3) **Extend your existing DB / managed service**: pgvector (add a vector column to the PostgreSQL you already run), or a managed cloud vector DB — when you want operations, backup, and permissions integrated with your existing stack. The criteria are not fashion but **piece count, update frequency, permission requirements, and ops staffing**. Choosing heavy infrastructure for a small workload is the most common waste.

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level07_vector_databases
python3 main.py
```

Reading the output:

- **[1]–[2]** The 22 internal document pieces get indexed with metadata (source/category/id). Confirm the store's substance is a `(22, 512)` matrix.
- **[3]** "What do I need in order to get a repair?" finds the warranty clause of the product manual at rank 1 (similarity 0.621).
- **[4]** The same question with `where={"category": "policy"}` excludes the manuals — only policy documents remain candidates.
- **[5]** After `save()`, a store restored with `load()` is verified to return identical search results (`True`). Artifacts: `outputs/mini_store.npz`, `outputs/mini_store.json`.
- **[6]** Exhaustive search timed over 1k/20k/200k fake vectors (about 0.1 / 4 / 45 ms on a typical laptop), then a tour of the world beyond (ANN).

Code heart: the single `MiniVectorStore` class is everything. Note the filter implementation in `search()` (`np.where(mask, sims, -inf)`) and the two-part persistence in `save/load` (vectors in npz, text/metadata in json). Even when you later use a production vector DB, knowing this structure lets you read instantly what each configuration knob means.

## 5. Try It Yourself

1. **(Easy)** Search within a single document using `where={"source": "policy_vacation.txt"}`. How could you apply this filter for access control?
2. **(Medium)** Add a `delete(chunk_ids)` method. Delete the pieces from one source and confirm the search results change. (Hint: one NumPy boolean mask can filter `vectors`, `texts`, and `metas` together.)
3. **(Challenge)** Fake a simple ANN: at indexing time, split the vectors into 2–4 clusters with k-means (`sklearn.cluster.KMeans` is allowed), and at search time do exhaustive comparison only inside the cluster closest to the query. On the 200k-vector experiment, compare speed and accuracy (top-1 agreement with exhaustive search).

## 6. Common Mistakes

- **Re-embedding on every run** — code that re-embeds the whole corpus at every server restart is the textbook cost leak. Design save/load from day one.
- **Storing vectors without the source text** — what you show a human is the text, not the vector. Manage vector, text, and metadata as one body.
- **Reusing old vectors after switching embedding models** — mixing vectors from different models makes similarity meaningless. Record the model version in metadata.
- **Heavy infrastructure for a small corpus** — with tens of thousands of pieces, today's NumPy store or SQLite/pgvector-class tooling suffices. Measure scale first ([6]), then decide.
- **Company-wide search with no permission filter** — a chatbot that can retrieve the salary table has its incident on launch day. Filters are a security feature, not a convenience.

## Next Level Preview

All the components are ready: embeddings (05) → chunking (06) → the vector store (07). In Level 08 we finally assemble them into the RAG chatbot that answers policy questions citing its source documents — and says "I don't know" when it doesn't.
