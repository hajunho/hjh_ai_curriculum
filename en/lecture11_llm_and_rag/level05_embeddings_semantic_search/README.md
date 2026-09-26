# Lecture 11 · Level 05 — Embeddings and Semantic Search

> Turn sentences into numeric coordinates (vectors) and you can find documents that "mean the same thing in different words" — the first component of RAG.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level00, lecture03 (NumPy) / **Estimated time** 50 min

## 1. Why Learn This — The Business View

We've all typed "working remotely" into the intranet search box, gotten nothing, and only found the policy after typing the exact phrase "remote work." Classic keyword search only finds **verbatim word matches**. But customers and employees never use the document writer's vocabulary — they say "give me my money back" instead of "refund," and "days off" instead of "annual leave."

Embedding-based semantic search solves this. It converts sentences into "meaning coordinates," so text that is close in meaning gets found even when the phrasing differs. The internal-policy chatbot, automated customer-support answers, finding similar complaints — right up to the RAG chatbot later in this lecture — everything stands on this one component.

## 2. Grasping It Through an Analogy

Think of **product placement in a supermarket**. Ramen, udon, and spaghetti share an aisle. The names differ, but the meaning — "noodles" — is the same. On the store map, the **distance** between two products is their **similarity**.

Embedding is the technique that assigns every sentence such "in-store coordinates" — except the aisles run along hundreds of axes (dimensions), not two. "How many days of annual leave can I take?" and "Employees are granted 15 days of annual leave" are different sentences with nearby coordinates, and both sit far from "the product warranty period." Search then becomes simple: **find the sentences nearest to the question's coordinates.** The ruler that measures how closely two directions align is cosine similarity — the closer to 1, the more same-direction (same-topic).

## 3. Core Concepts

### 3.1 Embedding = text → fixed-length numeric vector

A sentence of any length becomes an array of real numbers of the same length (512 cells today; real models use 1024–3072). The one property that matters: **similar meaning ⇒ nearby vectors.** Thanks to it, search, clustering, dedup, and recommendation all reduce to "computing vector distances."

### 3.2 Cosine similarity

The cosine of the angle between two vectors. Normalize the vectors to length 1 and a single dot product *is* the cosine — so one NumPy matrix multiplication compares thousands of sentences at once (`vectors @ qv`).

### 3.3 Today's mock embedding — character n-gram hashing + co-occurrence

Real embedding models are neural networks, but the principle-teaching `MockEmbedding` uses two classic tricks.

1. **Character n-gram hashing**: extract the 2–3 character chunks of a sentence ("re", "rem", "emo"...) and hash each into one of the 512 cells, counting there. "remote-work" and "remotely" share the chunks of "remot", so their vectors overlap — this is why inflections and hyphenation don't break it.
2. **Co-occurrence expansion**: from the training corpus, it remembers which words often share a sentence ("leave" ↔ "vacation") and mixes the neighbors' chunks in with a small weight when embedding. Even completely different words connect through their topic.

Real neural embeddings capture "context" far more precisely, but the philosophy is the same: **words used in similar places get nearby coordinates.**

### 3.4 The relation to keyword search — complement, not replacement

Semantic search does not always win. Product codes ("A-1042"), proper nouns, and exact article numbers are found more precisely by keywords. That is why practice uses hybrid search that blends the two (level09).

### 3.5 Other uses of embeddings — beyond search

The "similar ⇒ nearby" property serves widely beyond search. (1) **Grouping similar complaints**: cluster hundreds of same-issue complaints to spot problems early. (2) **Duplicate detection**: find nearly identical documents and FAQs to clean up. (3) **Recommendation**: "people who read this document also read...". (4) **Anomaly detection**: flag inputs whose coordinates sit far from the usual inquiries. All of them are the one cosine-similarity computation you learned today, and combined with lecture06's clustering (k-means) they become working tools immediately.

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level05_embeddings_semantic_search
python3 main.py
```

Reading the output:

- **[1]** The 5 documents of `SAMPLE_DOCS` are split into 22 sentences (short headers like "Article 1." count as sentences too).
- **[2]** Every sentence becomes a 512-dimensional vector. The sample header sentence has 14 nonzero cells — the cells its n-grams were hashed into; longer sentences light up more cells.
- **[3]** Three similarity pairs: leave question ↔ leave policy (0.734), leave question ↔ warranty sentence (0.321), warranty question ↔ warranty sentence (0.608). Confirm that same-topic pairs score visibly higher.
- **[4]** The key scene. For "What are the rules for working remotely?", keyword search fails with **0 matching words** (the document says "remote work," not "remotely"), while semantic search ranks all three remote-work-policy sentences on top (0.493 / 0.446 / 0.427). The second question ("When do I need to hand in expense receipts?") is found by both, but semantic search surfaces the related sentences more richly.
- **[5]** The vectors are projected to 2-D with SVD and saved as `outputs/embedding_map.png`. See with your own eyes whether sentences of the same document cluster.

Code heart: the single line `sims = vectors @ qv` in `semantic_search()` is the whole thing — a matrix product of normalized vectors = cosine similarity against every sentence. `keyword_search()` uses the classic "count of exact word matches" (with function words like "the" filtered out, or they would match everything). The difference between the two functions' results is this level's learning goal.

## 5. Try It Yourself

1. **(Easy)** Add a question to [4]: "How much do I get per day on a business trip?". What do keyword and semantic search each find? (Hint: the document says "daily allowance.")
2. **(Medium)** Shrink the dimensions with `MockEmbedding(dim=64)`. How do the similarities in [3] change? With few dimensions, different n-grams collide into the same cell and unrelated sentences drift closer — watch it happen.
3. **(Challenge)** Run [4] without calling `fit()` (no co-occurrence) and compare the rankings. Find out for which questions the co-occurrence expansion earns its keep.

## 6. Common Mistakes

- **Trying to interpret embedding cells by hand** — a cell does not mean "price" or "color." Vectors carry meaning only through their distances to each other.
- **Using a raw dot product as similarity without normalizing** — longer sentences get inflated scores. Normalize to length 1 or use `cosine()`.
- **Comparing vectors from different embedding models** — different models, different coordinate systems. Documents and queries must be embedded by the same model.
- **Semantic-search absolutism** — for code numbers and names, keywords win. The hybrid of level09 is the practical answer.

## Next Level Preview

Today the documents were 22 sentences, so we embedded them whole. Real internal documents run to hundreds of pages. How you cut documents (chunking) before embedding decides your search quality. Level 06 compares three chunking strategies by experiment.
