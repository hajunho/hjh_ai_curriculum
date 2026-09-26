# Lecture 11 · Level 09 — Improving Retrieval Quality: Hybrid Search, MMR, Reranking

> The bottleneck on RAG answer quality is almost always retrieval — we implement BM25 from scratch, fuse it with semantic search, and cut redundancy with MMR, comparing before and after.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level08 / **Estimated time** 60 min

## 1. Why Learn This — The Business View

Open the level08 chatbot to real users and the complaints arrive quickly. "I asked using the exact term from the policy and it pulled up the wrong document." "The answer just says the same thing three times." The cause is not generation — it is retrieval. **An LLM cannot produce an answer better than the material retrieval handed it.**

Improving retrieval is a far cheaper and far more certain investment than swapping models. The three techniques in this level — hybrid search (keyword + meaning), MMR (diversity), and reranking (a precise second pass) — are the standard configuration of real enterprise RAG systems, and all three do the same job: **make the list of retrieved pieces better.**

## 2. Grasping It Through an Analogy

Picture a library with **two librarians**. Librarian A (keyword search / BM25) is a meticulous card-index person. Say "unused annual leave allowance payout" and the exact page appears in a blink — but ask "do I get paid for days off I never took?" and A stares blankly, because none of those words are on the cards. Librarian B (semantic search) is quick on the uptake and understands you even when you talk around the subject, but now and then fetches a book that merely *feels* related.

**Hybrid search** is asking both librarians and merging their recommendations. They fail in different places, so merging is insurance.

**MMR** is the rule for the book basket. You can carry four books; if the librarian puts in two that say almost the same thing, one slot is wasted. Picking books that are "relevant *and* different from what's already in the basket" — that's MMR.

**Reranking** is a two-stage hiring process. Résumé screening (a fast first pass) narrows 200 applicants to 30; interviews (precise but slow scoring) pick the final three.

## 3. Core Concepts

### 3.1 BM25 — the standard formula of keyword search

The scoring formula that has been the backbone of search engines for decades. It combines three intuitions.

1. **TF (term frequency)**: the more often a query word appears in the piece, the higher the score — but saturating, so 10 occurrences are not twice as good as 5 (the `k1` parameter).
2. **IDF (rarity)**: matching a rare word like "core hours" counts heavily; matching a common word like "days" counts for very little.
3. **Length normalization**: a long piece contains any given word by chance more easily, so it gets a penalty (the `b` parameter).

### 3.2 Where keyword search trips up in English

English does not glue particles onto nouns the way Korean does, but it has its own mismatch problem: **the same idea gets written in different grammatical shapes.** The policy says "remote work"; the user asks about "working remotely". A keyword index sees two different tokens and scores zero. Add synonyms ("business trip" vs. "travel"), derivations ("request" / "requested" / "requesting"), and British/American spellings, and you have the whole family of keyword-search failures. (Korean is the extreme version of this problem — there the particles and endings fuse onto every noun, so "core hours" and "during core hours" are literally different words. Same disease, higher fever.)

In practice you soften this with stemming, lemmatization, synonym dictionaries, or sub-word (n-gram) indexing. Today's lab deliberately leaves the trap in place — because it is exactly the reason hybrid search exists.

### 3.3 Fusing the two scores

BM25 scores and cosine similarities are on different scales (BM25 runs 0 to 10-plus, cosine 0 to 1). So we min-max normalize each to 0–1 and take a weighted average: `score = α·BM25 + (1−α)·semantic`. You tune α against an evaluation set; 0.3–0.7 is the usual range. (Worth knowing: RRF, which fuses the *ranks* rather than the scores, is also widely used.)

### 3.4 MMR (Maximal Marginal Relevance)

Pick one piece at a time, each time choosing the candidate with the largest value of `λ·(relevance to the question) − (1−λ)·(highest similarity to what is already picked)`. At λ=1 you get plain similarity ordering; the smaller λ gets, the more diversity wins. Internal documents are precisely where MMR pays off, because the same content inevitably appears in the policy, the announcement, and the FAQ.

### 3.5 What to fix first — the improvement order

The practical sequence when retrieval disappoints. (1) First, **look at the search logs with your own eyes** — twenty failed questions are usually enough to reveal the cause (chunking? wording mismatch? duplicates?). (2) If it is chunking, go back to level06; if there are many wording mismatches, add **hybrid search**; if answers repeat themselves, add **MMR**. (3) Reranking has a big effect but adds call cost and latency, so introduce it last, measuring the gain against an evaluation set. The discipline is to fix one bottleneck and measure it, not to apply everything at once.

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level09_retrieval_quality
python3 main.py
```

Reading the output:

- **[1]** We add an "internal notice" whose content overlaps the leave policy, producing a realistic corpus with genuine redundancy in it: 6 documents → 25 pieces.
- **[2]** For the noun-pile query "unused annual leave allowance payout", BM25 puts the right sentence at #1 with a score of 6.473 — this is what keyword search is good at.
- **[3]** The two engines' failure modes. For "What are the rules for working remotely?" BM25 scores **0.000** (the document says "remote work" — no token ever matches) and falls back to "Article 1." from the vacation policy, while semantic search finds the remote-work rule at 0.493. For "What daily amount of money do I get on a work trip?" the roles flip: the literal word "daily" saves BM25 (2.670, correct), while semantic search drifts one sentence over to "Receipts must be submitted within 7 days…".
- **[4]** The O/X table for three questions: BM25 alone gets 2 of 3, semantic alone gets 2 of 3 — but they miss *different* questions, so the hybrid gets all 3. That is the whole point.
- **[5]** MMR before and after. Plain similarity top-4 spends two of its four slots on what is effectively the same sentence (the 3-business-day request deadline, once from the notice and once from the policy). MMR drops the duplicate and secures new information instead — the 15-day grant and the unused-leave payout — inside the same context budget.
- **[6]** An introduction to the two-stage reranking structure.

Code heart: the `BM25` class (`_idf()` plus the TF saturation and length normalization in `scores()`), the weighted sum after `minmax()` normalization, and the greedy selection loop in `mmr_select()` (relevance minus a redundancy penalty). All three are around twenty lines, and the skeleton is the same in production systems.

## 5. Try It Yourself

1. **(Easy)** Change `alpha` in [4] to 0.2 and then 0.8 and watch the O/X table move. If your service gets mostly one type of question, which way would you push α?
2. **(Medium)** Set `lam` in `mmr_select` to 0.9 and then 0.3 and compare the selections in [5]. Watch for the side effect when λ gets too small: "diversity" that is no longer relevant.
3. **(Challenge)** Build a mini reranker: take the top 10 candidates from the hybrid first pass, re-score them with "the overlap ratio between question tokens and piece tokens + the semantic similarity", pick the top 3, and compare against single-pass retrieval.

## 6. Common Mistakes

- **Summing scores without normalizing** — add BM25 (0 to 10+) and cosine (0 to 1) directly and you have, in effect, BM25 alone. Always put them on one scale first.
- **Retiring keyword search once semantic search is in** — quality falls off a cliff on product codes, clause numbers, and proper nouns. They are complements, not substitutes.
- **Trying to solve redundancy only in the index** — tidying documents is good, but duplication across policy, notice, and FAQ is a fact of corporate life. MMR at retrieval time is the realistic fix.
- **Tuning parameters with no evaluation set** — pick α, λ, and top-k by feel and you cannot tell an improvement from a regression. Measure hit rate, as in level06.
- **Forgetting that word forms differ** — a large share of "zero keyword matches" is just grammatical shape. Consider stemming, lemmatization, and n-gram indexing in a real service.

## Next Level Preview

So far the LLM has been a thing that answers questions. In Level 10 we **hand it tools** — a calculator and document search — and implement a mini agent loop that thinks (Thought), acts (Action), and observes (Observation) on its own.
