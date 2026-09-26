# Lecture 11 · Level 06 — Document Loading and Chunking

> "How you cut the document" decides half of your search quality — three chunking strategies compared on the same evaluation set.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level05 / **Estimated time** 45 min

## 1. Why Learn This — The Business View

"Can't we just feed the whole policy binder to the AI?" — the first question in every RAG project. There are two reasons why not. First, the amount an LLM can read at once (the context window) is finite, and you pay per token for everything you put in. Second, and more important — **feeding it whole makes the search blurry.** The embedding of a 100-page document becomes a mushy coordinate meaning roughly "this is the HR policy," useless for finding the specific fact "annual leave must be requested 3 business days in advance."

So we cut documents into reasonably sized pieces (chunks) and embed/search piece by piece. Deciding what "reasonable" means is chunking design — and the share of real RAG quality problems that get fixed purely by revising chunking, with no model change, is astonishingly high.

## 2. Grasping It Through an Analogy

Think of **filing in a records room**. Dump ten years of paperwork into one box (the whole document) and finding anything means rummaging through the entire box. At the other extreme, tear every page in half for storage (chunks too small) and "the contract amount is" and "KRW 120,000,000" land in different envelopes — the information becomes unusable.

Good filing groups by **units of meaning**: one contract, one policy article. And for papers whose boundary is ambiguous, you put **a copy in both folders** — that is overlap. Chunking design is, in the end, "the filing craft of dividing papers so they can be found."

## 3. Core Concepts

### 3.1 Why cut — three reasons

1. **Context-window limits**: the model reads a finite amount at once, and you pay for what you insert.
2. **Search precision**: a piece must carry a single topic for its embedding coordinate to be crisp.
3. **Evidence citation**: answers can cite evidence at the piece level, like "source: vacation policy, Article 1" (levels 08 and 11).

### 3.2 The three strategies

| Strategy | Method | Strength | Weakness |
|---|---|---|---|
| A. Fixed length | chop by character count | 1 minute to implement | words/sentences severed at boundaries |
| B. Fixed + overlap | A plus overlapping spans | patches boundary losses | more storage; still severs |
| C. Sentence/paragraph boundaries | group by meaning units | pieces keep complete meaning | more work; uneven lengths |

The practical default is "C as the base, with a length cap and 10–20% overlap." When structure exists — markdown headings, article numbers — respect those boundaries above all.

### 3.3 The size trade-off

- **Too small**: "The daily allowance is" and "KRW 30,000" split into different pieces, and neither carries a complete answer.
- **Too large**: vacation, expense, and remote-work rules mix into one piece, so a "remote work" question drags along a slab of "vacation" sentences. Search blurs and prompt tokens balloon.

The right size differs per document family, so you set it the way today's lab does: **measure hit rate on an evaluation set.** One more practitioner's trick: store each piece with its document title or article number as a prefix ("[Vacation policy, Art. 1] Employees are granted..."). Even a piece in isolation keeps its context, improving both retrieval and citation.

### 3.4 About loading

In practice, a loading stage precedes cutting: extracting text from PDF, Word, and web pages (tables, headers, and footers are quietly difficult). Today `SAMPLE_DOCS` is already plain text so we skip that stage, but remember the pipeline runs "loading quality → chunking quality → search quality."

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level06_loading_chunking
python3 main.py
```

Reading the output:

- **[1]** Loads the 5 documents (1,637 characters) and recaps the three reasons for cutting.
- **[2]** Piece counts and average lengths per strategy: A gives 16, B gives 22 (more, because of the overlap), C gives 10.
- **[3]** Inspect strategy A's pieces with your own eyes — the word "through" is severed into "...requested th" and "rough the approval system...", boundary damage in the flesh.
- **[4]** Hit rates on the same 5 questions: A scores 4/5, while B (overlap added) and C (sentence boundaries) score 5/5. Overlap and boundary respect visibly buy points.
- **[5]** Per-question details for the best strategy (at which rank the answer piece was found).

Code heart: `chunk_fixed()` (character count + overlap), `chunk_by_sentence()` (group sentences up to a cap while honoring boundaries), and `evaluate()` — embed the pieces into a search engine and grade "answer phrase present in the top 2 pieces." This evaluation style (retrieval hit rate) is the prototype of the level11 evaluator.

## 5. Try It Yourself

1. **(Easy)** Shrink strategy A's `size` to 60. What happens to the hit rate? And if you grow it to 800? (Hint: when one document becomes one single piece, reread [6] for what goes wrong.)
2. **(Medium)** Add 2 new questions to `EVAL_SET` (e.g. "What happens to unused leave?" / "allowance"). Does the ranking between strategies change? Feel how a larger eval set stabilizes the conclusion.
3. **(Challenge)** Build a strategy D of your own: sentence-based grouping where adjacent chunks share their last sentence (a "sentence overlap"). Compare hit rate and piece count against C.

## 6. Common Mistakes

- **Never reading the chunks with your own eyes** — when search misbehaves, nine times out of ten the pieces are broken. `print(chunks[:5])` is the best debugger.
- **Confusing characters with tokens** — in English a token is roughly 4 characters. Leave a margin when budgeting the context window.
- **One chunk size for every document** — policies (article units), manuals (procedure units), and minutes (agenda units) have different natural boundaries.
- **Setting overlap to either 0 or 50%** — zero loses boundaries; excessive overlap duplicates storage and retrieval. Start at 10–20%.
- **Changing the chunking without re-embedding** — new pieces need entirely new vectors. (The vector store of the next level manages this.)

## Next Level Preview

Once you have tens of thousands of pieces, today's "compare against every vector one by one" gets slow. In Level 07 we build the dedicated warehouse for storing and searching vectors — a vector database — from scratch in NumPy, and map the concepts onto real tools like Chroma and FAISS.
