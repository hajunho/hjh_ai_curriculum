# Lecture 11 · Level 08 — The RAG Chatbot: Assembling Retrieve, Augment, Generate

> Assemble embeddings, chunking, and the vector store into a chatbot that answers policy questions citing its sources — and says "I don't know" when it doesn't.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level05–07 / **Estimated time** 60 min

## 1. Why Learn This — The Business View

Level00 showed the LLM's two built-in limitations: it knows nothing about your company (knowledge cutoff), and it bluffs about what it doesn't know (hallucination). The standard design that softens both at once is RAG (Retrieval-Augmented Generation). Internal policy chatbots, automated customer support, contract Q&A, technical-documentation assistants — most enterprise LLM deployments are, in effect, RAG.

RAG's business appeal comes in three parts. (1) **No retraining** — update the documents and the knowledge updates. (2) **It cites sources** — "according to the vacation policy, Article 1" makes verification and accountability possible. (3) **It can be designed to say "I don't know"** — with no evidence, it refuses to answer, structurally reducing hallucination.

## 2. Grasping It Through an Analogy

Think of an **open-book exam**. In a closed-book exam (a bare LLM), fuzzy memory produces plausible wrong answers. An open-book exam (RAG) is different.

1. Read the question and **find the relevant pages in the textbook** — Retrieval.
2. **Spread those pages open next to your answer sheet** — Augmentation.
3. **Write only from what lies open, with footnotes giving the page numbers** — Generation.

And the iron rule of the exam: **for a question the book doesn't cover, a blank answer beats an invented one.** If the internal chatbot invents next year's minimum wage, that is not a feature — it is an incident. "Not confirmed in the documents; please contact the responsible department" is the correct answer.

## 3. Core Concepts

### 3.1 The whole pipeline

```
[indexing time (ahead)]  documents → chunking (06) → embedding (05) → vector store (07)
[question time (live)]   question → ① retrieve: top-k pieces → ② augment: insert into prompt
                          → ③ generate: LLM answers only within the evidence + cites sources
```

What's new is the joint between ② and ③; ① is reuse of parts you already built.

### 3.2 The anatomy of the augmented prompt

The prompt printed in the lab is the standard shape: (a) role assignment, (b) the constraint **"answer ONLY from the evidence documents"**, (c) the instruction to say so when the evidence is silent, (d) the retrieved pieces (with sources), (e) the question. (b) and (c) are the anti-hallucination clauses — without them the LLM blends the evidence with its own memory.

### 3.3 The "I don't know" threshold

If the top retrieval similarity falls below the bar (0.4 today), we skip generation and respond with a polite refusal plus a pointer to the responsible team. Set the bar too high and answerable questions get refused (annoying); too low and the bot answers from irrelevant evidence (dangerous). In production this value is tuned against an evaluation set (level11).

### 3.4 top-k and the context budget

How many pieces go into the prompt (top-k) is a cost/quality trade-off. More pieces mean fewer misses but higher token costs — and irrelevant pieces blur the answer. 3–8 is the usual starting range; the irrelevant-piece problem is what level09's reranking and MMR improve.

### 3.5 Two operations questions that always come up

**"What about follow-up questions?"** A follow-up like "and what about carry-over?" fails if searched as-is. In practice you rewrite it into a self-contained question using the previous turns ("what is the annual-leave carry-over policy?") before retrieval — and this rewriting is usually delegated to the LLM too. **"What if documents conflict?"** When old and new versions of a policy both get retrieved, prefer the latest by metadata (revision date) or exclude retired documents from the index. Notice both are design problems in the stages **around retrieval**, not in generation — that is the character of operating RAG.

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level08_rag_chatbot
python3 main.py
```

Reading the output:

- **[1]** 5 documents → 22 pieces indexed. The "'I don't know' similarity threshold 0.4" is shown alongside.
- **[2]** The first question (leave-request deadline, best similarity 0.875) exposes the entire process: the retrieval trace (kept/dropped with similarities), **the full augmented prompt that would go to a real LLM**, and the source-cited answer.
- **[3]–[5]** The expense, remote-work, and warranty questions each answer from the correct document (best similarities 0.829 / 0.866 / 0.668). In [4], an off-topic-enough piece at 0.396 gets caught by the threshold and marked "dropped."
- **[6]** The key scene: for the out-of-corpus question "What is next year's minimum wage?", the best similarity reaches only 0.379 — the chatbot does not invent; **it says it doesn't know** and points to the responsible department.

Code heart: the three methods of `RAGChatbot` are the pipeline itself — `retrieve()` (matrix-product search), `build_prompt()` (augmented-prompt assembly), `answer()` (threshold check, then generate or refuse). Generation is imitated by `MockLLM.answer_with_context()`, which reassembles the evidence sentences; as the bottom comment shows, with a real API you keep the retrieval/augmentation code untouched and swap only the generation call.

## 5. Try It Yourself

1. **(Easy)** Add a question: "What do I do to take my company laptop home?" Which pieces get kept? (Hint: check whether the remote-work policy's "taking company assets off-site" sentence is captured.)
2. **(Medium)** Rerun the 5 questions with `min_score` at 0.2 and at 0.6. What side effect appears in each case (answers from irrelevant evidence / excessive refusals)? Observe the trade-off between the two error types.
3. **(Challenge)** Append an automatic footer to every answer — "This answer was based on N documents; confirm the latest policy with the responsible team" — and add a "low confidence" warning when the average similarity of the kept pieces is below 0.5.

## 6. Common Mistakes

- **Omitting the "answer only from the evidence" instruction** — bolting on search alone does not make RAG. Without this clause, the model blends evidence with imagination.
- **Always answering, with no "I don't know" path** — retrieval always returns *something* at rank 1. Answering plausibly from low-similarity evidence is the main source of RAG hallucinations.
- **Skipping source citations** — an uncited answer cannot be verified. Citations are both a user-trust matter and the ops team's debugging tool.
- **No index-refresh plan** — a chatbot answering from a superseded policy is worse than no chatbot. Design the update procedure (document change → re-chunk → re-embed) together with the bot.
- **Blaming the model for retrieval problems** — when an answer looks wrong, look at the retrieved pieces first. Nine times out of ten a wrong piece got in or the right piece didn't.

## Next Level Preview

The chatbot works, but retrieval sometimes disappoints — it is weak on questions that use exact terms, and it can fetch stacks of near-identical pieces. Level 09 lifts retrieval quality with hybrid search (keyword BM25 + semantic) and the MMR diversity technique.
