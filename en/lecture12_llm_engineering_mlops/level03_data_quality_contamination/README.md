# Lecture 12 · Level 03 — Data Quality, Deduplication, and Contamination Checks

> Garbage in, garbage out — we detect duplicated training data and "leaked exam questions" (contamination) ourselves, using n-grams.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** Level 02 (data pipelines and packing) / **Estimated time** 50 min

## 1. Why Learn This — The Business View

Surprisingly, the place where LLM companies spend the most people and money is not model code but **data cleaning**. Model architectures are published in papers and differ little between companies; "what data went in, and how clean it was" is each company's trade secret and the true source of capability differences. The industry's oldest maxim, GIGO (Garbage In, Garbage Out), has only grown more powerful in the LLM era — a model can learn only what is in its data.

The two quality problems with the biggest business impact are this level's subject. First, **duplication**. Web-scraped data contains the same article copied hundreds of times. Train repeatedly on the same document and the model memorizes it wholesale, loses the chance to learn varied expression, and burns training budget (GPU hours = money) for nothing. Second, **benchmark contamination**. If the exam questions used to evaluate a model leak into its training data, the model scores through "memorized answer keys" rather than ability. A company that adopts a model on the strength of a contaminated benchmark score is in for a shock when real-world performance falls short. If you are evaluating a model for adoption, you must be able to ask: "Was this score decontamination-checked?"

## 2. Grasping It Through an Analogy

**Deduplication is like tidying the new-hire training folder.** Imagine the folder contains the same manual 200 times under different file names. The new hire spends most of the onboarding reading the same manual 200 times. They will be able to recite it word for word — but they never learn the rest of the job. What the training coordinator must do is obvious: keep one copy.

**Benchmark contamination is exam leakage.** A test-prep school advertises "98% pass rate" — but it turns out their textbook contains actual past exam questions, with the answers. That pass rate proves nothing about the school's teaching. The student (the model) did not understand the problems; it memorized the answers. In the LLM world this genuinely happens: scrape the whole web and you sweep in the blogs and forum posts where benchmark questions and answers were published. That is why serious model teams always run an **overlap check** between training data and evaluation items.

But how do you compare billions of documents against each other? Comparing full documents character by character is too slow, and it misses "almost-copies" where a single word changed. The tool for this is the **fingerprint**. Like a human fingerprint, treat the set of small pieces (n-grams) cut from a document as its fingerprint: two documents whose fingerprints largely overlap are almost certainly near-identical in content.

## 3. Core Concepts

### 3-1. n-grams and Shingles

An n-gram is a length-n consecutive slice of text. Cutting "team call" into character 3-grams gives "tea", "eam", "am ", "m c", " ca", "cal", "all". The set of n-grams drawn from a document is called its shingle set. The key intuition: **change a document slightly and most of its n-grams survive unchanged.** So if two documents' n-gram sets overlap heavily, they are "nearly the same document."

### 3-2. Jaccard Similarity

How much two sets A and B overlap is measured by Jaccard similarity.

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

The size of the intersection (pieces both have) divided by the size of the union (pieces either has). Identical documents give 1; no overlap gives 0. In practice, anything above roughly 0.7–0.8 is judged a "near-duplicate" and only one copy is kept.

| Situation | Jaccard (roughly) | Verdict |
|---|---|---|
| Exact copy | 1.0 | duplicate — remove |
| Copy with a word or two changed | 0.8–0.95 | near-duplicate — remove |
| Different article, same topic | 0.1–0.3 | keep |
| Unrelated article | 0.0–0.05 | keep |

### 3-3. At Scale: MinHash — Concept Only

With billions of documents you cannot compare every pair (billions × billions). Real pipelines use **MinHash/LSH** to shortlist only the pairs "likely to have high Jaccard" for comparison: summarize each shingle set as a handful of hash numbers, and precisely compare only documents whose summaries look similar. Our exercise has just a few dozen documents, so we honestly compare all pairs — but the principle (shingles + Jaccard) is identical to the large-scale technique.

### 3-4. Contamination Checks — Is the Exam in the Textbook?

Contamination checking points the other way. Deduplication looks for overlap *within* the training data; contamination checking looks for overlap *between* the evaluation items and the training data. The method is simple — cut each benchmark item into n-grams and count how many of those pieces appear in the training corpus. If a substantial share of an item's n-grams (say, 30% or more) exists verbatim in the corpus, flag it as "leak suspected." Once contamination is confirmed there are two responses: before training, drop the offending documents from the data (decontamination); after training, report a separate "clean score" that excludes the contaminated items.

### 3-5. How Contamination Inflates Scores

A model can memorize sentences from its training data wholesale. A leaked item is "a sentence it has seen," so it just continues with the answer — accuracy becomes abnormally high. In this exercise we simulate the effect with a hypothetical "memorizer-only model" — see for yourself how far apart the with-leak and without-leak scores land.

## 4. Hands-On — main.py

How to run:

```bash
cd lecture12_llm_engineering_mlops/level03_data_quality_contamination
python3 main.py
```

The output flows through five stages.

- **[1] Corpus construction** — builds 24 documents from `hjh_data.tiny_corpus()` sentences and deliberately plants one "exact copy," one "near copy with a single word changed," and one "leaked document" containing a benchmark item verbatim.
- **[2] Duplicate detection** — computes Jaccard similarity for every document pair and reports pairs at or above the 0.7 threshold. Confirm that exactly the two planted copies are caught, at similarities of 1.000 and 0.832.
- **[3] Deduplication** — drops the later document of each caught pair: 24 → 22 documents remain (dropped: 15, 19).
- **[4] Contamination check** — for each of the 6 benchmark items, computes "the share of the item's n-grams present in the corpus." The clean items sit at 3–5% overlap while the leaked item hits 100% and gets flagged.
- **[5] Score inflation demo** — the memorizer-only model scores 33.3% with the contaminated item included versus a 20.0% clean score. One leaked question visibly inflates the number.

The core of the code is three functions. `char_ngrams(text, n)` builds the set of character n-grams from a string, `jaccard(a, b)` computes the Jaccard similarity of two sets, and `contamination_ratio(question, corpus_grams)` finds the share of an item's n-grams contained in the corpus's n-gram set. All three are under 10 lines — the skeleton of production-scale pipelines rests on these same simple principles.

## 5. Try It Yourself

1. **Threshold experiment** — lower `DUP_THRESHOLD` from 0.7 to 0.3 and rerun. Do non-duplicate documents start getting caught? (Hint: these documents are built from the same word ingredients, so at low thresholds "different articles on the same topic" get swept in too. The threshold is a precision/recall trade-off.)
2. **n-size experiment** — shrink the `char_ngrams` n from 5 to 2: how do the Jaccard values shift overall? And if you raise it to 12? (Hint: smaller n means more common pieces, raising every pair's similarity; larger n catches only exact copying.)
3. **Craft a sneaky leak** — slightly reorder the words of the benchmark item in the leaked document, or swap a preposition, and check whether the current checker misses it. If it does, how would you shore it up? (Hint: reduce n, compare after stripping whitespace, or add word-level n-grams. Real teams struggle with paraphrased contamination for exactly this reason.)

## 6. Common Mistakes

- **Thinking about dedup after training** — duplicates must be removed *before* training. There is essentially no way to extract memorization from a model that has already memorized.
- **Checking only exact matches** — hash comparison misses a copy with even one character changed. Catching near-duplicates requires the n-gram + similarity approach.
- **Trusting benchmark scores unconditionally** — build the habit of asking whether a slide-deck score was decontamination-checked and whether a clean score is reported separately.
- **Picking the n-gram n carelessly** — too small and false positives explode (every common phrase matches); too big and you miss lightly edited copies. Set it by experiment, based on your data.
- **Declaring all duplication bad** — high-quality data is sometimes deliberately trained on multiple times (multi-epoch). The problem is *unintended, uncontrolled* duplication.

## Next Level Preview

The data is clean — time for the main event. In Level 04 we build a small two-layer decoder transformer from scratch and run **pre-training** on tiny_corpus ourselves. We only ever ask it to "guess the next character," yet as the steps tick by you will watch grammar emerge with your own eyes.
