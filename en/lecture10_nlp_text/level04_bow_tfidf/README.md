# Lecture 10 · Level 04 — Bag of Words and TF-IDF

> The standard technique for turning text into numeric vectors — implemented by hand in numpy and verified against sklearn.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level03, lecture03 (NumPy) / **Estimated time** 50 min

## 1. Why Learn This — The Business View

Machine-learning models eat only numbers. So the gateway to any text analysis is
always "how do we turn a document into a numeric vector?". BoW and TF-IDF are the
decades-old standard for this conversion, and they still earn their keep today:

- **Document search**: find the documents most similar to a query (TF-IDF similarity)
- **Keyword extraction**: automatically pull out "the words that represent this document" for tagging
- **Classifier input**: the sentiment analysis in the next level feeds on TF-IDF vectors

Even in the era of embeddings and LLMs, TF-IDF refuses to die. It is light, fast,
and easy to explain, so "set the baseline with TF-IDF first" remains standard
professional practice.

## 2. Grasping It Through an Analogy

**BoW (Bag of Words)** is grinding a document and pouring it into a bag. Word
order is thrown away; only "which words, how many times" is counted. If the
document is a smoothie, BoW is the ingredient label: delivery ×2, fast ×1,
satisfied ×1... Even with the order gone, the label tells you a lot about what
the smoothie was made of.

**TF-IDF** multiplies the label by **scarcity value**. Think of résumés. "Hard
worker" appears on every application, so it distinguishes nobody; "improved
semiconductor yield by 8%" appears on few, so it defines that candidate. TF-IDF
is exactly this intuition:

- **TF** (term frequency): the more it appears in this document, the more it matters,
- **IDF** (inverse document frequency): the more common it is across other documents, the more its value is discounted.

"A word that appears a lot here (TF↑) that others rarely use (IDF↑)" is the
document's signature keyword.

## 3. Core Concepts

### 3-1. BoW — the document-term matrix

Three documents and five vocabulary words give a 3×5 matrix. Rows are documents,
columns are words, cells are counts.

| | delivery | fast | packaging | price | satisfied |
|---|---|---|---|---|---|
| doc1 | 2 | 1 | 0 | 0 | 1 |
| doc2 | 0 | 0 | 1 | 2 | 1 |

One row of this matrix *is* that document's vector. Characteristically, most
cells are 0 — a sparse matrix. With a 10,000-word vocabulary, each document
becomes a 10,000-dimensional vector, yet it actually contains only a few dozen
distinct words.

### 3-2. The TF-IDF formula — math, just this once

For word $t$, document $d$, total documents $N$, and $df(t)$ documents containing $t$:

$$\text{tfidf}(t, d) = tf(t, d) \times \log\frac{1 + N}{1 + df(t)} + 1 \text{ (one of several variants)}$$

The heart is the fraction inside the $\log$. A word in every document has
$df = N$, the log approaches 0, and the score collapses; a word in only one
document gets a large log and a boosted score. The log is there so that a 10×
increase in document count raises the weight gently, not 10×.

Caution: textbooks and libraries differ in the fine print. sklearn's default has
"smoothing (+1)" and "L2 normalization" switched on, so a hand-rolled version
must use the same variant to compare. main.py reproduces exactly this form and
confirms the match.

### 3-3. L2 normalization — stopping long documents from cheating

Long documents have big TF everywhere, so their vectors are big wholesale.
To compare by content rather than length, each document vector is scaled to
length 1 (L2 normalization). The dot product of normalized vectors is then
cosine similarity — "how alike are these two documents' topics" as a number
between 0 and 1.

### 3-4. The limit of BoW — the price of discarding order

"the quality is good" and "the quality is not good" are nearly the same vector
in BoW world (one word apart). This inability to handle order, context, and
negation is exactly why level06 embeddings and the sequence models from level07
onward exist. Know a tool's limits and the tool stays excellent.

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

- **[1] BoW by hand**: builds the vocabulary and the document-term matrix for 3
  mini documents in numpy and prints it as a table — a (3, 11) matrix with 61%
  zeros (sparsity in action).
- **[2] TF-IDF from scratch**: implements sklearn's default formula (smoothed
  IDF + L2 normalization) in about 10 lines of numpy.
- **[3] Cross-check against sklearn**: the same 80 reviews (115 vocabulary
  types) go through `TfidfVectorizer`; the maximum difference between the two
  matrices prints as `1.11e-16` — machine precision. The moment "what happens
  inside the library" is proven by verification.
- **[4] SAMPLE_DOCS keyword extraction**: top-5 TF-IDF words for the 5 company
  policy/manual documents. The vacation policy surfaces `leave(0.35)`,
  `days(0.23)`, `annual(0.23)`; expenses surfaces `trip(0.38)`,
  `reimbursed(0.25)`; the warranty manual `warranty(0.54)`. Notice the
  freeloaders "of", "to", "the" also sneaking into the top-5 — see section 6.
- **[5] Document similarity**: prints the cosine similarity matrix. Values are
  low overall (the most similar pair is policy_vacation ↔ policy_remote_work at
  just 0.16) because the documents share few literal words. TF-IDF only credits
  likeness when the letters match — the essential limit that motivates level06
  embeddings.

## 5. Try It Yourself

1. **(easy)** In [4], extract the top 8 words instead of 5 and observe what
   climbs into the lower ranks.
2. **(medium)** Drop IDF and rank keywords by TF alone. Watching common filler
   words ("the", "is", "must") take over the top ranks makes IDF's value visceral.
3. **(challenge)** Vectorize the query "business trip expense receipt" the same
   way, compute cosine similarity against the 5 documents, and build a mini
   search engine that returns the best match. Hint: treat the query as one more
   document but reuse the existing documents' IDF — this is the skeleton of
   search engines and of RAG (lecture11).

## 6. Common Mistakes

- **Computing similarity without normalizing**: dot products without
  normalization make the longest document win every similarity contest.
- **Calling a formula difference a "bug"**: TF-IDF has many variants. When
  comparing implementations, check the exact formula in the docs.
- **Rebuilding the vocabulary on new data**: the vocabulary and IDF are fixed at
  training time and applied as-is to new documents. Rebuild them each time and
  the vector dimensions shift, breaking the model.
- **Extracting keywords without stopword handling**: exactly what you see in
  [4], where "of", "to", and "the" gatecrash the top-5. The preprocessing of
  level01 (and a stopword list) is a precondition here too.

## Next Level Preview

Now that documents become vectors, it's time to attach a classifier. In level05
we build a review sentiment analyzer with TF-IDF + logistic regression, then
open up the model coefficients to interpret "which words count as evidence of
positive or negative".
