# Lecture 10 · Level 06 — Word Embeddings

> Give words coordinates so that "similar meaning = nearby" — an embedding trained by hand with co-occurrence + SVD.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level04, lecture03 (NumPy) / **Estimated time** 50 min

## 1. Why Learn This — The Business View

In TF-IDF world, "delivery" and "shipping" are unrelated dimensions. A customer
ticket that says "the shipment is late" never finds the "delivery delay"
document. Word embeddings turn words into **coordinates (vectors)** where
similar meanings sit at nearby points. Then search works "even when the letters
differ, as long as the meaning is close", and a classifier can handle a word it
has never seen thanks to its neighbors.

This technique is the foundation of everything today: intranet document search,
recommender systems ("customers who viewed this also viewed" is an embedding),
the similar-document retrieval inside RAG chatbots (lecture11), and the very
first layer of an LLM is an embedding layer. Without knowing what an embedding
is, you'll miss half of any modern AI product meeting.

## 2. Grasping It Through an Analogy

An embedding is **making a map of words**. Pin every branch office on a map and
you can see "Downtown and Riverside are close"; pin words on a map and you can
measure "delivery and shipping are close, delivery and lasagna are far" as
distance.

But how do we assign each word its coordinates? Enter one of linguistics'
oldest insights: **"you shall know a word by the company it keeps"**
(the distributional hypothesis). "Shipping" and "delivery" both show up next to
"late, fast, arrived, box". Similar friends → similar meaning — so build the
coordinates from **the statistics of co-occurring words**. It is exactly the
intuition from level00, where neighboring words disambiguated "bank".

## 3. Core Concepts

### 3-1. The co-occurrence matrix — a friendship ledger

Sweep through the corpus counting "how many times did word B appear within the
window around word A", producing a vocabulary × vocabulary table — the
co-occurrence matrix. One row is that word's "friendship profile", and two words
with similar profiles can be considered similar in meaning.

### 3-2. PPMI — separating chance seating from real friendship

Common words like "a" and "the" sit next to everyone. Sitting beside someone at
the company anniversary dinner doesn't make you friends; raw counts are weak
evidence. PPMI (positive pointwise mutual information) measures how much more
often two words co-occur than **chance alone would predict**, keeping only the
beyond-chance associations. Same spirit as IDF discounting common words.

### 3-3. SVD — compressing the big ledger into small coordinates

With a 10,000-word vocabulary the co-occurrence matrix has 10,000 dimensions.
SVD (singular value decomposition) keeps only **the few most important axes**,
compressing to low-dimensional coordinates (the same family as PCA from
lecture06). Compression shakes off fine-grained noise, leaving the "major axes
of meaning" — similarity often gets *better*. The resulting dense vectors are
word embeddings.

Note: the famous word2vec learns embeddings by training a neural network to
"guess the surrounding words", and is known to be mathematically close kin to
compressed co-occurrence statistics. Today's "co-occurrence + PPMI + SVD" is
the classic method that shows the principle most transparently.

### 3-4. Cosine similarity and neighbor search

Closeness of two words in embedding space is measured with cosine similarity.
Listing a word's neighbors (most similar words) is the basic quality check for
an embedding: plausible neighbors → good embedding; nonsense → suspect the
data, preprocessing, or dimension count.

Know the limit, too: this method gives each word **one** coordinate, so the
river "bank" and the money "bank" get squashed onto a single point. Contextual
embeddings — coordinates that shift with context — are what the transformers of
levels 08–10 deliver.

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

The material is tiny_corpus (a mini English corpus with repeating sentence
templates, about 2,000 sentences / 84,505 characters).

- **[1] Corpus prep**: tokenize (periods stripped) → 14,816 tokens, 28
  vocabulary types. Top frequencies: 'a' (3,583), 'This' (791), 'Over'/'the'/
  'weekend' (429 each).
- **[2] Co-occurrence matrix**: a 28×28 matrix at window size 2. The row for
  "student" shows its main seatmates: a(808x), checked(86x), fixed(86x) — the
  article and verbs that flank a subject noun.
- **[3] PPMI transform**: 'student'–'a' co-occur 808 times, yet PPMI is only
  0.73 — chance seating with an article that sits next to everyone gets
  discounted.
- **[4] SVD to 8-dim embeddings**: numpy's SVD compresses 28 → 8 dimensions;
  the top 8 singular values hold 71% of the total information.
- **[5] Similar-word search**: actual neighbors —
  `student → chef(1.00), teacher(1.00), developer(1.00)`;
  `report → program(1.00), letter(1.00), proposal(1.00)`;
  `made → fixed(1.00), checked(1.00), organized(1.00)`;
  `Yesterday → Today(0.98)`. Subjects gather with subjects, objects with
  objects, verbs with verbs.
- **[6] Embedding map PNG**: the 2-D scatter saves to `outputs/embedding_map.png`.
  See with your own eyes whether nouns, verbs, and time expressions claim
  their own regions.

## 5. Try It Yourself

1. **(easy)** In [5], search the neighbors of other words ("stew", "weekend").
2. **(medium)** Shrink the window to 1 or grow it to 4 and compare the neighbor
   results. Narrow windows tend to emphasize grammatical role; wide windows,
   topical association.
3. **(challenge)** Vary the embedding dimension across 2/8/16/32 and compare the
   neighbor quality of "student". Too few dimensions smears everything together;
   on a corpus this small, too many adds noise back — the dimension count is a
   hyperparameter chosen together with data size.

## 6. Common Mistakes

- **Big expectations from a small corpus**: embedding quality depends heavily on
  data volume. Today's mini corpus demonstrates the principle; real embeddings
  train on hundreds of millions of words.
- **Over-reading similarity**: "close" means "appears in similar contexts", not
  "synonym guaranteed". Antonyms ("fast/slow") share contexts and often come
  out close.
- **Dot-product similarity without normalization**: frequent words have long
  vectors that distort similarity. Use cosine (normalize, then dot).
- **Forgetting polysemy**: with one vector per word, ambiguous words like
  "bank" become the average of their senses. That's the signal that contextual
  embeddings are needed.

## Next Level Preview

Embeddings gave words coordinates, but a sentence is also an **order** of words.
In level07 we build an RNN in torch that reads while remembering order, and
train a mini language model that predicts the next character of a sequence.
