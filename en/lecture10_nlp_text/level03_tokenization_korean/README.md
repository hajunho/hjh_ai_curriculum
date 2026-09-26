# Lecture 10 · Level 03 — Tokenization and the Quirks of Korean

> What unit should a sentence be split into? We build and compare whitespace, character, n-gram, morphological, and subword strategies — and see why some languages make this far harder than others.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level01, level02 / **Estimated time** 45 min

## 1. Why Learn This — The Business View

Every text analysis begins with the decision "what unit do we count sentences
in?". That unit is called a token, and the splitting is tokenization.
How you tokenize changes everything downstream: search quality, classification
accuracy, keyword statistics. If someone types "expect" into your intranet
search and "expectations" doesn't come back, nine times out of ten it's a
tokenization problem.

It also connects directly to your LLM invoice. GPT-family APIs bill by token
count, and the same content costs more tokens in some languages than others —
English tends to be cheap, Korean expensive. Why that happens is exactly the
last topic of this level (subwords).

## 2. Grasping It Through an Analogy

Tokenization is **slicing a sandwich roll**. The same roll becomes more or less
useful depending on how you cut it.

- **Whole** (sentence level): easy to carry, hard to share.
- **Bite-sized** (word level): about right for most purposes.
- **Minced** (character level): anyone can swallow it, but the flavor (meaning) scatters.
- **Taken apart by ingredient** (morpheme level): tomato here, cheese there —
  the most work, but you know exactly what's inside.

English is a roll with the knife guides (spaces) already scored in, so it slices
easily — though the guides lie sometimes: "don't" is one visual chunk hiding two
words, and "fast," carries a comma stuck to it. Korean is a roll where the
filling (stem) and the wrapping (particles and endings) are rolled up tight
together, so cutting along the guides means chewing the wrapper too.

## 3. Core Concepts

### 3-1. Whitespace tokenization — a good default with sneaky failures

English "I love this product" splits on spaces and you're nearly done.
But the corners bite:

- **Punctuation sticks**: "fast," and "fast" become different tokens.
- **Casing splits**: "Delivery" (sentence-initial) and "delivery" become
  different tokens.
- **Contractions hide words**: "don't" is *do + not*; "store's" is
  *store + 's*.
- **Hyphenation wobbles**: "e-mail", "email", and "e mail" are three spellings
  of one word.
- **Inflection scatters**: "expect / expected / expectations" — one concept,
  three strings (as level00 showed).

Whitespace alone leaves the vocabulary needlessly bloated, with the same
concept scattered across surface variants.

### 3-2. Character tokenization and n-grams — simple but surprisingly strong

Split by character — "e/x/p/e/c/t" — and the unknown-word (new slang) problem
vanishes: every word always splits into characters. The price is that a single
token carries almost no meaning. The letter "e" belongs to everything.

The compromise is the **n-gram**: overlapping windows of n consecutive
characters. The 2-grams of "expect" are "ex, xp, pe, ec, ct". Meaningful chunks
sometimes get captured while unknown words stay handleable, which is why search
engines have long used n-grams for fuzzy matching (and why they are essential
for languages without spaces, like Chinese and Japanese). The cost: token counts
balloon, and meaningless fragments like "xp" ride along.

### 3-3. Morphological analysis — taking words apart into units of meaning

A morpheme is the smallest unit of language that carries meaning.
"expectations" analyzed into morphemes:

```
expect (stem) + ation (noun-forming suffix) + s (plural suffix)
```

Do this and "expected / expecting / expectations" all collapse onto the stem
"expect-", and "package / packages / package's" onto "package". Counting,
search, and classification all step up a grade. In English practice, this is
the job of stemmers and lemmatizers (Porter stemmer, WordNet lemmatizer, and
friends in open-source toolkits).

**The contrast case — Korean.** English inflection is a handful of suffixes;
Korean is an **agglutinative language**, where a stem takes chains of particles
and endings: the verb "meok-" (eat) surfaces as *meogeotda, meokneunda,
meogeosseumnida, meokgo...* — dozens of forms, and nouns carry case particles
glued straight on. For Korean, morphological analysis isn't a refinement, it's
survival: skip it and every noun splinters into four or more strings. That is
why the folder for this level carries Korean in its name — it is the textbook
example of tokenization difficulty.

In this level we build a **mini rule-based analyzer** — a small dictionary of
nouns and stems plus suffix/ending lists — to feel the principle in our hands.
It's a toy, but the skeleton is the real thing: *dictionary + rules +
longest-match-first*.

### 3-4. Subwords (BPE) — what modern LLMs chose

Morphological analysis needs a dictionary and grammar rules per language.
The subword approach flips the premise: with zero linguistic knowledge,
**repeatedly merge the character pair that most often appears together** in the
data, and let the token list emerge. That is the BPE (Byte Pair Encoding)
algorithm.

1. Start with every word split into characters.
2. Find the most frequent adjacent pair in the corpus and merge it into one
   piece (e.g. "i"+"n" → "in").
3. Repeat step 2 until the vocabulary reaches the size you want.

Frequent words end up as one long token; rare words as several short pieces.
New coinages are always expressible as pieces, so the unknown-word problem
disappears. GPT-family models use exactly this — and if a language is
underrepresented in the training data (Korean, for instance), fewer merge rules
form for it, so **the same sentence splits into more tokens and costs more**.

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

- **[1] Four-way comparison**: the same review sentence
  ("Delivery was incredibly fast, loved it") split by whitespace / characters /
  2-grams / the mini analyzer — 6 / 33 / 27 / 8 tokens. Note whitespace keeping
  "fast," and capital-D "Delivery", while the analyzer yields
  `['delivery', 'was', 'incredib-', 'ly(ending)', 'fast', 'lov-', 'ed(ending)', 'it']`.
- **[2] Inside the mini analyzer**: words broken by "longest match first" —
  `packages → [package, s(suffix)]`, `store's → [store, 's(suffix)]`,
  `expectations → [expect-, ations(ending)]` — and the dictionary-less
  "doomscrolling" staying whole (the OOV problem).
- **[3] Vocabulary-size experiment**: 60 hjh_data reviews tokenized per strategy.
  Actual result: whitespace 123 types with the 'expect' family scattered as
  `['expectations', 'expected', 'expected,']`; characters 37; 2-grams 216;
  mini morph 122 types with the family collapsed to `['expect-']`.
- **[4] Mini BPE training**: 40 merge rules learned from the review corpus
  (the first merges are the workhorses 'in', 'el', 'th', 'st', 'er', 'ing'...).
  Then the learned rules tokenize words — and the never-seen compound
  "redelivery" comes out as `['r', 'ed', 'el', 'i', 'ver', 'y']`: pieces, not a
  failure.

## 5. Try It Yourself

1. **(easy)** In [1], change the n-gram n to 3. How do the token count and the
   capture of meaning change?
2. **(medium)** Feed the mini analyzer a sentence containing a word missing from
   its dictionary (`NOUNS`), watch what happens, then add the word to the
   dictionary and see the improvement. This is the OOV problem in your own hands.
3. **(challenge)** Vary the BPE merge count in [4] across 10/30/80 and tabulate
   how the token count of one sentence changes. Hint: more merges → fewer tokens
   but a bigger vocabulary — this trade-off is the central dilemma of real
   tokenizer design.

## 6. Common Mistakes

- **Treating `split()` as the whole answer for English**: it runs fine, but
  "fast," / "Delivery" / "don't" quietly fragment your vocabulary and degrade
  performance. No error is raised, which makes it more dangerous.
- **Using the mini rule analyzer in production**: this level's analyzer is a
  teaching toy ("rating" problems from level01 apply here too). In practice use
  a proven stemmer/lemmatizer — for Korean, a real morphological analyzer — but
  today's principles let you sanity-check their output.
- **Confusing the preprocessing/tokenization order**: tokenize before cleaning
  (level01) and you get tokens like "good!!!". Clean → tokenize, in that order.
- **Never checking the vocabulary size**: whenever you change tokenization,
  eyeball the vocabulary size and the top-frequency tokens. Junk in the top
  ranks means the strategy is wrong.

## Next Level Preview

Now that sentences split into tokens, the next step is turning tokens into
**numbers**. In level04 we implement Bag of Words and TF-IDF with numpy,
verify the results against sklearn, and extract each document's signature
keywords from internal company docs.
