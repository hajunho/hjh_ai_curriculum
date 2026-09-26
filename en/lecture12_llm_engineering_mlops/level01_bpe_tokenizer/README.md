# Lecture 12 · Level 01 — Building a Tokenizer — Implementing BPE from Scratch

> One simple rule — "keep gluing together the character pairs that appear side by side most often" — is enough to build an LLM's vocabulary (BPE) from the ground up.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level00 / **Estimated time** 50 min

## 1. Why Learn This — The Business View

Every LLM price sheet is written "per token." The same document can come out two or three times longer or shorter in tokens depending on how the tokenizer (the text-slicing machine) cuts it — and that difference goes straight into your API bill and response latency. Languages other than English get chopped much finer by English-centric tokenizers; Korean text, for instance, routinely costs more than the same question asked in English. To answer "why is our service's LLM bill higher than projected?", you need to know how tokens are made.

One more thing: the tokenizer is a hidden determinant of model ability. As we will demonstrate below, how you split numbers changes a model's arithmetic skill. The root of "the model can't add" may lie not in training but in the knife that slices the data — that is engineering intuition. The BPE (Byte Pair Encoding) we build today is the prototype of the scheme used by most LLMs, including the GPT family.

## 2. Grasping It Through an Analogy

Think of the text-expansion shortcuts in your team's messaging app. At first you type every letter. Then you notice you type "quarterly report" dozens of times a day, so you register a shortcut: "qr → quarterly report." Then you register the next most common phrase, and the next… **Registering shortcuts for the most frequent combinations first, one at a time — that is all BPE is.**

- Starting point: every character is its own token (zero shortcuts)
- Step 1: find the two tokens that most often appear side by side in the corpus and merge them into one (register the first shortcut)
- Repeat: merged chunks become candidates too, so ever-longer pieces form: "e"+"r" → "er", "report"+"." → "report."

The more shortcuts you register (the bigger the vocab), the fewer keystrokes (tokens) you need — but the shortcut list balloons and gets costly to manage. This trade-off is exactly why real LLMs agonize over vocab sizes between 30k and 250k.

## 3. Core Concepts

### 3-1. The BPE Training Algorithm

1. Split each corpus word into characters and append an `</w>` end-of-word marker. (This distinguishes the same letters in the middle of a word from those at the end of one.)
2. Count the frequency of every adjacent token pair.
3. Merge the most frequent pair into a single new token and record that rule.
4. Repeat steps 2–3 for as many merges as you want.

What remains after training is an **ordered list of merge rules**. To tokenize a new sentence, split it into characters and re-apply the rules in the order they were learned. Because rule order is priority, the same vocab applied in a different order gives different results.

### 3-2. The Vocab-Size Trade-off

| Merge count (= vocab size) | Pros | Cons |
|---|---|---|
| Few (small vocab) | small embedding table; every token appears often, so it trains well | sentences get long → higher inference cost and latency |
| Many (large vocab) | shorter sentences → more content per context window | rare tokens get too few training examples; embedding parameters balloon |

This table also explains why rebuilding a tokenizer on your own language's corpus (say, Korean for a Korean-language service) slashes token counts: the frequent combinations of that language make it into the vocab.

### 3-3. Digit Splitting

Run BPE naively and whatever digit combinations are frequent in the corpus become whole tokens. With lots of price data, chunks like "00" and "500" appear. The problem: **the same kind of number gets split inconsistently**. If 987 splits as "9"+"87" while 986 stays whole as "986", the two numbers look like totally different symbols to the model, and it cannot learn the concept of place value. That is why modern LLMs add a preprocessing rule that forces digits apart into single (or at most three-digit) pieces. One slicing rule changes the model's math grades.

## 4. Hands-On — main.py

```bash
cd lecture12_llm_engineering_mlops/level01_bpe_tokenizer
python3 main.py
```

`[1]` loads tiny_corpus (84,505 characters, only 28 unique characters) and `[2]` runs 60 BPE merges, printing the early merge steps as they happen. Watch how common endings and words clump together: "e + r → er", "Thi + s → This", "prepar + ed → prepared". `[3]` then cuts the same sentence with four tokenizers (0/10/30/60 merges) and shows the token count falling from 29 to 17 — by merge 60, "report." survives as one whole token. `[4]` is the highlight: we train twice on an amount-heavy corpus, without and with digit splitting, then tokenize "98700". The unsplit tokenizer produces the lumpy `['9', '8', '700']`, while the split one gives a consistent `['9', '8', '7', '0', '0']`.

The heart of the code is three functions. `count_pairs()` counts adjacent-pair frequencies (the digit_split flag blocks digit merges), `merge_pair()` glues that pair together in every word, and `train_bpe()` loops the two. The principle is identical to production library implementations — only the scale is smaller.

## 5. Try It Yourself

1. **(Easy)** Raise `num_merges=60` to 200. It stops early at 115 merges — find out why in the code. (Hint: tiny_corpus has few unique words, so the supply of pairs to merge runs out.)
2. **(Medium)** Replace the `sample` sentence with one containing words the corpus never saw (e.g., "Today the director reviewed the slides."). Which parts shatter into single characters? (Hint: combinations never seen in training have no merge rules, so they stay character-level. This is why neologisms and jargon eat so many tokens.)
3. **(Challenge)** In `make_price_corpus()`, change `* 100` to `* 1` (trailing zeros become rare) and observe how the digit merges change. Then summarize in one line why the corpus's statistics get engraved directly into the vocab. (Hint: BPE only looks at frequency. The data's habits become the dictionary.)

## 6. Common Mistakes

- **Assuming token = word**: tokens are fragments carved by frequency and do not line up with words. Estimating a bill as "word count × 1" goes badly wrong, especially outside English.
- **Ignoring the order of merge rules**: save only the vocab list and lose the order, and the same sentence tokenizes differently. Training and inference must use the very same rules file.
- **Swapping the tokenizer but keeping the model**: replace the tokenizer and every token id changes meaning, so you must retrain from the embeddings up. This is why "let's just swap in a tokenizer for our language" is not a small proposal.
- **Treating preprocessing (digit splitting etc.) as trivial**: as you saw today, one preprocessing line changes model capability. Always record preprocessing rules in your data-pipeline documentation.

## Next Level Preview

Now that you can wield the knife (tokenization), it is time to stack the sliced ingredients in the warehouse. In level02 we build a miniature of a large-scale training data pipeline: **packing** tokens into fixed-length blocks joined by `<s>...</s>`, saving uint16 binary shards, and memmap loading that opens even multi-hundred-GB files instantly.
