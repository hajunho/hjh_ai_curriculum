# Lecture 10 · Level 11 — The Pre-training Paradigm: BERT and GPT

> Contrast "fill-in-the-blank" and "continue-the-text" with mini models, and see why pre-train → fine-tune changed the world.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level10 / **Estimated time** 55 min

## 1. Why Learn This — The Business View

Before 2018, every NLP project started from zero. A sentiment model here, a
search model there — each needing tens of thousands of labeled examples, each
trained separately. Labeling cost strangled projects.

The pre-training paradigm flipped this. Learn language ability first from
internet-scale text using **tasks that need no labels** (fill-in-the-blank,
continuation), then adapt to each job with only a small amount of data
(fine-tuning). It is the shift from "hire a blank slate and teach everything,
every time" to "hire an experienced professional with language and common
sense, and give only job training". Today's LLM adoption decisions — fine-tune
or prompt-only, which model family — are all choices *on top of* this paradigm.
Know the big picture and the choices get easier.

## 2. Grasping It Through an Analogy

Compare two study methods.

- **BERT's method — the fill-in-the-blank workbook**: to fill the blank in
  "travel expenses are reimbursed only with [blank] approval", you must read
  **both sides**. It builds deep reading comprehension of the whole sentence.
  But ask this student to "write an essay" and they freeze — they've only ever
  filled blanks, never continued a text.
- **GPT's method — continuation drills**: keep writing after "travel expenses
  are reimbursed...". Predicting what's next from what came before makes
  **composition (generation) the day job**. And because continuing well
  eventually demands comprehension too, scaling it up brought understanding
  along for the ride.

The shared trait is the essence: with both methods, **nobody has to write the
workbook**. Take any text, hide or cut a piece, and you have a problem. The
whole internet became a free workbook, and the "data labeling" bottleneck
vanished. This is self-supervised learning.

## 3. Core Concepts

### 3-1. The two tasks, precisely

- **Masked language modeling (MLM, the BERT way)**: hide some tokens (15% in
  the original paper) behind `[MASK]` and recover them. Attention is
  bidirectional — both sides of the blank are visible (level09's encoder).
- **Causal language modeling (CLM, the GPT way)**: predict the next token at
  each position. The causal mask allows only the left side (the one you built
  in level10).

Same transformer parts; only the mask direction and the task differ. That small
difference produces the two temperaments: "understanding specialist" and
"generation specialist".

### 3-2. Pre-train → fine-tune → (these days) prompt

- **Stage 1, pre-training**: acquire language ability from massive unlabeled
  text. Expensive, but paid once, by the model maker.
- **Stage 2, fine-tuning**: teach the pre-trained model a specific job with a
  small labeled set. What once needed tens of thousands of examples now needs
  hundreds.
- **Stage 3 (the LLM era), prompting / in-context learning**: once models grew
  big enough, the ability emerged to perform new tasks from instructions and
  examples alone, without touching the weights. Lecture11 covers this properly.

### 3-3. Why the world changed — three reasons

1. **The labeling bottleneck died**: the era when data labeling was the top
   cost ended.
2. **Transfer**: language ability learned once transfers to translation,
   summarization, classification, QA — every task. Per-task models → a
   general-purpose foundation model.
3. **Scaling laws**: growing data, model, and compute raises performance
   predictably, and beyond certain scales abilities nobody trained for
   (reasoning, instruction following) appeared. This is why "bigger" became a
   research strategy.

### 3-4. Which way won — and what remains

The side that could generate sparked the interface revolution (chatbots), so
the GPT way became mainstream. But the BERT way isn't dead — wherever
"understanding" is the goal (search, classification, producing embeddings),
bidirectional encoders remain the standard. The models that embed documents in
lecture11's RAG are exactly their descendants. "Generation → decoder;
understanding & retrieval → encoder" maps the practical landscape roughly right.

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

Two mini transformers of identical size train on tiny_corpus, each on a
different task, to demonstrate the difference in temperament. Both trainings
together stay under 30 seconds on CPU (about 11s actual).

- **[1] The problem generator**: from the same sentence ("Over the weekend a
  chef..."), an MLM problem (mask 15%) and a CLM problem (shift by one) are
  built — labels appear out of thin air. Word vocabulary 30 types, 12,000
  training tokens.
- **[2] Training the two models**: encoder(bidirectional)+MLM reaches loss
  0.700 and decoder(causal)+CLM reaches 0.791 after 300 steps each.
- **[3] The fill-in-the-blank exam**: `[MASK] a student made a report .` — the
  BERT-style model answers with time-flavored words ('weekend' 41%,
  'Yesterday' 14%, 'evening' 11%), while the GPT-style model, blind to the
  right context, bets 99% on '.' — it can basically only guess.
- **[4] The continuation exam**: from "Over the weekend a chef", the GPT-style
  model writes `organized a pot of stew . Yesterday a` — template intact —
  while the BERT-style model stumbles out `a weekend a report . Over the
  weekend`.
- **[5] A taste of fine-tuning**: a character-level decoder pre-trains on
  unlabeled review text (loss 3.800 → 0.267 over 800 steps); then a logistic
  head learns sentiment from just 60 labeled reviews. Test accuracy:
  pre-trained 83.0% vs random-weights control 78.5% — the pre-trained
  representation learns more from the same few labels.

## 5. Try It Yourself

1. **(easy)** Write your own mask problems for [3]. How does the gap between
   the two models differ when you mask the sentence's last word versus a middle
   word?
2. **(medium)** Raise the MLM mask ratio from 15% to 40% and retrain. As the
   problems get harder, what happens to the loss and the fill-in accuracy at
   the same step count?
3. **(challenge)** In [5], vary the fine-tuning data across 20/60/200 examples
   and plot the "pre-trained vs not" gap. The scarcer the data, the more
   pre-training is worth — which is exactly why companies consider fine-tuning.

## 6. Common Mistakes

- **Memorizing BERT/GPT like brand names**: the essence is the task design —
  "bidirectional + blanks" vs "unidirectional + continuation". Remember the
  task, not the name.
- **Generalizing from a mini experiment's numbers**: today's figures
  demonstrate the principle. The real gaps and abilities appear at scales tens
  of thousands of times larger.
- **Believing "pre-training = something our company must also do"**:
  pre-training is the model maker's job. For most companies the menu is
  prompting or fine-tuning.
- **Giving generation tasks to understanding models and vice versa**: using a
  generative LLM for search embeddings, or an encoder for a chatbot, loses on
  both cost and quality.

## Next Level Preview

Lecture10 ends here. We started from the failures of string matching and
arrived at the principles of GPT. In the next lecture, **lecture11 — Working
with LLMs and RAG**, we build on this foundation and switch from the "maker's"
side to the "user's" side: prompt engineering, embedding search, and building a
RAG chatbot. Those tools are no longer black boxes to you.
