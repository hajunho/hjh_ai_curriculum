# Lecture 12 · Level 04 — Pre-training a Mini GPT

> We only ask it to "guess the next character," and grammar emerges — build a two-layer transformer from scratch and run pre-training yourself.

**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** Level 03, lecture08 (deep learning foundations) / **Estimated time** 70 min

## 1. Why Learn This — The Business View

This level is the summit of the entire curriculum. We reproduce, in miniature, **pre-training** — the heart of the chatbots you use every day. A real large model trains hundreds of billions of parameters on trillions of tokens for months; we train about 108 thousand parameters on 85 thousand characters in roughly 12 seconds. The scale differs by a factor of millions, but **the principle is literally identical.** Same loss function, same optimizer, same attention equations.

Why should a business professional know this? First, **negotiating power**. When a vendor pitches "we'll build your company its own model," being able to tell whether they mean pre-training, fine-tuning, or prompt engineering lets you judge whether the quote is sane (the costs differ by factors of thousands). Second, **a feel for the limits**. Why models tell plausible lies (hallucinate) becomes self-evident once you know the training objective — the model was trained not "to tell the truth" but "to guess the token likely to come next." Third, every level ahead (LoRA, SFT, DPO) is the story of "how to refine this pre-trained model," so you must pass through this level for the rest to make sense.

## 2. Grasping It Through an Analogy

**Pre-training is how a baby acquires language.** Nobody hands a baby a grammar book. The baby simply hears an enormous amount of adult speech. Then one day it stops saying "want milk" and starts saying "I want some milk." Nobody explicitly taught articles or word order. Given enough exposure to statistical patterns, the rules become **internalized on their own**. A language model is the same. We pose exactly one question, millions of times: "guess the next character." To answer it well, the model has no choice but to learn vocabulary, articles, and word order. Grammar arises not as the goal but as a **by-product**.

Another analogy is **an expanded game of finish-the-sentence**. To guess what follows "Yesterday a student made a ___", you must know that "made" takes an object, that objects here are noun phrases starting with an article, and that a student plausibly makes "a report" but not "a pot of stew"… or does she? Next-token prediction looks simple, but doing it well demands nearly everything there is to know about the language.

## 3. Core Concepts

### 3-1. The Training Objective — Next-Token Prediction

Shift the input sentence by one character to produce the answer key. If the input is "Today a s", the target is "oday a st". Every position simultaneously answers "what's the next character?" The loss is cross entropy — "the lower the probability you gave the correct character, the bigger the penalty." Loss values read intuitively: with 28 character types, pure random guessing has a loss of ln(28) ≈ 3.33. Reaching about 0.20 after training means the model narrows down the next character to near-certainty.

### 3-2. The Decoder Transformer's Anatomy

Our model (TinyGPT) shares its skeleton with the real GPT family.

| Component | Role | Analogy |
|---|---|---|
| Token embedding | character → 64-dim vector | printing a business card per character |
| Position embedding | "which position am I in?" | writing a seat number on the card |
| Self-attention ×2 layers | attend to the relevant earlier characters | looking at the relevant speaker in a meeting |
| MLP ×2 layers | compute over the gathered information | thinking it over after listening |
| Output layer (head) | vector → 28 character scores | a scorecard of next-character candidates |

The crux is the **causal mask**. Each position can see only the characters before it — the future is hidden (a lower-triangular matrix). Without it the model would cheat by peeking at the answers, and training would be meaningless.

### 3-3. The Training Loop — a Four-Beat Rhythm

Draw a batch → forward pass (predict, compute loss) → backward pass (assign responsibility to each parameter) → optimizer step (nudge the parameters). Repeat the four beats 700 times. The optimizer is AdamW, the standard in real LLM training. The only difference from large-scale training is scale — in reality this loop runs for hundreds of thousands of steps in parallel across thousands of GPUs.

### 3-4. Generation (Decoding) — From Probabilities to Sentences

A trained model only ever emits "a probability distribution over the next character." To make a sentence, sample one character from that distribution, append it to the input, predict again, repeat. Our code uses temperature (sharpening/flattening the distribution) and top-k (keeping only the k best candidates) sampling. The `temperature` parameter in chatbot APIs is exactly this.

### 3-5. What We Compromised in This Exercise

Two things differ from reality. First, instead of Level 01's BPE we use **character-level tokens** (with a corpus this small, character level trains more stably). Second, the corpus is artificial text with strong repeating patterns, so the model learns fast. On real web text, this model would not stand a chance — and that gap is precisely the subject of the next level (scaling laws).

## 4. Hands-On — main.py

How to run:

```bash
cd lecture12_llm_engineering_mlops/level04_mini_gpt_pretraining
python3 main.py
```

The whole run takes about 12 seconds on CPU. Watch for the following in the output.

- **[1]–[2]** The corpus (84,505 characters), the 28-character dictionary, and a model of 107,804 parameters get set up.
- **[3] Before training** — pure random character soup: `'Today  im.sczcTszhlssksvsvrTps tgihTnglnTn mpa...'`. This is the starting point.
- **[4] The training log** — loss plunges from 3.48 (random level) to about 0.20. Look closely at the **mid-run sample at step 30**: `'Today a ay a The modofinixefizeved ma s a Thed a ay ay chis a aded'` — it has learned word fragments but still stitches them clumsily, like a baby babbling. By step 200 it already writes complete sentences: `'Today a student made a pot of stew. Today an office worker fixed m'`.
- **[5] After training** — out come sentences with flawless word order and articles, such as "Today an office worker organized a program. Yesterday a student checked a report." Crucially, many of these are not sentences copied from the corpus but sentences the model **composed** from learned patterns — note step 200's "a student made a pot of stew," a combination the corpus pairs with a chef more often than a student.
- **[6] Checkpoint saved** — weights, character dictionary, and config go to `outputs/tiny_gpt.pt`. Levels 06 (LoRA), 07 (SFT), and 08 (DPO) pick up the story by refining exactly this "pre-trained base model."

The must-read parts of the code are `CausalSelfAttention.forward` (computing attention scores and hiding the future), the single `cross_entropy` line in `TinyGPT.forward` (the entire training objective), and `generate` (the loop that samples and appends one character at a time). All three together are under 30 lines.

## 5. Try It Yourself

1. **Temperature experiment** — at the `sample_text` call sites, change `generate`'s temperature to 0.2 and 1.5. Low repeats only the most common patterns; high gets creative(?) but grammar collapses. (Hint: this is the very knob you turn when a chatbot API's answers are too bland or too scattered.)
2. **Grow the model** — with `D_MODEL=128, N_LAYER=4`, how much lower does the loss go? How many times longer does training take? (Hint: record parameter count and time together. It is a preview of the next level's scaling laws.)
3. **Remove the causal mask** — comment out the `masked_fill` line and train. The loss falls abnormally fast toward 0, but generation is garbage. Why? (Hint: a student who takes the exam while looking at the answer sheet gets a high score and has no skill.)

## 6. Common Mistakes

- **Believing a low loss is always good** — with a mask bug (future-peeking) or heavy data duplication, the loss is low but generation quality is dismal. Always eyeball generated samples.
- **Watching only the train loss** — generalization means val loss falls too. If only train falls, that is memorization (overfitting).
- **Not fixing the seed** — without reproducibility you cannot compare hyperparameter experiments. `torch.manual_seed` is table stakes.
- **Using a big model on a tiny corpus** — when parameters outnumber the data, the model memorizes it wholesale. Model size must be balanced against data volume (next level's subject).
- **Mistaking generated sentences for evidence of 'understanding'** — the model only samples a plausible next character. Keep that in mind and hallucination stops being surprising.

## Next Level Preview

Our model learned in 12 seconds with 108k parameters — so why do real models train for months on thousands of GPUs, and what does that cost? In Level 05 we study the triangle of model size, data volume, and compute (scaling laws), the Chinchilla intuition of "about 20 tokens per parameter," and build an estimator that converts GPU hours into electricity and cloud bills.
