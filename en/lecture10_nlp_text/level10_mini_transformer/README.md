# Lecture 10 · Level 10 — Implementing a Mini Transformer from Scratch

> Build a small decoder (GPT architecture) in torch, train next-token prediction, and watch the generations transform.
**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** level07, level09 / **Estimated time** 70 min

## 1. Why Learn This — The Business View

The experience of "I've built a GPT myself" permanently changes how you see
LLMs. Today's build is a toy with about a hundred thousand parameters, while
commercial LLMs have hundreds of billions — but **the architecture is the
same**: embeddings → stacked decoder blocks → next-token probabilities.
Only the scale differs.

Build it once and these things become self-evident: why an LLM generates one
token at a time (and hence why responses stream), why training needs GPUs, why
"hallucination" is not a bug but the nature of probabilistic generation, and
what fine-tuning actually changes. You become the person in the meeting who
knows precisely what "an LLM is just a next-word guessing machine" does and
does not mean.

## 2. Grasping It Through an Analogy

If everything up to level09 was **studying car parts**, today is **assembly,
ignition, and a road test**. The parts (attention, FF, residuals, layer norm)
are already built. Only three things are new today:

1. **Assemble the parts in torch**: unlike the numpy version, autograd works,
   so training becomes possible.
2. **Driving practice (training)**: show it tiny_corpus and make it play
   "guess the next token" thousands of times. Every mistake back-propagates a
   small adjustment through all the parts.
3. **The road test (generation)**: generate from the same prompt before,
   during, and after training and watch gibberish turn into plausible
   sentences.

The goal of this driving practice is simple — lap the course (the corpus) over
and over until "in this situation, do this" is second nature. It is
LLM pre-training in miniature.

## 3. Core Concepts

### 3-1. GPT architecture = a transformer of stacked decoders only

The original transformer was for translation, so it had encoder + decoder;
GPT keeps **only the decoder**.

```
token IDs → token embedding + position embedding
         → [decoder block: causal-mask attention → FF (+residual/norm)] × N layers
         → final layer norm → linear layer → scores over the vocabulary (logits)
```

The last linear layer outputs "a score for each vocabulary candidate as the
next token", and softmax turns it into probabilities. Today's model is 2
layers, 64 dims, 4 heads — about a hundred thousand parameters, ultra-small.

### 3-2. The magic of parallel training — dozens of problems per sentence

An RNN had to process characters in order, but thanks to the causal mask a
transformer can train on **every position of a sequence at once**. One
length-32 sequence is "see up to 1, guess 2", "see up to 2, guess 3"... —
32 problems, all solved in one forward pass. This parallelism is one of the
practical reasons transformers displaced RNNs: it fills a GPU to the brim.

### 3-3. Falling loss = shrinking surprise

Cross-entropy loss grows the lower the probability given to the correct token.
Early in training, with a vocabulary of $V$ types, the loss starts around
$\ln V$ (blind guessing). A falling loss means the model is less surprised by
each next token. The language-model metric perplexity is $e^{loss}$, readable
as "between how many candidates is it hesitating at each moment". Both numbers
print during the run.

### 3-4. Stages of generation quality

As a small model trains, generations climb roughly these stairs:

1. Random character salad (before training)
2. Mimicking character statistics — common letters and short patterns appear
3. Word skeletons — the "X did Y." template locks in
4. Plausible sentences — recombinations of corpus patterns

A commercial LLM pours in tens of thousands of times more data, model, and
compute to reach stage 5 (knowledge and reasoning). Scale creates the stages —
which is exactly the topic of the next level (the pre-training paradigm).

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

Trains a character-level mini GPT on 30,000 characters of tiny_corpus.
Designed to finish within 60 seconds on CPU (about 6 seconds on the author's
machine).

- **[1] Data**: the character dictionary (28 types) and the (input,
  one-char-shifted target) batch construction.
- **[2] Model**: a 64-dim, 2-layer, 4-head decoder — 105,756 parameters. The
  attention and block are torch versions of level09's parts, so the code will
  look familiar.
- **[3] Training**: 400 steps. Actual run: loss starts near ln(28)=3.33; at
  step 50 loss 1.292 (perplexity 3.6) with word-like fragments
  (`'Yesterday r. ster anixer teled ache...'`); by step 150 loss 0.268
  (perplexity 1.3) and the model writes
  `'Yesterday a student prepared a pot of stew. This m'`; step 400 lands at
  loss 0.226. The loss curve saves to `outputs/loss_curve.png`.
- **[4] Generation test**: three prompts, actual outputs —
  `'Yesterday a student '` → `'...checked a program. This morning a teacher checked '`,
  `'Over the weekend a chef '` → `'...prepared a proposal. This morning a developer prep'`.
- **[5] Peeking at the attention**: the first layer's head-0 weights print as a
  clean lower-triangular matrix — the causal (no-future) structure holds.

## 5. Try It Yourself

1. **(easy)** Cut `STEPS` to 100 and observe at which stage (section 3-4) the
   generations stall.
2. **(medium)** Compare the final loss with `N_LAYERS` reduced to 1 and raised
   to 3. At this data size, is adding layers always a win?
3. **(challenge)** Experiment with "accidentally" removing the causal mask
   (comment out the mask line in `CausalSelfAttention`). The loss drops
   abnormally fast toward 0, but generation is garbage — an exam score earned
   by peeking at the future is not ability.

## 6. Common Mistakes

- **Disappointment at a small model's prose**: today's goal is not ChatGPT but
  observing "structure being learned". If the sentence template locks in,
  that's a triumph.
- **Aiming for loss 0**: language has inherent uncertainty; the loss cannot and
  should not reach 0 (perfect memorization = overfitting).
- **Confusing context length (BLOCK) with generation length**: the model sees
  only the latest BLOCK characters. Generate longer than that and forgetting
  the beginning is normal.
- **Mistaking a mask bug for a performance gain**: as in challenge 3, when the
  future is visible the loss looks great but generation ability is zero. A good
  metric is not automatically good news.

## Next Level Preview

We just trained "continue the text" — the GPT way. In level11 we build the
other camp, "fill in the blank" (the BERT way), compare them, and complete the
big picture of why the pre-train → fine-tune paradigm changed the entire AI
industry.
