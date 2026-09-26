# Lecture 10 · Level 09 — Dissecting the Transformer Architecture

> Build multi-head attention, positional encoding, residual connections, and layer norm in numpy, and assemble one full block.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level08 / **Estimated time** 60 min

## 1. Why Learn This — The Business View

GPT, Claude, Gemini, translators, code assistants — different names, same
insides: the Transformer. Proposed in 2017, this one architecture has been AI's
standard blueprint for a decade, and the spec races in the news — "parameter
count", "layer count", "context length" — are all talk about the number of
parts in this blueprint.

Know the parts and the conversation changes. In decision meetings like "is a 7B
model enough for our service?" or "is a 128k-context model worth switching to?",
the person who knows transformer anatomy can reason how spec numbers connect to
cost, speed, and quality. This level takes the blueprint apart, part by part,
and builds each one. No training — we focus purely on structure and data flow
(shapes).

## 2. Grasping It Through an Analogy

A transformer block is **a task-force team that runs on cycles of meeting →
write-up**. Each token is a team member holding their own piece of information
(a vector).

1. **The all-hands (multi-head attention)**: members consult each other's
   information and update their own thinking. And the meeting isn't held once —
   it splits into **parallel working groups**: the budget group, the schedule
   group, the risk group each discuss from their own angle, then merge results.
   That is multi-head.
2. **Private digestion (feed-forward)**: after the meeting, everyone returns to
   their desk and digests what they heard in their own way. No consulting the
   neighbor — and every member uses the same digestion technique (shared weights).
3. **Preserving the original (residual connection)**: if only the meeting notes
   survive, your original opinion may vanish. So you carry "my original opinion
   + what I gained in the meeting", added together.
4. **Condition management (layer norm)**: meeting after meeting, some members
   overheat and some fall silent. After every step, speaking volume is
   standardized back into range.

One "meeting + write-up" set is a block; a transformer is this block stacked
dozens of times.

## 3. Core Concepts

### 3-1. Positional encoding — restoring the sense of order

Attention looks at all tokens simultaneously, so it **doesn't know order**.
"The company acquired the customer" and "the customer acquired the company" are
the same set. So we add a **seat-number tag** to the input embedding. The
original transformer uses sine/cosine waves:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(\cdot)$$

Overlaying waves of different frequencies gives every position a unique
pattern, and relative distances between positions become expressible by vector
arithmetic. It is the same principle as binary counting, where each digit flips
at a different period. (Newer models use variants like RoPE, but the goal —
injecting position information — is identical.)

### 3-2. Multi-head attention — dividing the perspectives

A single attention can capture only one kind of relationship. Split the
embedding across heads (e.g. 32 dims → 8 dims × 4 heads), attend independently
in each slice, then concatenate — and one head can latch onto "modifier
relations" while another tracks "subject-verb relations". Compute stays about
the same as one big attention while expressiveness grows.

### 3-3. Residual connections and layer norm — the safety gear for going deep

- **Residual connection**: output = input + part(input). Even while a part is
  undertrained, the input passes through a shortcut, so the training signal
  reaches the bottom layers of a very deep stack. Same idea you saw with CNNs
  in lecture09.
- **Layer normalization**: standardize each token vector to mean 0, variance 1,
  then apply a learnable scale/shift. It stops values from exploding or dying
  as they cross layers.

### 3-4. Encoder vs decoder

- **Encoder**: sees the whole sentence **bidirectionally**. Purpose:
  understanding — classification, retrieval, fill-in-the-blank (BERT family).
- **Decoder**: a causal mask lets each token see **only what came before it**.
  Purpose: generating the next token — the GPT family. Looking ahead would be
  cheating off the answer sheet.

Same parts; one mask decides the destiny. In level10 we build and train a
decoder; in level11 we compare the two camps' philosophies.

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

One sentence (6 tokens, 32 dims) passes through a transformer block while the
program prints **every part's input/output shape and role**. No training —
seed-fixed random weights. Today's star is the structure, not the numbers.

- **[1] Positional encoding**: builds the sine/cosine table and shows
  numerically that "the same word in a different seat becomes a different
  input" (difference norm 2.73 between positions 0 and 3).
- **[2] Multi-head attention**: splits 32 dims into 4 heads × 8 dims, prints
  each head's weight-matrix shape (4, 6, 6), and shows the heads attending to
  different places — in this run the 'student' token's strongest attention is
  head0='a' but head2='report'.
- **[3] Residual + layer norm**: prints per-token standard deviations before
  (1.27, 1.18, 1.47...) and after normalization (all exactly 1).
- **[4] Feed-forward**: the shape flow of expand (32→64) → ReLU → contract
  (64→32).
- **[5] Block assembly & stacking**: chains the parts into a block, passes the
  input through two blocks ((6,32) in, (6,32) out each time), and counts 8,192
  parameters per block.
- **[6] Encoder vs decoder**: the same input with the causal mask off and on,
  printed side by side — the decoder's weight matrix is visibly
  lower-triangular, upper-right all zeros.

## 5. Try It Yourself

1. **(easy)** Change the head count from 4 to 2, then 8. Find in [2]'s shape
   output why the total parameter count doesn't change.
2. **(medium)** Comment out the positional-encoding addition in [1] and confirm
   that shuffling token order leaves the set of attention weights identical.
3. **(challenge)** Remove the layer norms inside the block and pass the input
   through the block 10 times, printing the output's standard deviation each
   pass. Watching the values drift up or down is understanding, in your own
   hands, why deep models need normalization.

## 6. Common Mistakes

- **Reading the code without watching shapes**: the royal road to understanding
  transformers is tracking tensor shapes at every step. Sketch how
  (token count, dims) changes and where.
- **Thinking multi-head = attention repeated several times**: it's not
  repetition, it's splitting the dimensions and looking in parallel. That's why
  compute doesn't multiply by the head count.
- **Treating positional encoding as decoration**: without it the transformer
  knows nothing of word order — it degenerates into a bag-of-words model.
- **Memorizing encoder/decoder as separate technologies**: same parts, one mask
  of difference. Remember it as "what is each token allowed to see", not as a
  different structure.

## Next Level Preview

We've read the whole blueprint; now we start the factory. In level10 we port
today's parts to torch, build a small decoder, and actually train next-token
prediction on tiny_corpus, watching the loss fall and the generations change.
