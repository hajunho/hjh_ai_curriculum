# Lecture 10 · Level 08 — The Attention Mechanism

> Implement attention — the computation of "where to look, and how hard" — in numpy, and see the weights as a heatmap.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level06, level07 / **Estimated time** 55 min

## 1. Why Learn This — The Business View

Every AI moving the world right now — chatbots, translators, code generators,
even image generators — has the same part at its heart: attention. Exactly as
the 2017 paper title said, "Attention Is All You Need": this one part replaced
the RNN's relay memo and opened the transformer era.

Two practical reasons a non-engineer should know attention. First, **it
explains how LLMs behave and where they hit limits**: why context length is
capped (attention cost grows with the square of length), why prompt advice says
to place key information at the beginning or end — all properties of attention.
Second, **explainability**: opening the attention weights gives visual clues
about "which words the model looked at when deciding".

## 2. Grasping It Through an Analogy

Attention is **distributing your ear across speakers in a meeting**. When you
write the wrap-up memo of a meeting with 10 speakers, you don't weight all 10
equally. You open your ears wide for remarks relevant to the sentence you're
writing (large weight) and let unrelated remarks wash by (small weight). The
final memo is "a weighted average of the remarks".

To mechanize this, three things are needed. Staying with the meeting:

- **Query**: what I want to know right now. "Who said what about the budget?"
- **Key**: each speaker's name tag — "I handle budgets", "I handle PR".
  Matching query against name tags scores the relevance.
- **Value**: what each speaker actually said. You take it, mixed in proportion
  to relevance.

Match the query (Q) against all keys (K) to score attention, then take the
weighted average of values (V) in that proportion — that's the whole of
attention. Unlike an RNN, **any remark is directly accessible whenever it was
made**, so nothing blurs across relay hand-offs.

## 3. Core Concepts

### 3-1. Scaled dot-product attention — dissecting the one-line formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Piece by piece:

1. $QK^\top$ — dot products of the query with every key. A dot product grows
   when directions align, so this is the relevance score table. (The similarity
   from level06 reappears here.)
2. $\div \sqrt{d_k}$ — with large vector dimensions the dot products get big
   and softmax saturates to one side. Dividing by the square root of the
   dimension keeps the scores in scale — hence the name 'scaled'.
3. $\text{softmax}$ — turns scores into probabilities summing to 1 (the
   attention budget): "budget owner 60%, planning 30%, everyone else 10%".
4. $\times V$ — the weighted average of value vectors by attention. That is
   the output.

### 3-2. Where do Q, K, V come from?

In self-attention, all three come from the same sentence's token embeddings
$X$, multiplied by different matrices $W_Q, W_K, W_V$. The same person plays
both questioner and answerer in the meeting. These matrices are **learned
parameters**, so during training the model teaches itself "what to ask, and
what to print on the name tags". Today's exercise looks at structure only, no
training, so we use seed-fixed random matrices.

### 3-3. Reading the attention weight matrix

With $n$ tokens the weights form an $n \times n$ matrix. Cell $(i, j)$ is
"how much token $i$ consulted token $j$". Every row sums to 1 (softmax).
Drawn as a heatmap, you see the model's 'eye movements'. In "bank of the river
flooded", if the 'bank' row is bright at the 'river' and 'flooded' columns,
the representation of 'bank' was rebuilt from those words — level00's
ambiguity problem handled, in one picture.

### 3-4. The cost — why long context is expensive

Every token looks at every token, so compute and memory grow as $n^2$.
1,000 tokens → a million cells; 10,000 → a hundred million. This is the root
cause of LLM context-length limits and of "long inputs cost more".

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

- **[1] Attention in 5 lines**: softmax and scaled dot-product attention in
  numpy. The core function is 5 lines.
- **[2] Toy-example numbers**: a 3-token micro example prints every
  intermediate — score matrix → scaling → softmax → weighted average — and
  confirms each row sums to 1.
- **[3] Self-attention over the 'bank' sentences**: with mini handmade
  embeddings (level06 style), self-attention is computed for
  "bank approved my loan" vs "bank of the river flooded". Actual output: in the
  loan sentence 'bank' attends bank(0.76), approved(0.12), loan(0.09); in the
  river sentence bank(0.16), river(0.52), flooded(0.30) — the same word, pulled
  toward different context.
- **[4] Heatmap PNG**: both sentences' attention weights save to
  `outputs/attention_heatmap.png`. Rows = attending token, columns = attended.
- **[5] Feeling the cost**: weight-matrix sizes for 10/100/1,000/10,000 tokens
  (100 cells up to 100,000,000 cells, ~400 MB per layer per head) make the
  $n^2$ cost visceral.

## 5. Try It Yourself

1. **(easy)** Replace the sentences in [3] with your own and redraw the heatmap.
   (Combine words that exist in the embedding dictionary.)
2. **(medium)** Remove the scaling (the √d division) and compare how lopsided
   the softmax results in [2] become.
3. **(challenge)** Add a mask: fill the upper-right triangle of the score
   matrix with −1e9 so each token cannot see tokens after itself, then softmax —
   the attention matrix becomes lower-triangular.
   Hint: `np.triu(np.ones((n,n)), k=1)`. This is the causal mask GPT uses,
   deployed for real in level10.

## 6. Common Mistakes

- **Believing attention weights are a full explanation**: the weights are a
  useful clue, not the whole of the model's reasoning. With many layers and
  heads intertwined, interpretation blurs.
- **Confusing the softmax direction**: the weight matrix sums to 1 along rows,
  not columns. When reading a heatmap, remember "the row is the one doing the
  looking".
- **Skipping the scaling**: at high dimension the softmax saturates and
  gradients die. The √d division is not decoration.
- **Mystifying Q, K, V**: each is just "input X times one matrix". Different
  matrices are used only so the roles can differ.

## Next Level Preview

We have the heart — attention; now to assemble the body. In level09 we build
multi-head attention, positional encoding, residual connections, and layer
normalization one by one in numpy, completing a full transformer block.
