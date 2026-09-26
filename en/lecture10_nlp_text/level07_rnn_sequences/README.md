# Lecture 10 · Level 07 — RNNs and Sequence Models

> Train a mini next-character language model with a neural network that reads while remembering order (an RNN).
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level06, lecture08 (deep learning basics) / **Estimated time** 55 min

## 1. Why Learn This — The Business View

Every tool so far (BoW, TF-IDF, embeddings) **dumped words into a bag** and
counted. But half of language is order. "The company acquired the customer" and
"the customer acquired the company" contain the same words with opposite
meanings. Without order there is no contract interpretation, no text
generation, no translation.

The RNN (Recurrent Neural Network) is the first serious neural network for
order. It powered the early golden age of autocomplete, speech recognition, and
machine translation, and though it has since ceded the throne to the
transformer, you need its core idea — "carry a running summary of what you've
read so far as state" — and its limitation (long-term dependency) to understand
**what exactly the transformer fixed**. This level is that stepping stone.

## 2. Grasping It Through an Analogy

An RNN is **a line of employees reading a document with a relay memo**. The
document is so long that each employee reads only one word. Employee #1 reads
the first word and writes a running summary on **a single memo sheet**, handing
it to #2. #2 reads the memo plus the second word, updates the memo, hands it to
#3... The last employee's memo is the summary of the whole document.

The strength: however long the document, each employee only ever handles
"one memo + one word". The weakness is also the memo. The sheet has a fixed
size (fixed-size hidden state), so content from the document's beginning blurs
with every hand-off. How much of employee #1's note survives in employee #100's
memo? That is the **long-term dependency problem**.

## 3. Core Concepts

### 3-1. The recurrence — the formula is one line

The body of an RNN is startlingly simple. At timestep $t$:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

$x_t$ is the token being read (its embedding), $h_{t-1}$ is the memo so far
(the hidden state), $h_t$ is the updated memo. The key is that the same weights
$W$ are reused at every timestep — even with 100 employees, there is only one
"memo-updating procedure".

### 3-2. Language models — the game of guessing what comes next

A language model predicts the next token given the text so far: knowing that
after "For breakfast I had scrambled ___" the word 'eggs' is likely. This
simple game matters because playing it well requires grammar, vocabulary, and
common sense all at once. The G in GPT (Generative) means champion of exactly
this game — and today we build a micro, character-level version.

Training: given "Yesterday a", the input "Yesterday " must predict the target
"esterday a" (shifted by one character). Each timestep becomes a classification
over the next character, trained with the cross-entropy loss from lecture08.

### 3-3. Long-term dependency — the relay memo's limit

To fill the blank in "the Swiss ___ I bought yesterday at the department store,
in the blue box", you may need information from the very start of the sentence.
During backpropagation an RNN multiplies gradients across timesteps until they
**vanish**, so the learning signal barely reaches the distant past. LSTM/GRU
(gates that selectively preserve the memo) soften this; the radical fix is
attention (level08), which connects every timestep to every timestep directly.

### 3-4. Generation — the dial called temperature

To generate text, sample the next character from the predicted distribution,
append it to the input, repeat. The dial that controls how "peaked" the
distribution is used is temperature. Low = pick only the most probable
characters, safe and monotonous; high = varied but erratic. The temperature
parameter in LLM APIs is exactly this.

## 4. Hands-On — main.py

Run:

```bash
python3 main.py
```

Trains a character-level RNN language model on part of tiny_corpus (20,000
characters). Small enough to finish in about a second on CPU.

- **[1] Data prep**: builds the character dictionary (28 types) and converts
  text to integer sequences — 499 training sequences of length 40. Prints the
  "target = input shifted by one" pair construction.
- **[2] Model assembly**: embedding → `nn.RNN` → output layer, 8,988 parameters
  (a billionth-ish of today's LLMs).
- **[3] Training**: compares generations before and after. Actual run: before
  training the model babbles (`'Yesterday a student wrYehuotozkoTefTe.xlnfm...'`);
  by epoch 1 (loss 3.055) it produces letter soup with spaces; epoch 6
  (loss 1.242) shows word-like shapes; epoch 12 (loss 0.390) yields
  `'Yesterday a student stew. Todand a student made a proposal. '` — the
  sentence skeleton has clearly taken shape.
- **[4] Next-character check**: after `'a student checked a re'` the top
  prediction is `'p'` at 87% (heading for "report"); after `'a pot of st'`,
  `'e'` at 95% (stew).
- **[5] Temperature experiment**: temperature 0.3 gives the clean
  `'Over the weekend a teacher made a proposal. This morning a chef che'`,
  while 1.5 derails into `'Over the weekend a devefxud vef anstaod a program...'`.
  Ends with a memo on the relay-memo's long-term dependency limit.

## 5. Try It Yourself

1. **(easy)** Change the temperatures in [5] to 0.1 and 2.0 and compare the
   generations.
2. **(medium)** Grow the hidden state (`HIDDEN`) from 64 to 128 and watch the
   loss, generation quality, and training time.
3. **(challenge)** Swap `nn.RNN` for `nn.GRU` (a one-line edit). Does the loss
   land lower at the same epoch? This experiment shows how much the gates
   reduce the relay memo's forgetting.

## 6. Common Mistakes

- **Mismatched character dictionaries**: generation must use the exact
  char→id dictionary from training. Rebuild the dictionary and the ids shuffle —
  the output turns to gibberish.
- **Blaming the model the moment generation looks odd**: check temperature,
  prompt, and training volume first. With this little data, don't expect perfect
  sentences — the experiment is about "is structure emerging?".
- **Forgetting to reset the hidden state**: training across unrelated sentences
  without cutting the state lets the previous sentence's memory leak into the
  next.
- **Deploying an RNN in production today**: as of 2026 the default for sequence
  work is the transformer. We learn the RNN to understand *why* transformers.

## Next Level Preview

The relay memo's limit was "no direct view of the distant past". Level08's
attention flips the premise — what if every timestep could look **directly** at
every other timestep? We implement attention in numpy and draw the weight
heatmap.
