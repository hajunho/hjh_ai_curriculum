# Lecture 11 · Level 00 — What Is an LLM? A Giant 'Next-Word Predictor'

> Experience first-hand, with a mini model, that what large language models (LLMs) like ChatGPT and Claude actually do is "guess the next word."
**Difficulty** ⭐ / **Prerequisites** lecture02 (Python basics) / **Estimated time** 30 min

## 1. Why Learn This — The Business View

These days the question "Can't AI do this for us?" comes up several times a day at work. But if you don't know what an LLM (Large Language Model) is good and bad at, you will make one of two mistakes. One is overestimating it — pasting its output into a report unchecked because "the AI will get it right." The other is underestimating it — writing it off as "just a chatbot" and missing real automation opportunities.

Once you understand the principle of an LLM in a single sentence, every later judgment becomes easier. **An LLM is a machine that predicts, as probabilities, "the word that comes next after the text so far."** From this one sentence follow all of its strengths (fluent writing, summarization, translation) and all of its weaknesses (no fact-checking, no up-to-date knowledge, hallucination). The person who knows how the tool works is the one who controls the tool.

## 2. Grasping It Through an Analogy

Picture **a supersized version of your phone's autocomplete**. Type "How are" and the keyboard suggests "you." Now imagine that this autocomplete (1) has read the equivalent of millions of books rather than a few words, (2) remembers tens of thousands of words of context instead of the last word or two, and (3) suggests whole paragraphs rather than a single word — that is an LLM.

**Parameters are experience points.** Just as a game character with more XP can clear harder quests, a model with more parameters (the numbers it tuned during training) remembers subtler language patterns. A new hire (a small model) can only handle what's in the manual, while a 20-year veteran (a large model) handles situations they have never seen by analogy to "something similar I've been through."

Even the veteran has weaknesses, though. **An employee who left the company knows nothing about what happened after they left.** Likewise, an LLM knows nothing after the point its training ended (the knowledge cutoff). And when its memory is fuzzy, it confidently improvises — "it was probably something like that" — which is exactly what we call hallucination.

## 3. Core Concepts

### 3.1 Token-by-token generation

An LLM does not produce a sentence in one piece. It predicts one token (a word fragment), appends it to the input, then predicts the next token again. A 100-word answer means this loop ran hundreds of times. This is why API pricing is "per token" and why long answers are slow.

### 3.2 Probability distributions and sampling

At every step the model builds a probability table over "every token that could come next." For example, after "Yesterday a developer" the distribution might read "fixed 41%, checked 22%, ...". If you always pick the #1 entry (greedy selection), the answer is always the same and monotone. If you draw according to the probabilities (sampling), you get varied answers that differ slightly every time. That is why ChatGPT gives a different answer to the same question.

### 3.3 The n-gram language model — today's lab material

Before neural networks, language models were plain statistics tables. An n-gram model is a table that counts "when the previous (n-1) words were these, what came next."

| Context (previous 2 words) | Next-word candidate | Count |
|---|---|---|
| a developer | fixed | 87 |
| a developer | checked | 45 |

A real LLM replaces this "table" with a neural network so it can also extrapolate to contexts never seen in the table. The principle is identical: **predict the next word from the statistics of past text.**

### 3.4 Two limitations built into its nature

1. **Knowledge cutoff**: what is not in the training data does not exist for the model. Your company's policies, yesterday's news, your customer database — it knows none of it. (The fix is RAG, later in this lecture.)
2. **Hallucination**: the model learned to "speak fluently," not to "speak only the truth." Any statistically plausible combination gets said with full confidence, factual or not.

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level00_what_is_llm
python3 main.py
```

Reading the output:

- **[1]–[2]** We build bigram/trigram statistics tables from the mini English corpus in `hjh_data.tiny_corpus()` (about 84,500 characters, 14,816 words). This "table building" *is* the training — 28 bigram contexts and 77 trigram contexts.
- **[3]** A bar chart of the probability distribution of the word after "worker" (prepared 23.0%, organized 20.6%, made 19.7%, ...). This is exactly what an LLM computes internally at every token.
- **[4]** From the same starting context ("Today a") sampling produces 3 different sentences, while greedy selection always produces the same one ("Today a student checked a report.").
- **[5]** Among 20 generated sentences, 1 is a **new combination that never appeared** in the training data ("This evening a developer made a letter."). The grammar is fine but nothing guarantees it is true — a miniature of hallucination.
- **[6]** Give the model a word missing from its corpus ("manager") as context and there is no distribution at all. A miniature of the knowledge cutoff.

The heart of the code is a 3-stage structure: `build_ngram_model` (context → next-word counts), `next_word_distribution` (counts → probabilities), `generate` (probabilities → draw → repeat). An LLM's skeleton is the same; only the count table has been swapped for a neural network.

## 5. Try It Yourself

1. **(Easy)** Extend generation with `generate(..., max_words=30)` and make longer text. Where does it start to fall apart? (Hint: a trigram only remembers the previous 2 words — the limitation of a model with a small context window.)
2. **(Medium)** Change the starting context in `[4]` to `("This", "morning")` and rerun. Also generate with the bigram model (`n=2`) and compare sentence quality with the trigram. (Hint: match the context length to 1, e.g. `generate(bigram, ("teacher",), rng)`.)
3. **(Challenge)** Build a 4-gram model. Generation quality improves — but what happens to the number of distinct contexts stored in `model`? Verify with numbers the trade-off that remembering longer contexts makes the required data and parameters explode.

## 6. Common Mistakes

- **Believing "the LLM searches the web for its answer"** — a base LLM searches nothing. It generates from statistics it saw during training. Bolting on search is a separate technique (RAG, level08).
- **"It sounded confident, so it must be true"** — it only means the top-probability token was picked, not that it is factual. Fluency ≠ accuracy.
- **Thinking it's broken because the same question gets different answers** — that is sampling working as designed. If you need reproducibility, lower the temperature (closer to greedy selection).
- **Experimenting without fixing a seed** — today's code pins `random.Random(42)`. Always fix the seed when comparing experiments.

## Next Level Preview

Now that you know the principle, it's time to answer "so what can we make it do at the office?" In Level 01 we draw a map of LLM use cases at work — summarization, drafts, classification, extraction — and automate an email draft, meeting-minutes summary, and complaint triage with a mock LLM.
