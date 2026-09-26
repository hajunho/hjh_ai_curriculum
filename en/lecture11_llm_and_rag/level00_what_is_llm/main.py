"""
Experience the principle behind LLMs (large language models) as a
'next-word predictor'.
1) Count n-gram statistics (which word follows which) on a mini English corpus.
2) Inspect the probability distribution of the word that follows a context.
3) 'Generate' sentences by sampling one word at a time from those probabilities.
A real LLM scales this exact principle up to a neural network with
hundreds of billions of parameters.
"""

import pathlib
import random
import sys
from collections import Counter, defaultdict

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def build_ngram_model(words: list[str], n: int) -> dict:
    """Build a table: previous (n-1) words (context) -> next-word counts."""
    model = defaultdict(Counter)
    for i in range(len(words) - n + 1):
        context = tuple(words[i:i + n - 1])
        nxt = words[i + n - 1]
        model[context][nxt] += 1
    return model


def next_word_distribution(model: dict, context: tuple) -> list[tuple[str, float]]:
    """Compute the probability distribution over words that follow a context."""
    counter = model.get(context, Counter())
    total = sum(counter.values())
    if total == 0:
        return []
    return [(w, c / total) for w, c in counter.most_common()]


def generate(model: dict, start: tuple, rng: random.Random,
             max_words: int = 12, greedy: bool = False) -> str:
    """Start from a context and repeatedly draw the next word to build a sentence.
    greedy=True always picks the top word (temperature 0); False samples by probability."""
    out = list(start)
    context = start
    for _ in range(max_words):
        dist = next_word_distribution(model, context)
        if not dist:
            break
        if greedy:
            word = dist[0][0]                      # the most probable word
        else:
            words, probs = zip(*dist)
            word = rng.choices(words, weights=probs, k=1)[0]
        out.append(word)
        if word.endswith("."):                     # a period ends the sentence
            break
        context = tuple(out[-(len(start)):])       # slide the context window
    return " ".join(out)


def main() -> None:
    rng = random.Random(42)                        # fixed seed for reproducibility

    print("=" * 62)
    print("Level 00 | What is an LLM — a giant 'next-word predictor'")
    print("=" * 62)

    # [1] Training data: a mini English corpus
    corpus = hjh_data.tiny_corpus()
    words = corpus.split()
    print(f"\n[1] Training corpus: {len(corpus):,} characters, {len(words):,} words")
    print(f"    Opening sample: {' '.join(words[:12])} ...")

    # [2] Building the statistics = 'training'. More parameters -> longer memory.
    bigram = build_ngram_model(words, n=2)
    trigram = build_ngram_model(words, n=3)
    print(f"\n[2] Training done — {len(bigram):,} bigram contexts, {len(trigram):,} trigram contexts")
    print("    (a real LLM stores this not as a 'table' but as a neural net")
    print("     with hundreds of billions of parameters)")

    # [3] Next-word probability distribution — what an LLM does at every token
    print("\n[3] Probability distribution of the word after 'worker' (top 5)")
    for w, p in next_word_distribution(bigram, ("worker",))[:5]:
        bar = "#" * int(p * 40)
        print(f"    {w:<10} {p:6.1%} {bar}")

    print("\n    Word after 'a developer' (trigram, top 5)")
    for w, p in next_word_distribution(trigram, ("a", "developer"))[:5]:
        bar = "#" * int(p * 40)
        print(f"    {w:<12} {p:6.1%} {bar}")

    # [4] Generation — build a sentence one word at a time (token-by-token)
    print("\n[4] Sentence generation: same start, different sentence every time with sampling")
    for i in range(3):
        print(f"    Sample {i + 1}: {generate(trigram, ('Today', 'a'), rng)}")
    print(f"    Greedy (always the top word): {generate(trigram, ('Today', 'a'), rng, greedy=True)}")
    print("    -> This 'sampling' is why ChatGPT/Claude answer the same question differently.")

    # [5] Experiencing the limits — plausible combos never seen in the corpus
    #     = the seed of hallucination
    print("\n[5] Limitation: the mechanics of plausible falsehoods (hallucination)")
    sentences = set(s.strip() for s in corpus.split(".") if s.strip())
    novel, sample_novel = 0, ""
    for i in range(20):
        g = generate(trigram, ("This", "evening"), rng).rstrip(".")
        if g not in sentences:
            novel += 1
            sample_novel = g
    print(f"    New combinations NOT found in the training data: {novel} of 20")
    print("    generated sentences.")
    print(f"    Example: \"{sample_novel}.\"")
    print("    The grammar is fine, but nothing was fact-checked — LLM hallucination")
    print("    works on the very same principle.")

    # [6] Knowledge cutoff
    print("\n[6] Knowledge cutoff: this model knows nothing about words missing from its corpus.")
    print(f"    Distribution after 'manager': "
          f"{next_word_distribution(bigram, ('manager',)) or 'none (never seen in training)'}")
    print("    -> A real LLM likewise knows nothing after its training ended. (Hence RAG.)")

    print("\nSummary: an LLM = a giant autocomplete that learned 'next-token probabilities'")
    print("         from massive text. Fluency and factuality are separate things —")
    print("         that insight is the starting point of this whole lecture.")


if __name__ == "__main__":
    main()
