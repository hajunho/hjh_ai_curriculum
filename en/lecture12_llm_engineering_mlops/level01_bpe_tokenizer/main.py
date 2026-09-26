"""
We implement a BPE (Byte Pair Encoding) tokenizer from scratch.
1) Build a vocab by repeatedly merging the "most frequent adjacent character
   pair" in tiny_corpus,
2) compare how many tokens a sentence becomes at different merge counts
   (vocab sizes), and
3) see how digit tokenization changes with and without digit splitting —
   and how that affects math ability.
"""

import sys
import pathlib
import random
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

END = "</w>"  # end-of-word marker — distinguishes "port" in "report" from "port" alone.


def word_freqs(text: str) -> Counter:
    """Count whitespace-separated words into a {character tuple: frequency} dict."""
    freqs = Counter()
    for word in text.split():
        freqs[tuple(list(word) + [END])] += 1
    return freqs


def count_pairs(freqs: Counter, digit_split: bool) -> Counter:
    """Count occurrences of adjacent symbol pairs. With digit_split=True,
    symbols containing digits are excluded from merge candidates, so digits
    always stay one character each."""
    pairs = Counter()
    for symbols, f in freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            if digit_split and (any(c.isdigit() for c in a) or
                                any(c.isdigit() for c in b)):
                continue
            pairs[(a, b)] += f
    return pairs


def merge_pair(freqs: Counter, pair) -> Counter:
    """Glue the given symbol pair into one symbol in every word."""
    a, b = pair
    new_freqs = Counter()
    for symbols, f in freqs.items():
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        new_freqs[tuple(merged)] += f
    return new_freqs


def train_bpe(text: str, num_merges: int, digit_split: bool = False,
              verbose: bool = False):
    """BPE training: merge the most frequent pair num_merges times and return the merge rules."""
    freqs = word_freqs(text)
    merges = []
    for step in range(num_merges):
        pairs = count_pairs(freqs, digit_split)
        if not pairs:            # stop early when no pairs are left to merge
            break
        best, best_n = pairs.most_common(1)[0]
        freqs = merge_pair(freqs, best)
        merges.append(best)
        if verbose and (step < 8 or (step + 1) % 20 == 0):
            print(f"    merge {step + 1:3d}: '{best[0]}' + '{best[1]}'"
                  f" -> '{best[0] + best[1]}'  ({best_n} occurrences)")
    return merges


def encode(text: str, merges) -> list:
    """Tokenize a new sentence by applying the learned merge rules in order."""
    tokens = []
    for word in text.split():
        symbols = list(word) + [END]
        for pair in merges:  # apply in the order learned (priority)
            symbols = list(next(iter(
                merge_pair(Counter({tuple(symbols): 1}), pair))))
        tokens.extend(symbols)
    return [t.replace(END, "") for t in tokens if t != END]


def make_price_corpus(n: int = 300, seed: int = 42) -> str:
    """A corpus of sales-report sentences for the digit-split experiment
    (amounts with lots of trailing zeros)."""
    rng = random.Random(seed)
    lines = [f"revenue {rng.randrange(1, 999) * 100} KRW logged"
             for _ in range(n)]
    return " ".join(lines)


if __name__ == "__main__":
    corpus = hjh_data.tiny_corpus()
    print(f"[1] Corpus loaded: {len(corpus):,} characters, "
          f"{len(set(corpus)):,} unique characters (we start at character level)")

    # ---- watch the BPE merge process ----
    print("\n[2] BPE training — merging the most frequent adjacent pairs, one by one.")
    merges = train_bpe(corpus, num_merges=60, verbose=True)
    print(f"    {len(merges)} merges completed (60 requested; stops early if pairs run out)")

    # ---- vocab size trade-off ----
    sample = "Yesterday a student made a report."
    print(f"\n[3] The vocab-size trade-off — how many tokens does the same sentence become?")
    print(f"    Sentence: \"{sample}\"")
    base_vocab = len(set(corpus)) + 1  # unique characters + </w>
    for n in [0, 10, 30, 60]:
        m = train_bpe(corpus, num_merges=n)
        toks = encode(sample, m)
        print(f"    {n:3d} merges (vocab~{base_vocab + len(m):3d}):"
              f" {len(toks):2d} tokens -> {toks}")
    print("    -> A bigger vocab makes sentences shorter (cheaper inference),")
    print("       but the embedding table grows and rare tokens get fewer chances to learn.")

    # ---- digit split experiment ----
    print("\n[4] Digit-split experiment — training BPE twice on an amount-heavy corpus")
    price_corpus = make_price_corpus()
    m_free = train_bpe(price_corpus, num_merges=40, digit_split=False)
    m_split = train_bpe(price_corpus, num_merges=40, digit_split=True)
    digit_merges = [a + b for a, b in m_free
                    if any(c.isdigit() for c in a + b)]
    print(f"    no split: {len(digit_merges)} merges containing digits appeared,"
          f" e.g.: {digit_merges[:6]}")

    for test in ["98700", "12500"]:
        t_free = encode(test, m_free)
        t_split = encode(test, m_split)
        print(f"    \"{test}\"  no split  : {t_free}")
        print(f"    {'':>7}  with split: {t_split}")

    print("\n[5] Why this matters for math")
    print("    No split: chunk tokens like '00' and '500' appear, so the model")
    print("      can no longer see a carrying calculation like 987+13 digit by digit.")
    print("    With split: every number breaks into consistent single digits, giving")
    print("      the model a chance to learn place-value rules (carrying and so on).")
    print("    This is why today's frontier LLMs force numbers apart into 1-3 digit pieces.")
