"""
level03 — Tokenization and the Quirks of Korean

Tokenizes the same sentence with 4 strategies — whitespace / characters /
n-grams / a mini rule-based morphological analyzer — then runs a vocabulary-size
experiment and trains a mini BPE (Byte Pair Encoding) from scratch.
English is the working language here; Korean appears as the contrast case that
shows how much harder agglutinative languages push the same problems.
Runs on the standard library only, no NLP packages.
"""

import re
import sys
import pathlib
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# ---------------------------------------------------------------------------
# The 4 tokenization strategies
# ---------------------------------------------------------------------------

def tokenize_space(text: str) -> list[str]:
    """Strategy 1: whitespace split — the English default, but punctuation and
    capitalization stay glued on ('fast,' vs 'fast', 'Delivery' vs 'delivery')."""
    return text.split()


def tokenize_syllable(text: str) -> list[str]:
    """Strategy 2: character split — robust to unknown words, but meaning gets thin."""
    return [ch for ch in text if not ch.isspace()]


def tokenize_ngram(text: str, n: int = 2) -> list[str]:
    """Strategy 3: character n-grams — overlapping windows of n characters within a word."""
    grams = []
    for word in text.split():
        if len(word) < n:
            grams.append(word)
        else:
            grams.extend(word[i:i + n] for i in range(len(word) - n + 1))
    return grams


# --- Strategy 4: a mini rule-based morphological analyzer -------------------
# A shrunken teaching version of a real analyzer's skeleton
# (dictionary + rules + longest-match-first).

NOUNS = ["description", "performance", "atmosphere", "delivery",
         "packaging", "inquiry", "quality", "package", "rating", "manual",
         "color", "photo", "price", "staff", "store", "setup", "thing",
         "order", "star", "week"]
STEMS = ["disappoint", "incredib", "confus", "packag", "suggest", "arriv",
         "expect", "relax", "pleas", "decid", "deliver", "help", "look",
         "lov", "quick"]
JOSA = ["'s", "es", "s"]                      # noun suffixes: plural / possessive
EOMI = ["ations", "ing", "ed", "ly", "ful", "es", "s"]   # verb/adjective endings


def analyze_word(word: str) -> list[str]:
    """Break one word into [noun + suffix] or [stem + ending] (longest match first)."""
    word = word.strip(".,!?").lower()          # analyzers normalize case/punct first
    for noun in sorted(NOUNS, key=len, reverse=True):        # longest nouns first
        if word == noun:
            return [noun]
        if word.startswith(noun):
            rest = word[len(noun):]
            for josa in sorted(JOSA, key=len, reverse=True):
                if rest == josa:
                    return [noun, rest + "(suffix)"]
    for stem in sorted(STEMS, key=len, reverse=True):        # longest stems first
        if word.startswith(stem):
            rest = word[len(stem):]
            if rest == "":
                return [stem + "-"]
            for eomi in sorted(EOMI, key=len, reverse=True):
                if rest == eomi:
                    return [stem + "-", rest + "(ending)"]
    return [word]                                            # unknown words stay whole


def tokenize_morph(text: str) -> list[str]:
    """Strategy 4: mini morphological analysis — analyze_word on every word."""
    tokens = []
    for word in text.split():
        tokens.extend(analyze_word(word))
    return tokens


# ---------------------------------------------------------------------------
# Mini BPE — repeatedly merge the character pair that co-occurs most often
# ---------------------------------------------------------------------------

def bpe_train(corpus_words: list[str], n_merges: int) -> list[tuple[str, str]]:
    """Learn merge rules from the corpus. Returns: [(left piece, right piece), ...]"""
    # initialize every word as a list of characters
    words = [list(w) for w in corpus_words]
    merges = []
    for _ in range(n_merges):
        pair_count = Counter()
        for w in words:
            for a, b in zip(w, w[1:]):
                pair_count[(a, b)] += 1
        if not pair_count:
            break
        (a, b), freq = pair_count.most_common(1)[0]
        if freq < 2:                       # merging pays off only at 2+ occurrences
            break
        merges.append((a, b))
        merged = a + b
        for w in words:                    # apply the merge to every word
            i = 0
            while i < len(w) - 1:
                if w[i] == a and w[i + 1] == b:
                    w[i:i + 2] = [merged]
                else:
                    i += 1
    return merges


def bpe_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize a word by applying the learned merge rules in order."""
    pieces = list(word)
    for a, b in merges:
        i = 0
        while i < len(pieces) - 1:
            if pieces[i] == a and pieces[i + 1] == b:
                pieces[i:i + 2] = [a + b]
            else:
                i += 1
    return pieces


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_compare(sentence: str) -> None:
    print("[1] One sentence, 4 tokenization strategies")
    print(f"    sentence: {sentence!r}\n")
    strategies = [
        ("whitespace", tokenize_space(sentence)),
        ("characters", tokenize_syllable(sentence)),
        ("2-gram", tokenize_ngram(sentence, 2)),
        ("mini morph", tokenize_morph(sentence)),
    ]
    for name, tokens in strategies:
        print(f"    {name:10s} ({len(tokens):2d} tokens): {tokens}")
    print()


def demo_morph_inside() -> None:
    print("[2] Inside the mini analyzer — breaking words by 'longest match first'")
    for word in ["packages", "store's", "arrived", "expectations", "confusing",
                 "doomscrolling"]:
        print(f"    {word:14s} -> {analyze_word(word)}")
    print("    -> Words missing from the dictionary, like 'doomscrolling', stay "
          "whole (the OOV problem).\n")


def demo_vocab_size() -> None:
    print("[3] Vocabulary-size experiment — tokenizing 60 reviews per strategy")
    reviews = [r["text"] for r in hjh_data.review_corpus(60, seed=3)]
    strategies = {
        "whitespace": tokenize_space,
        "characters": tokenize_syllable,
        "2-gram": tokenize_ngram,
        "mini morph": tokenize_morph,
    }
    for name, fn in strategies.items():
        vocab = Counter()
        for r in reviews:
            vocab.update(fn(r))
        delivery = sorted(t for t in vocab if t.startswith("expect"))
        print(f"    {name:10s}: vocab {len(vocab):4d} types | 'expect' family tokens: {delivery}")
    print("    -> Whitespace scatters 'expected,/expected/expectations' into separate")
    print("       types; the morph strategy collapses them onto the stem 'expect-'.\n")


def demo_bpe() -> None:
    print("[4] Mini BPE training — 'inventing' tokens by merging frequent character pairs")
    reviews = [r["text"] for r in hjh_data.review_corpus(200, seed=3)]
    corpus_words = [w for r in reviews for w in r.split()]
    merges = bpe_train(corpus_words, n_merges=40)
    print(f"    Learned {len(merges)} merge rules. First 10:")
    for i, (a, b) in enumerate(merges[:10], start=1):
        print(f"      {i:2d}. '{a}' + '{b}' -> '{a + b}'")
    print("\n    Tokenizing words with the learned rules:")
    for word in ["delivery", "quality", "expectations", "redelivery"]:
        note = "  <- a word never seen before, still expressible as pieces!" if word == "redelivery" else ""
        print(f"      {word:13s} -> {bpe_encode(word, merges)}{note}")
    print("    -> Frequent words compress into big chunks, and the unseen 'redelivery'")
    print("       is expressed as a combination of learned pieces. This is the BPE idea")
    print("       behind GPT-family tokenizers — and why languages underrepresented in")
    print("       the training data (Korean, for instance) get fewer merge rules, more")
    print("       tokens per sentence, and therefore a bigger API bill.")


if __name__ == "__main__":
    print("=" * 70)
    print("Tokenization — the unit you split on decides the quality")
    print("=" * 70 + "\n")
    demo_compare("Delivery was incredibly fast, loved it")
    demo_morph_inside()
    demo_vocab_size()
    demo_bpe()
