"""
level01 — Text Preprocessing Basics

Takes messy English reviews (emoticons, URLs, repeated characters, stray
whitespace, inflected word endings) and cleans them through a step-by-step
pipeline. We compare word-frequency counts before and after preprocessing
to see why preprocessing is non-negotiable.
"""

import random
import re
import sys
import pathlib
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# ---------------------------------------------------------------------------
# [1] Making dirty reviews — real-world text is never clean
# ---------------------------------------------------------------------------

NOISE_PREFIX = ["★★★ ", "[Photo review] ", "", "♡♡ "]
NOISE_SUFFIX = [" lolllll", " :((((", "!!!!!", " sooooo good",
                " see: http://blog.example.com/review123", ""]


def make_dirty_reviews(n: int = 8, seed: int = 42) -> list[str]:
    """Deliberately smear noise onto hjh_data reviews to make them 'realistic'."""
    rng = random.Random(seed)
    base = hjh_data.review_corpus(n * 3, seed=3)[:n]
    dirty = []
    for row in base:
        text = (rng.choice(NOISE_PREFIX) + row["text"].replace(" ", "  ", 1)
                + rng.choice(NOISE_SUFFIX))
        if rng.random() < 0.5:
            text = "  " + text + "   "          # stray leading/trailing whitespace
        if rng.random() < 0.4:
            text = text.replace("was", "was TOTALLY")  # mixed-case shouting
        dirty.append(text)
    return dirty


# ---------------------------------------------------------------------------
# [2] Defining the pipeline steps — small functions, string in, string out
# ---------------------------------------------------------------------------

STOPWORDS = {"and", "the", "but", "see"}
# Crude suffix list to peel from word endings (incomplete on purpose —
# real stemming/morphology comes in level03)
JOSA = ["ing", "ed", "es", "ly", "s"]


def lowercase_and_strip_url(text: str) -> str:
    """Lowercase + remove URLs. Deleting URLs first makes the later steps easier."""
    text = text.lower()
    return re.sub(r"https?://\S+", " ", text)


def remove_special(text: str) -> str:
    """Keep only letters/digits/whitespace, drop the rest (★, ♡, !, [] ...)."""
    return re.sub(r"[^a-z0-9\s]", " ", text)


def collapse_repeats(text: str) -> str:
    """Collapse 3+ repeats of the same character to 2: sooooo -> soo, lolll -> loll"""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_space(text: str) -> str:
    """Squash runs of whitespace into one space, strip both ends."""
    return re.sub(r"\s+", " ", text).strip()


def drop_stopwords_and_josa(text: str) -> str:
    """Drop stopwords + peel crude suffixes off word endings (only if 2+ chars remain)."""
    words = []
    for word in text.split():
        if word in STOPWORDS:
            continue
        for josa in JOSA:                      # try longer suffixes first
            if word.endswith(josa) and len(word) - len(josa) >= 2:
                word = word[: -len(josa)]
                break
        words.append(word)
    return " ".join(words)


PIPELINE = [
    ("lowercase + strip URLs", lowercase_and_strip_url),
    ("clean special chars", remove_special),
    ("collapse repeats", collapse_repeats),
    ("normalize whitespace", normalize_space),
    ("stopwords + crude suffixes", drop_stopwords_and_josa),
]


def clean(text: str) -> str:
    """Pass the text through the whole pipeline in order."""
    for _, step in PIPELINE:
        text = step(text)
    return text


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_step_by_step(sample: str) -> None:
    print("[2] Pushing one review through, step by step")
    print(f"    original: {sample!r}")
    text = sample
    for i, (name, step) in enumerate(PIPELINE, start=1):
        text = step(text)
        print(f"    step {i} {name:27s} -> {text!r}")
    print()


def word_freq(texts: list[str], top: int = 8) -> list[tuple[str, int]]:
    counter = Counter()
    for t in texts:
        counter.update(w for w in t.split() if len(w) >= 2)
    return counter.most_common(top)


if __name__ == "__main__":
    print("=" * 70)
    print("Text preprocessing — cleaning messy reviews with a pipeline")
    print("=" * 70 + "\n")

    dirty = make_dirty_reviews(n=8, seed=42)
    print(f"[1] Generated {len(dirty)} dirty reviews (emoticons, URLs, repeats, stray spaces)")
    for d in dirty[:3]:
        print(f"    example: {d!r}")
    print()

    demo_step_by_step(dirty[0])

    print("[3] Cleaning the whole batch (original -> result)")
    cleaned = [clean(d) for d in dirty]
    for before, after in zip(dirty, cleaned):
        print(f"    {before.strip()[:34]:36s} -> {after}")
    print()

    print("[4] Word-frequency top 8, before vs after preprocessing")
    before_freq = word_freq(dirty)
    after_freq = word_freq(cleaned)
    print("    before                   | after")
    print("    " + "-" * 50)
    for (bw, bc), (aw, ac) in zip(before_freq, after_freq):
        print(f"    {bw:14s} x{bc}      | {aw:12s} x{ac}")
    print()
    print("    -> Before, 'My/my' and 'honest./honest' are counted as different words")
    print("       and noise tokens like '[Photo' sneak in; after, the same concept")
    print("       merges into one and the tally is report-ready.")
    print("\nBottom line: preprocessing is 'prepping ingredients to fit the dish'. "
          "Next level: regular expressions, properly.")
