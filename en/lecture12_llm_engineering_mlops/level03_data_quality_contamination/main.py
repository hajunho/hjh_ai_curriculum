"""
Hands-on data quality, deduplication, and contamination checking.
- Detect and remove exact/near-duplicate documents using character n-gram
  (shingle) sets and Jaccard similarity.
- Check for 'contamination' (leaked exam questions) — benchmark items mixed
  into the training corpus — via n-gram overlap, and simulate how much
  contamination inflates benchmark scores.
"""
import random
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

DUP_THRESHOLD = 0.7    # judged a near-duplicate at or above this value
NGRAM_N = 5            # character n-gram length
CONTAM_N = 10          # n-gram length for contamination checks (longer catches only wholesale leaks)
CONTAM_THRESHOLD = 0.3 # leak suspected if this share of an item's n-grams exists in the corpus


def char_ngrams(text: str, n: int) -> set:
    """Cut a string into the set of its length-n consecutive pieces (character n-grams)."""
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity = |intersection| / |union|. Identical 1.0, unrelated 0.0."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def contamination_ratio(question: str, corpus_grams: set, n: int) -> float:
    """The share of an item's n-grams that exist verbatim in the training corpus."""
    q_grams = char_ngrams(question, n)
    if not q_grams:
        return 0.0
    return sum(1 for g in q_grams if g in corpus_grams) / len(q_grams)


def build_documents(rng: random.Random):
    """Build 24 documents from tiny_corpus sentences, deliberately planting duplicates/leaks."""
    sentences = [s.strip() + "." for s in hjh_data.tiny_corpus().split(". ") if s.strip()]
    docs = []
    for i in range(24):
        picked = rng.sample(sentences, 3)          # 3 sentences = 1 document
        docs.append(" ".join(picked))
    docs[15] = docs[2]                             # plant an exact copy
    docs[19] = docs[5].replace("prepared", "completed", 1)  # a near-copy with one word changed
    return docs


BENCHMARK = [  # (question, answer) — assume a 4-choice general-knowledge exam
    ("What is the capital of France? Options: Lyon, Paris, Marseille, Nice", "Paris"),
    ("At what Celsius temperature does water boil? Options: 50, 80, 100, 120", "100"),
    ("How many days are in a week? Options: 5, 6, 7, 8", "7"),
    ("What do the interior angles of a triangle sum to? Options: 90, 180, 270, 360", "180"),
    ("Which is faster, light or sound? Options: light, sound, equal, unknown", "light"),
    ("Roughly how many days are in a year? Options: 300, 330, 365, 400", "365"),
]
LEAKED_IDX = 3  # this item gets leaked into a training document


def memorizer_score(corpus_text: str, items, rng: random.Random):
    """A hypothetical model that only gets memorized items right: if the question
    appears verbatim in the corpus it answers correctly, otherwise it guesses
    among 4 choices (25% accuracy). Returns: per-item (seen, correct) list."""
    results = []
    for question, _answer in items:
        seen = question in corpus_text          # a question from the leaked exam bank?
        correct = True if seen else (rng.random() < 0.25)
        results.append((seen, correct))
    return results


def main():
    rng = random.Random(42)  # fixed seed for reproducibility

    # [1] build the corpus --------------------------------------------------
    docs = build_documents(rng)
    q_leak, a_leak = BENCHMARK[LEAKED_IDX]
    docs[9] = docs[9] + f" Today's trivia quiz. {q_leak} The answer is {a_leak}." # leaked document
    print("[1] Training corpus built:", len(docs), "documents")
    print("    - Planted problems: exact copy (2<->15), near copy (5<->19), benchmark leak (doc 9)")
    print("    - Example document:", docs[0][:44], "...")

    # [2] near-duplicate detection ------------------------------------------
    grams = [char_ngrams(d, NGRAM_N) for d in docs]
    dup_pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = jaccard(grams[i], grams[j])
            if sim >= DUP_THRESHOLD:
                dup_pairs.append((i, j, sim))
    print(f"\n[2] Duplicate detection (character {NGRAM_N}-gram Jaccard >= {DUP_THRESHOLD})")
    for i, j, sim in dup_pairs:
        kind = "exact duplicate" if sim > 0.999 else "near duplicate"
        print(f"    - doc {i:2d} <-> doc {j:2d} : Jaccard {sim:.3f}  -> {kind}")
    if not dup_pairs:
        print("    - no duplicates found")

    # [3] deduplication ------------------------------------------------------
    drop = {j for _i, j, _s in dup_pairs}       # drop the later occurrence
    kept = [d for k, d in enumerate(docs) if k not in drop]
    print(f"\n[3] Deduplication: {len(docs)} -> {len(kept)} documents (dropped: {sorted(drop)})")
    print("    We saved training tokens that would have been wasted and cut the memorization risk.")

    # [4] benchmark contamination check --------------------------------------
    corpus_text = " ".join(kept)
    corpus_grams = char_ngrams(corpus_text, CONTAM_N)
    print(f"\n[4] Contamination check (share of item {CONTAM_N}-grams present in corpus >= {CONTAM_THRESHOLD:.0%})")
    contaminated = []
    for idx, (question, _a) in enumerate(BENCHMARK):
        ratio = contamination_ratio(question, corpus_grams, CONTAM_N)
        flag = ratio >= CONTAM_THRESHOLD
        if flag:
            contaminated.append(idx)
        mark = "*LEAK SUSPECTED" if flag else "clean"
        print(f"    - item {idx}: overlap {ratio:5.1%}  [{mark}]  {question[:26]}...")

    # [5] demo: contamination inflates the score ------------------------------
    print("\n[5] Benchmark scores of a hypothetical 'memorizer-only' model")
    results = memorizer_score(corpus_text, BENCHMARK, random.Random(6))  # dedicated guessing seed
    total = len(BENCHMARK)
    score_all = sum(c for _s, c in results) / total
    clean_items = [r for k, r in enumerate(results) if k not in contaminated]
    score_clean = sum(c for _s, c in clean_items) / max(1, len(clean_items))
    for k, (seen, correct) in enumerate(results):
        note = "saw it in the leaked bank -> automatic correct" if seen else ("guessed right" if correct else "wrong")
        print(f"    - item {k}: {'O' if correct else 'X'}  ({note})")
    print(f"    Score with contamination: {score_all:.1%}  <- the number that tends to land in slide decks")
    print(f"    Clean score             : {score_clean:.1%}  <- the number closer to real ability")
    print("    -> One leaked item inflates the score. Ask: 'Was this score decontamination-checked?'")


if __name__ == "__main__":
    main()
