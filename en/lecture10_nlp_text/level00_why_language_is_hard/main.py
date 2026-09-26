"""
level00 — Why Language Is Hard for Computers

We build a customer-support bot out of plain string matching (keyword search)
and then deliberately break it, demonstrating four failure modes:
synonyms / ambiguity / context (sarcasm) / changing word forms
(with agglutinative languages like Korean as the extreme case).
Standard library only.
"""


def keyword_match(text: str, keywords: list[str]) -> bool:
    """The simplest possible approach: True if any keyword appears as a substring."""
    return any(kw in text for kw in keywords)


# ---------------------------------------------------------------------------
# [1] The synonym problem — same meaning, different characters
# ---------------------------------------------------------------------------

REFUND_REQUESTS = [  # every one of these is a customer asking for a refund.
    "Please process a refund for me",
    "I want my money back",
    "I'd like to cancel the charge",
    "Cancel my order and return the payment please",
    "When will the refund be issued?",
    "Is a chargeback possible?",
]


def demo_synonym() -> None:
    print("[1] Synonyms — how many requests does a 'refund' keyword search find?")
    keywords = ["refund"]
    hit = 0
    for text in REFUND_REQUESTS:
        found = keyword_match(text, keywords)
        hit += found
        mark = "caught " if found else "missed x"
        print(f"    {mark} | {text}")
    print(f"    -> Of {len(REFUND_REQUESTS)} genuine refund requests, only {hit} detected "
          f"(hit rate {hit / len(REFUND_REQUESTS):.0%})")
    print("    -> The string has no idea that 'I want my money back' means 'refund'.\n")


# ---------------------------------------------------------------------------
# [2] The ambiguity problem — the two meanings of 'bank'
# ---------------------------------------------------------------------------

AMBIGUOUS_SENTENCES = [
    ("I deposited the check at the bank this morning", "money"),
    ("We had a picnic on the grassy bank of the river", "river"),
    ("The bank finally approved my loan application", "money"),
    ("Tall wildflowers grew all along the bank", "river"),
    ("She works as a teller at the bank downtown", "money"),
    ("The fisherman sat on the bank all afternoon", "river"),
]

# Neighboring-word (context) hints: the clue that separates the meanings
# is not the word 'bank' itself but the words around it.
CONTEXT_HINTS = {
    "river": ["river", "fisherman", "fishing", "wildflowers", "grassy", "shore"],
    "money": ["deposit", "check", "loan", "teller", "account", "interest"],
}


def guess_by_context(sentence: str) -> str:
    """A micro 'context' classifier that guesses the sense from neighboring words."""
    for sense, hints in CONTEXT_HINTS.items():
        if any(h in sentence for h in hints):
            return sense
    return "?"


def demo_ambiguity() -> None:
    print("[2] Ambiguity — the same letters 'bank', completely identical to a computer")
    correct = 0
    for sentence, answer in AMBIGUOUS_SENTENCES:
        naive = "bank" in sentence        # string matching: every line ends as 'bank found'
        guess = guess_by_context(sentence)
        correct += guess == answer
        print(f"    string match={'bank found' if naive else '-'} | "
              f"context guess={guess:5s} | answer={answer:5s} | {sentence}")
    print(f"    -> String matching cannot tell the two senses apart, but looking at "
          f"'neighboring words' separates {correct}/{len(AMBIGUOUS_SENTENCES)}")
    print("    -> 'A word's meaning is decided by its neighbors' — the core idea behind "
          "embeddings in level06.\n")


# ---------------------------------------------------------------------------
# [3] The context/sarcasm problem — the betrayal of positive keywords
# ---------------------------------------------------------------------------

REVIEWS = [  # (review, true sentiment 1=positive 0=negative)
    ("The quality is really great", 1),
    ("Fast shipping, absolutely the best", 1),
    ("Everyone said it was great, but I'm so disappointed", 0),
    ("So much for 'the best' — it broke within a day", 0),
    ("Not that great for the price", 0),
    ("Careful packaging, very satisfied", 1),
]


def demo_sarcasm() -> None:
    print("[3] Context — what if we call a review positive when it says 'great/best'?")
    keywords = ["great", "best", "satisfied"]
    wrong = 0
    for text, label in REVIEWS:
        pred = 1 if keyword_match(text, keywords) else 0
        ok = pred == label
        wrong += not ok
        mark = "right  " if ok else "wrong x"
        print(f"    {mark} | pred={'positive' if pred else 'negative'} "
              f"actual={'positive' if label else 'negative'} | {text}")
    print(f"    -> {wrong} of {len(REVIEWS)} misclassified. "
          f"The letters 'great' and the meaning 'great' are not the same thing.\n")


# ---------------------------------------------------------------------------
# [4] The word-form problem — inflection turns one word into many strings
# ---------------------------------------------------------------------------

DELIVERY_SENTENCES = [
    "Please ship it to my office address",
    "When will you ship the replacement?",
    "The shipping was surprisingly fast",
    "My order shipped a day late",
    "I want to cancel my membership",       # contains 'ship' but a different concept
    "Our relationship with this vendor is great",  # this one too
]


def demo_agglutinative() -> None:
    print("[4] Word forms — we want 'ship', but inflection keeps changing the string")
    exact = [s for s in DELIVERY_SENTENCES if "ship" in s.split()]     # exact-token match
    substr = [s for s in DELIVERY_SENTENCES if "ship" in s]            # substring match
    print(f"    exact token match ('ship' alone)   : {len(exact)} found {exact}")
    print(f"    substring match   ('ship' in s)    : {len(substr)} found")
    for s in substr:
        note = " <- memberSHIP/relationSHIP dragged in" if ("membership" in s or "relationship" in s) else ""
        print(f"        - {s}{note}")
    print("    -> Exact matching misses 'shipping/shipped' entirely, and substring")
    print("       matching drags in unrelated words. In agglutinative languages like")
    print("       Korean, one verb has dozens of surface forms — which is why")
    print("       tokenization and morphology (level03) matter so much.\n")


# ---------------------------------------------------------------------------
# [5] Roadmap
# ---------------------------------------------------------------------------

def print_roadmap() -> None:
    print("[5] The failures we met today, and where this lecture fixes them")
    roadmap = [
        ("synonyms / spelling drift", "level01 preprocessing, level06 embeddings (similar meaning -> nearby coordinates)"),
        ("ambiguity (context)", "level07 RNN, level08 attention (meaning decided by neighboring words)"),
        ("sarcasm / negation", "level05 classifier + context models from level08 onward"),
        ("word forms (inflection)", "level03 tokenization, stemming, subwords (BPE)"),
        ("brand-new words", "level03 subwords (unknown words handled as pieces)"),
    ]
    for problem, solution in roadmap:
        print(f"    {problem:26s} -> {solution}")
    print("\nBottom line: language is not 'characters', it is 'usage'. "
          "We conquer it one level at a time.")


if __name__ == "__main__":
    print("=" * 70)
    print("Why language is hard for computers — watching plain string matching collapse")
    print("=" * 70 + "\n")
    demo_synonym()
    demo_ambiguity()
    demo_sarcasm()
    demo_agglutinative()
    print_roadmap()
