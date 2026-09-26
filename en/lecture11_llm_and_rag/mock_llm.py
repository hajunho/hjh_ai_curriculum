"""
Shared mock LLM / embedding module for lecture11.

A rule-based stand-in so the LLM lectures run with no API key and no internet.
- MockLLM       : a language model imitation that uses pattern matching to
                  summarize, classify, draft emails, and so on. Given retrieved
                  context, it reassembles those sentences into an answer.
- MockEmbedding : character n-gram hashing + word co-occurrence expansion.
                  Sentences about the same topic really do come out with
                  higher cosine similarity.
In a real product, an actual model such as the Claude API sits in this spot.
"""

import re
import zlib

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """A very simple sentence splitter based on ./?/! boundaries."""
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization + punctuation stripping + lowercasing.
    (Kept as simple as possible — no external tokenizer needed.)"""
    return [w.strip(".,?!():;\"'").lower() for w in text.split()
            if w.strip(".,?!():;\"'")]


# Function words that carry no topical meaning. Without this filter, an
# English question like "What is the ...?" would spuriously overlap with
# almost every document sentence through words like "the" and "is".
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "from", "with", "as", "is", "are", "was", "were", "be",
    "been", "am", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "must", "it", "its", "this", "that",
    "these", "those", "i", "we", "you", "he", "she", "they", "my", "our",
    "your", "their", "me", "us", "them", "what", "which", "who", "when",
    "where", "how", "why", "not", "no", "so", "than", "then", "there",
    "here", "up", "out", "about", "into", "over", "after", "before", "per",
    "get", "got",
}


def _stem(word: str) -> str:
    """A featherweight stem: strips common English suffixes so that
    'days'/'day' and 'granted'/'grant' can still match each other."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Equals the dot product when vectors are normalized."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# MockEmbedding — character n-gram hashing + co-occurrence expansion
# ---------------------------------------------------------------------------

class MockEmbedding:
    """
    Text -> fixed-length vector.

    How it works:
    1) Character 2-3 grams are extracted and hashed with CRC32 into one of
       dim buckets, adding a count there. "annual leave" and "annual-leave"
       share most n-grams, so their vectors land close together even when
       the surface spelling differs slightly.
    2) fit() learns per-word co-occurrence neighbors from a corpus, and at
       embedding time the neighbors' n-grams are mixed in with a small
       weight. (If "leave" and "vacation" often appear in the same document,
       sentences using either word drift toward each other.)
    """

    def __init__(self, dim: int = 512, ngram_range: tuple = (2, 3),
                 neighbor_weight: float = 0.35, max_neighbors: int = 4):
        self.dim = dim
        self.ngram_range = ngram_range
        self.neighbor_weight = neighbor_weight
        self.max_neighbors = max_neighbors
        self._cooc: dict[str, list[str]] = {}   # word -> co-occurring neighbors

    # -- internal: character n-gram extraction ------------------------------
    def _char_ngrams(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^0-9A-Za-z ]", "", text.lower()).replace(" ", "_")
        grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            grams.extend(cleaned[i:i + n] for i in range(len(cleaned) - n + 1))
        return grams

    def _add_ngrams(self, vec: np.ndarray, text: str, weight: float) -> None:
        for g in self._char_ngrams(text):
            idx = zlib.crc32(g.encode("utf-8")) % self.dim   # stable hash
            vec[idx] += weight

    # -- training: co-occurrence statistics ----------------------------------
    def fit(self, texts: list[str]) -> "MockEmbedding":
        """Build 'words that appeared in the same sentence' statistics."""
        counts: dict[str, dict[str, int]] = {}
        for text in texts:
            words = [w for w in tokenize(text)
                     if len(w) >= 2 and w not in _STOPWORDS]
            for w in words:
                bucket = counts.setdefault(w, {})
                for other in words:
                    if other != w:
                        bucket[other] = bucket.get(other, 0) + 1
        self._cooc = {
            w: [x for x, _ in sorted(nb.items(), key=lambda kv: -kv[1])[: self.max_neighbors]]
            for w, nb in counts.items()
        }
        return self

    # -- embedding ------------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        self._add_ngrams(vec, text, 1.0)
        # Mix in co-occurrence neighbors with a small weight
        # (a "leave" question also moves closer to "vacation" documents)
        for w in tokenize(text):
            for nb in self._cooc.get(w, []):
                self._add_ngrams(vec, nb, self.neighbor_weight)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])


# ---------------------------------------------------------------------------
# MockLLM — rule-based mock language model
# ---------------------------------------------------------------------------

# Keyword lexicon for complaint/inquiry classification
_CATEGORY_LEXICON = {
    "Shipping": ["shipping", "deliver", "courier", "tracking", "arrive", "package"],
    "Refund/Billing": ["refund", "payment", "card", "charge", "cancel", "double"],
    "Product defect": ["broken", "defect", "damaged", "noise", "stopped working", "repair"],
    "Customer service": ["agent", "staff", "rude", "hold", "waiting", "representative"],
}


class MockLLM:
    """
    Rule-based mock LLM.

    complete(prompt)              : looks at the instruction verbs in the
                                    prompt (summarize/classify/email...) and
                                    produces a pattern response. The more a
                                    "good prompt" spells out role, format,
                                    and examples, the more structured the
                                    answer becomes.
    answer_with_context(q, ctxs)  : for RAG. Picks the sentences that overlap
                                    the question from the retrieved context
                                    and assembles an answer that cites them.
    """

    def __init__(self, name: str = "mock-llm-v1"):
        self.name = name

    # -- split the payload (body text) out of the prompt ----------------------
    @staticmethod
    def _payload(prompt: str) -> str:
        for marker in ("Text:", "Content:", "---"):
            if marker in prompt:
                return prompt.split(marker, 1)[1].strip()
        return prompt.strip()

    # -- detect prompt-quality signals ----------------------------------------
    @staticmethod
    def _quality_flags(prompt: str) -> dict[str, bool]:
        pl = prompt.lower()
        return {
            "role": ("you are" in pl or "role" in pl),
            "format": any(k in pl for k in ("format", "bullet", "json", "numbered")),
            "example": "example" in pl,
            "context": any(k in pl for k in ("situation", "audience", "context")),
        }

    # -- sub-tasks -------------------------------------------------------------
    def _summarize(self, payload: str, n: int, structured: bool) -> str:
        """Extractive summary: pick the sentences richest in frequent words."""
        sents = split_sentences(payload)
        freq: dict[str, int] = {}
        for s in sents:
            for w in tokenize(s):
                if len(w) >= 2 and w not in _STOPWORDS:
                    freq[w] = freq.get(w, 0) + 1
        scored = [(sum(freq.get(w, 0) for w in tokenize(s)) / (len(tokenize(s)) + 1), i, s)
                  for i, s in enumerate(sents)]
        top = sorted(sorted(scored, key=lambda x: -x[0])[:n], key=lambda x: x[1])
        if structured:
            return "\n".join(f"- {s}" for _, _, s in top)
        return top[0][2] if top else "(nothing to summarize)"

    def _classify(self, payload: str) -> tuple[str, list[str]]:
        text = payload.lower()
        best, best_hits = "Other", []
        for cat, words in _CATEGORY_LEXICON.items():
            hits = [w for w in words if w in text]
            if len(hits) > len(best_hits):
                best, best_hits = cat, hits
        return best, best_hits

    def _draft_email(self, prompt: str, structured: bool) -> str:
        def _field(key, default):
            m = re.search(key + r"\s*[:：]\s*(.+)", prompt)
            return m.group(1).strip() if m else default
        to = _field("To", "whom it may concern")
        purpose = _field("Purpose", "a business request")
        deadline = _field("Deadline", "")
        body = [f"Dear {to},", "",
                f"I hope this finds you well. I am writing regarding {purpose}."]
        if deadline:
            body.append(f"If possible, I would appreciate a reply by {deadline}.")
        body += ["Please let me know if you need any material for your review.",
                 "", "Best regards,", "(your name)"]
        if not structured:
            return (f"You could just send an email about {purpose}. "
                    "Start with a greeting and end with thanks.")
        return "\n".join(body)

    # -- public API -------------------------------------------------------------
    def complete(self, prompt: str) -> str:
        """Mock responder that routes on the instruction verbs it spots."""
        flags = self._quality_flags(prompt)
        structured = flags["format"] or flags["example"]
        payload = self._payload(prompt)
        pl = prompt.lower()

        if "classif" in pl:
            cat, hits = self._classify(payload)
            if structured:
                return (f"Category: {cat}\n"
                        f"Evidence keywords: {', '.join(hits) if hits else 'none'}")
            return f"Hmm, this looks like something about {cat.lower()}."
        if "summar" in pl:
            n = 3 if structured else 1
            return self._summarize(payload, n, structured)
        if "email" in pl or "mail" in pl:
            return self._draft_email(prompt, structured or flags["role"])
        if "translat" in pl:
            return "(mock translation) Bonjour, ceci est une traduction fictive du texte."
        return ("Understood. Give me more specific instructions "
                "(role, format, examples) and the accuracy will go up.")

    def answer_with_context(self, question: str, contexts: list[tuple[str, str]],
                            max_evidence: int = 2) -> str:
        """
        RAG answer assembler. contexts = [(source, body), ...]
        Picks sentences whose words overlap the question and cites them as
        evidence. With no evidence it says it does not know — the minimal
        form of hallucination control.
        """
        q_words = {w for w in tokenize(question)
                   if len(w) >= 2 and w not in _STOPWORDS}
        # Light stemming absorbs inflection differences (days/day, granted/grant).
        q_stems = {_stem(w) for w in q_words}
        evidence = []
        for source, text in contexts:
            for sent in split_sentences(text):
                s_words = [w for w in tokenize(sent) if w not in _STOPWORDS]
                overlap = sum(1 for w in s_words
                              if w in q_words or (len(w) >= 2 and _stem(w) in q_stems))
                if overlap > 0:
                    evidence.append((overlap, source, sent))
        if not evidence:
            return ("I could not find supporting evidence in the provided documents. "
                    "I cannot answer this question.")
        evidence.sort(key=lambda x: -x[0])
        picked = evidence[:max_evidence]
        lines = ["Based on the documents, here is the answer."]
        for _, source, sent in picked:
            lines.append(f"- {sent} (source: {source})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# [Reference] With a real API it looks like this — Claude API call with a key
# ---------------------------------------------------------------------------
# import anthropic                              # pip install anthropic
# client = anthropic.Anthropic()                # uses ANTHROPIC_API_KEY env var
# response = client.messages.create(
#     model="claude-opus-5",
#     max_tokens=1024,
#     messages=[{"role": "user", "content": "Summarize these minutes in 3 lines: ..."}],
# )
# print(response.content[0].text)
# ---------------------------------------------------------------------------
