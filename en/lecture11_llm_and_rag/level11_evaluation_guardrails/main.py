"""
Evaluation and guardrails: the techniques that get a RAG chatbot to a state
you can actually ship.
1) Auto-grade RAG responses against an eval set (question + expected evidence)
   - whether the expected keyword appears + answer/evidence agreement (groundedness)
2) Hallucination simulation: check that the grader catches a sentence that
   is not in the evidence
3) Guardrails: masking personal data (phone / national ID / email) +
   a banned-phrase / over-promise filter
"""

import pathlib
import re
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, MockLLM, split_sentences

# The eval set: question / keyword the answer must contain / expected source
# (when there is none, refusing is the right answer)
EVAL_SET = [
    {"q": "How many days in advance must I request annual leave?",
     "keyword": "3 business days", "source": "policy_vacation.txt"},
    {"q": "What is the daily allowance for an overseas business trip?",
     "keyword": "KRW 80,000", "source": "policy_expenses.txt"},
    {"q": "How many times a week is remote work allowed?",
     "keyword": "twice a week", "source": "policy_remote_work.txt"},
    {"q": "How long is the warranty period?",
     "keyword": "2 years", "source": "manual_warranty.txt"},
    {"q": "What is next year's minimum wage?",
     "keyword": None, "source": None},                      # refusing is the right answer
]

BANNED_WORDS = ["no matter what", "100% guaranteed",
                "we accept no legal liability", "our competitor"]


# ---------------------------------------------------------------------------
# Mini RAG (the condensed version of level08)
# ---------------------------------------------------------------------------

class MiniRAG:
    def __init__(self, docs: dict[str, str], min_score: float = 0.15):
        self.min_score = min_score
        self.llm = MockLLM()
        self.chunks = [(n, s) for n, d in docs.items() for s in split_sentences(d)]
        texts = [t for _, t in self.chunks]
        self.emb = MockEmbedding(dim=512).fit(texts)
        self.vectors = self.emb.embed_batch(texts)

    def answer(self, question: str) -> tuple[str, list[tuple[str, str]]]:
        sims = self.vectors @ self.emb.embed(question)
        order = np.argsort(-sims)[:3]
        kept = [(self.chunks[i][0], self.chunks[i][1])
                for i in order if sims[i] >= self.min_score]
        if not kept:
            return ("I'm sorry — I could not find evidence in the documents, "
                    "so I cannot answer this question."), []
        return self.llm.answer_with_context(question, kept), kept


# ---------------------------------------------------------------------------
# The grader: groundedness + keyword + refusal check
# ---------------------------------------------------------------------------

def groundedness(answer: str, evidence: list[tuple[str, str]]) -> tuple[float, list[str]]:
    """Computes the share of the answer's claim sentences that really do
    appear in the evidence documents. A low share means the model said
    something the evidence never supported (a hallucination signal)."""
    context = " ".join(text for _, text in evidence)
    claims, unsupported = [], []
    for line in answer.splitlines():
        line = line.strip()
        if not line.startswith("- "):          # skip boilerplate such as the lead-in line
            continue
        claim = re.sub(r"\s*\(source:.*?\)\s*$", "", line[2:]).strip()
        claims.append(claim)
        if claim not in context:               # a claim absent from the source text
            unsupported.append(claim)
    if not claims:
        return 1.0, []                         # no claims (a refusal, say) is not a violation
    return 1 - len(unsupported) / len(claims), unsupported


def grade(item: dict, answer: str, evidence: list) -> dict:
    refused = "cannot answer" in answer
    if item["keyword"] is None:                 # out-of-corpus question: full marks for refusing
        ok = refused
        return {"keyword_ok": ok, "ground": 1.0 if ok else 0.0, "refused": refused}
    g, _ = groundedness(answer, evidence)
    src_ok = any(item["source"] == s for s, _ in evidence)
    return {"keyword_ok": (item["keyword"] in answer) and src_ok,
            "ground": g, "refused": refused}


# ---------------------------------------------------------------------------
# Guardrails: masking personal data + the banned-phrase filter
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    (re.compile(r"\d{3}-\d{2}-\d{4}"), "[national ID]"),
    (re.compile(r"\(?\d{3}\)?[- ]?\d{3}-?\d{4}"), "[phone number]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
]


def mask_pii(text: str) -> tuple[str, int]:
    """Hides personal data before it goes to an external API or into a log."""
    count = 0
    for pattern, token in PII_PATTERNS:
        text, n = pattern.subn(token, text)
        count += n
    return text, count


def check_banned(text: str) -> list[str]:
    """Checks the output for banned phrases and over-promises."""
    lowered = text.lower()
    return [w for w in BANNED_WORDS if w in lowered]


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 11 | Evaluation and guardrails — a chatbot you can ship")
    print("=" * 62)

    rag = MiniRAG(hjh_data.SAMPLE_DOCS)

    # [1] Auto-grading the eval set
    print("\n[1] Auto-grading a 5-item eval set (keyword/source + groundedness)")
    total_kw, total_g = 0, 0.0
    for item in EVAL_SET:
        answer, evidence = rag.answer(item["q"])
        result = grade(item, answer, evidence)
        total_kw += result["keyword_ok"]
        total_g += result["ground"]
        expected = item["keyword"] or "(refusal)"
        print(f"    Q: {item['q']}")
        print(f"       expected: {expected:<16} | correct {'O' if result['keyword_ok'] else 'X'}"
              f" | groundedness {result['ground']:.0%}"
              f"{' | refused' if result['refused'] else ''}")
    print(f"    ---- Total: accuracy {total_kw}/{len(EVAL_SET)}, "
          f"mean groundedness {total_g / len(EVAL_SET):.0%}")
    print("    -> Comparing these numbers before/after a release, and before/after a")
    print("       prompt change, is what 'evaluation' means.")

    # [2] Hallucination detection: mix a sentence absent from the evidence into the answer
    print("\n[2] Hallucination simulation — does the grader catch the invented sentence?")
    q = "How many days in advance must I request annual leave?"
    answer, evidence = rag.answer(q)
    fake = answer + ("\n- Unused annual leave may be carried over to the next year, "
                     "up to a maximum of 30 days. (source: policy_vacation.txt)")
    for label, ans in (("clean answer", answer), ("answer with a hallucination", fake)):
        g, unsupported = groundedness(ans, evidence)
        print(f"    {label}: groundedness {g:.0%}")
        for u in unsupported:
            print(f"      !! unsupported claim detected: \"{u[:40]}...\"")
    print("    -> Below the bar (100%, say) the answer is blocked automatically and sent")
    print("       to a human reviewer. The model invents plausible source labels too, so")
    print("       verification means comparing against the source text, not trusting the label.")

    # [3] Guardrail 1: masking personal data (the input stage)
    print("\n[3] Guardrail 1 — masking personal data (before it leaves for an external API)")
    user_input = ("I would like to ask about a refund. You can reach me at 415-555-0137, "
                  "my email is j.harper@example.com, and my national ID is 900-10-1234.")
    masked, n = mask_pii(user_input)
    print(f"    original: {user_input}")
    print(f"    masked  : {masked}")
    print(f"    -> {n} pieces of personal data hidden. The LLM only ever sees the masked version.")

    # [4] Guardrail 2: the output banned-phrase / over-promise filter
    print("\n[4] Guardrail 2 — the output filter (the last gate before sending)")
    outputs = [
        "Per policy, receipts may be submitted within 7 days after the trip ends.",
        "Dear customer, a refund is available no matter what, and that is 100% guaranteed!",
    ]
    for out in outputs:
        hits = check_banned(out)
        verdict = "pass" if not hits else f"blocked (banned: {', '.join(hits)})"
        print(f"    \"{out[:34]}...\" -> {verdict}")
    print("    -> A blocked response is not sent: it is replaced by a safe default")
    print("       message or handed to a human.")

    # [5] The operating quality system on one page
    print("\n[5] The operating quality system on one page")
    print("    before release : grade the eval set (accuracy, groundedness); set a pass bar")
    print("    input stage    : mask personal data, and filter prompt injection if needed")
    print("    output stage   : groundedness check + banned-phrase/over-promise filter")
    print("    after release  : keep adding failure cases to the eval set (it is an asset)")

    print("\nSummary: 'shipping without evaluation' is quality control by gut feeling.")
    print("         Numbers (accuracy, groundedness) and gates (masking, filters) build trust.")


if __name__ == "__main__":
    main()
