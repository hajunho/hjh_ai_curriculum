"""
Experience the 3 flagship scenarios for using an LLM at work, with a mock LLM.
1) Drafting an email  2) Summarizing meeting minutes  3) Auto-classifying complaints
At the end we print the risk checklist you must go through before adopting LLMs.
Runs with the rule-based MockLLM and no API key; a real API call example is
provided in the comments.
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from mock_llm import MockLLM

# Use-case map: flagship tasks per type, with risk level
USE_CASE_MAP = [
    ("Summarize", "meeting minutes, reports, contracts", "low (source text at hand)"),
    ("Draft", "first version of emails, notices, plans", "low (human finishes it)"),
    ("Classify", "complaint routing, email tagging", "medium (watch for mislabels)"),
    ("Extract", "pull dates, amounts, terms from docs", "medium (validation required)"),
    ("Translate", "overseas partner emails, manuals", "medium (experts check contracts)"),
    ("Code", "Excel formulas, SQL, scripts", "medium (review before running)"),
    ("Advise", "customer-facing chatbot", "high (hallucination/liability -> RAG)"),
]

RISK_CHECKLIST = [
    "How costly is a wrong answer in this task? (money, legal liability)",
    "Is a final human-review step built into the design?",
    "May company secrets or customer data be sent to an external API? (security policy)",
    "Can the answer's evidence be traced? (with RAG: source citations)",
    "Is there a procedure to undo a model mistake (rollback, correction notice)?",
    "Cost: is call volume x per-token price actually cheaper than the human cost?",
]

MEETING_NOTE = (
    "Q3 revenue grew 12% year over year, but operating profit slipped slightly "
    "because logistics costs went up. "
    "The marketing team requested a KRW 50 million increase in the campaign budget. "
    "The development team committed to finishing the order-system upgrade next month. "
    "The next meeting was scheduled for October 15 at 10 a.m. "
    "Discussion of the lunch menu was postponed until the cafeteria reopens."
)

COMPLAINTS = [
    "My package has not arrived for a week and the tracking number does not work either.",
    "I requested a refund but the card charge was never cancelled, and I was double billed.",
    "The product makes a strange noise and it stopped working since yesterday.",
    "It took 40 minutes to reach an agent and the person was rude to me.",
]


def main() -> None:
    llm = MockLLM()

    print("=" * 62)
    print("Level 01 | A map of LLM use cases at work — 3 task demos")
    print("=" * 62)

    # [1] The use-case map
    print("\n[1] LLM use-case map (type | flagship task | risk)")
    for kind, desc, risk in USE_CASE_MAP:
        print(f"    {kind:<10}| {desc:<38}| {risk}")

    # [2] Task demo 1 — email draft
    print("\n[2] Demo 1: drafting an email")
    prompt = (
        "You are an executive assistant. Draft a polite business email from the "
        "information below. Format: greeting-body-closing\n"
        "To: Kim, the sales team lead\n"
        "Purpose: a request to share the Q3 sales performance data\n"
        "Deadline: this Friday"
    )
    print("    --- Prompt gist: give To/Purpose/Deadline and ask for a draft ---")
    for line in llm.complete(prompt).splitlines():
        print(f"    | {line}")
    print("    -> The draft takes 30 seconds; the human only 'reviews and signs'.")
    print("       That is how you use LLMs as a draft generator.")

    # [3] Task demo 2 — meeting-minutes summary
    print("\n[3] Demo 2: summarizing minutes (5 sentences -> 3 key lines)")
    summary = llm.complete("Summarize the following minutes in bullet format.\nText: " + MEETING_NOTE)
    for line in summary.splitlines():
        print(f"    {line}")
    print("    -> A human still checks that low-value lines (like the lunch menu) dropped out.")

    # [4] Task demo 3 — automatic complaint classification
    print("\n[4] Demo 3: automatic complaint classification (department routing)")
    for text in COMPLAINTS:
        result = llm.complete("Classify the following complaint. Format: category/evidence keywords\nText: " + text)
        cat = result.splitlines()[0].replace("Category: ", "")
        basis = result.splitlines()[1].replace("Evidence keywords: ", "")
        print(f"    \"{text[:46]}...\"")
        print(f"      -> Category: {cat:<16} (evidence: {basis})")
    print("    -> 500 complaints a day get pre-sorted by department before anyone reads them.")

    # [5] Adoption risk checklist
    print("\n[5] Pre-adoption risk checklist — one 'no' means redesign")
    for i, item in enumerate(RISK_CHECKLIST, 1):
        print(f"    {i}. {item}")

    print("\nSummary: the safe way to start with LLMs is 'draft generator + classifier'.")
    print("         Always design in a review step where a human takes final responsibility.")

    # ------------------------------------------------------------------
    # [Reference] With a real API key it looks like this (Claude API)
    # import anthropic
    # client = anthropic.Anthropic()          # ANTHROPIC_API_KEY env var
    # response = client.messages.create(
    #     model="claude-opus-5",
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # print(response.content[0].text)
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
