"""
Prompt engineering: experience the quality gap that different instruction
sheets create with the very same model.
- Run a bad prompt vs a good prompt (with role, context, format, example)
  side by side
- Grade instruction sheets with a prompt-component checker (checklist)
MockLLM is built to answer in a more structured way the more explicitly the
prompt states role/format/examples, reproducing the quality differences
observed with real LLMs.
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from mock_llm import MockLLM

MEETING_NOTE = (
    "At this week's inventory meeting we decided to postpone the warehouse "
    "automation investment to Q1 next year. "
    "Instead, 20 temporary workers will be hired from November for the peak season. "
    "The number of errors in the inventory management system fell 40% from last month. "
    "At the next meeting we will reset the safety-stock levels for each branch. "
    "The request to repair the meeting-room air conditioner went to General Affairs."
)

COMPLAINT = ("I paid by card but my refund has not come through for two weeks, "
             "and I cannot reach customer support.")

# The 4 prompt components: Role, Context, Format, Example
CHECK_ITEMS = [
    ("Role", ["You are", "you are", "role"], "tell the model which expert to act as"),
    ("Context", ["Situation", "situation", "audience", "context"], "why / for whom the work is done"),
    ("Format", ["Format", "format", "bullet", "JSON", "numbered"], "pin down the output shape (table/list/JSON...)"),
    ("Example", ["Example", "example"], "show input/output samples (few-shot)"),
]


def audit_prompt(prompt: str) -> tuple[int, list[str]]:
    """Grade whether the 4 components are present in the prompt."""
    passed = []
    for name, keywords, _ in CHECK_ITEMS:
        if any(k in prompt for k in keywords):
            passed.append(name)
    return len(passed), passed


def show_case(title: str, prompt: str, llm: MockLLM) -> None:
    score, passed = audit_prompt(prompt)
    print(f"\n  ({title}) components present {score}/4: {', '.join(passed) if passed else 'none'}")
    print("  --- prompt ---")
    for line in prompt.strip().splitlines():
        print(f"  > {line}")
    print("  --- MockLLM response ---")
    for line in llm.complete(prompt).splitlines():
        print(f"  | {line}")


def main() -> None:
    llm = MockLLM()

    print("=" * 62)
    print("Level 02 | Prompt engineering — writing good instruction sheets")
    print("=" * 62)

    # [1] The 4 components of a prompt
    print("\n[1] The 4 components of a good instruction sheet")
    for name, _, desc in CHECK_ITEMS:
        print(f"    {name}: {desc}")

    # [2] Case A — minutes summary: bad prompt vs good prompt
    print("\n[2] Case A: summarizing meeting minutes")
    bad = "Summarize this.\nText: " + MEETING_NOTE
    good = (
        "You are an assistant who prepares meeting minutes for executive briefings.\n"
        "Situation: a busy executive must be able to read it in 30 seconds.\n"
        "Summarize the following minutes in bullet format, focusing on key decisions.\n"
        "Text: " + MEETING_NOTE
    )
    show_case("bad prompt", bad, llm)
    show_case("good prompt", good, llm)
    print("\n    -> The bad prompt just tosses back a sentence, while the good prompt")
    print("       states 'who reads it and why', earning a structured answer.")

    # [3] Case B — complaint classification: the power of few-shot
    print("\n[3] Case B: complaint classification (few-shot)")
    bad2 = "What kind of complaint is this? Classify it.\nText: " + COMPLAINT
    good2 = (
        "You are a complaint-triage specialist at a customer service center.\n"
        "Classify the following complaint. Format: one line 'Category: <name>' plus one line of evidence keywords.\n"
        "Example: 'My delivery is late' -> Category: Shipping / Evidence keywords: deliver\n"
        "Text: " + COMPLAINT
    )
    show_case("bad prompt", bad2, llm)
    show_case("good prompt", good2, llm)
    print("\n    -> Give an example (few-shot) and the output format locks in, so the")
    print("       next stage (automation) can consume it directly. Wobbly formats")
    print("       break automation.")

    # [4] Effect of each component
    print("\n[4] What each component buys you")
    effects = [
        ("Role", "sets the voice and expertise level ('like an assistant', 'like a lawyer')"),
        ("Context", "gives criteria for what to keep and drop (for executives vs practitioners)"),
        ("Format", "makes output predictable, so downstream automation gets easy"),
        ("Example", "conveys a style that is hard to describe in words with a few samples"),
    ]
    for name, effect in effects:
        print(f"    {name}: {effect}")

    print("\nSummary: a prompt is 'the work instruction sheet you hand a new employee'.")
    print("         Fill in role, context, format, example and quality climbs step by step.")

    # ------------------------------------------------------------------
    # [Reference] With a real API: the role goes in the system prompt,
    # the task goes in the user message
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-opus-5",
    #     max_tokens=1024,
    #     system="You are an assistant who prepares meeting minutes for executive briefings.",
    #     messages=[{"role": "user", "content": "Summarize the following minutes in bullet format..."}],
    # )
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
