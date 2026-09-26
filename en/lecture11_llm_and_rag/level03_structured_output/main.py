"""
How a program consumes LLM output: structured output (JSON) with
validation and retry.
1) Ask for the key facts of a contract to be extracted as JSON
2) The mock LLM's first response shows the failures you actually meet in
   practice (chit-chat mixed in, type errors)
3) Implement the loop: schema validation -> re-ask with the failure reasons
   -> success
This 'request-validate-retry' pattern is reused as-is with real APIs.
"""

import json
import pathlib
import re
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # mock_llm path (kept by convention)

CONTRACT = (
    "Service agreement. Hanbit Trading Co. (the 'Client') and Cheongram Solutions Inc. "
    "(the 'Contractor') hereby contract as follows for the build-out of an inventory "
    "management system. "
    "The total contract amount is KRW 120,000,000 (VAT excluded). "
    "The contract term runs from November 1, 2026 to April 30, 2027. "
    "If the Contractor delays delivery, a penalty of 0.1% of the contract amount "
    "is payable per day of delay."
)

# The schema the program expects: field name -> (type, description)
SCHEMA = {
    "party_a": (str, "client company name"),
    "party_b": (str, "contractor company name"),
    "amount_krw": (int, "contract amount (KRW, digits only)"),
    "start_date": (str, "start date YYYY-MM-DD"),
    "end_date": (str, "end date YYYY-MM-DD"),
    "penalty_rate_per_day": (float, "penalty rate per day of delay (%)"),
}


class MockExtractorLLM:
    """A mock LLM that answers extraction requests.
    1st pass: chit-chat + markdown fence + amount as a string (a classic
    real-world failure)
    2nd pass: given a reinforced prompt containing the error feedback,
    it returns correct JSON"""

    def complete(self, prompt: str) -> str:
        strict_retry = "Error" in prompt and "digits" in prompt
        if not strict_retry:
            return (
                "Sure! Here is the information extracted from the contract.\n"
                "```json\n"
                "{\n"
                '  "party_a": "Hanbit Trading Co.",\n'
                '  "party_b": "Cheongram Solutions Inc.",\n'
                '  "amount_krw": "120 million won",\n'
                '  "start_date": "2026-11-01",\n'
                '  "end_date": "2027-04-30"\n'
                "}\n"
                "```\n"
                "Hope this helps!"
            )
        return (
            "{\n"
            '  "party_a": "Hanbit Trading Co.",\n'
            '  "party_b": "Cheongram Solutions Inc.",\n'
            '  "amount_krw": 120000000,\n'
            '  "start_date": "2026-11-01",\n'
            '  "end_date": "2027-04-30",\n'
            '  "penalty_rate_per_day": 0.1\n'
            "}"
        )


def extract_json_block(text: str) -> str:
    """Defensively cut just the JSON out of a response (strip fences and chit-chat)."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    return brace.group(0) if brace else text.strip()


def validate(data: dict) -> list[str]:
    """Schema validation: collect and return every missing field and type error."""
    errors = []
    for field, (ftype, desc) in SCHEMA.items():
        if field not in data:
            errors.append(f"missing field: {field} ({desc})")
        elif not isinstance(data[field], ftype):
            errors.append(f"type error: {field} must be {ftype.__name__} "
                          f"(current value: {data[field]!r})")
    for d in ("start_date", "end_date"):
        if isinstance(data.get(d), str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", data[d]):
            errors.append(f"format error: {d} must be YYYY-MM-DD")
    return errors


def extract_with_retry(llm: MockExtractorLLM, max_attempts: int = 3) -> dict | None:
    """The loop: request -> parse -> validate -> (on failure) re-ask, citing the errors."""
    base_prompt = (
        "Extract the information from the following contract and output JSON only.\n"
        f"Fields: {', '.join(f'{k}({v[1]})' for k, v in SCHEMA.items())}\n"
        "Text: " + CONTRACT
    )
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt + feedback
        print(f"\n  --- attempt {attempt} ---")
        raw = llm.complete(prompt)
        preview = raw.replace("\n", " ")[:76]
        print(f"  response (raw excerpt): {preview}...")

        try:
            data = json.loads(extract_json_block(raw))
        except json.JSONDecodeError as e:
            print(f"  [parse failure] cannot interpret as JSON: {e}")
            feedback = "\n[Error] Output JSON only. No explanatory sentences."
            continue

        errors = validate(data)
        if not errors:
            print("  [validation passed] matches the schema.")
            return data
        print(f"  [validation failed] {len(errors)} issue(s):")
        for err in errors:
            print(f"    - {err}")
        # Ship the failure reasons back in the next prompt (induces self-correction)
        feedback = ("\n[Error] Problems in the previous response: " + " / ".join(errors) +
                    "\nAmount as digits only, include every field, output JSON only.")
    return None


def main() -> None:
    print("=" * 62)
    print("Level 03 | Structured output — JSON extraction, validation, retry")
    print("=" * 62)

    print("\n[1] Task: contract text -> structured data for a downstream system")
    print(f"    Source ({len(CONTRACT)} chars): {CONTRACT[:56]}...")
    print("    Expected schema:")
    for field, (ftype, desc) in SCHEMA.items():
        print(f"      {field:<22}{ftype.__name__:<7}{desc}")

    print("\n[2] Running the request-validate-retry loop")
    llm = MockExtractorLLM()
    result = extract_with_retry(llm)

    print("\n[3] Final result")
    if result is None:
        print("    All 3 attempts failed -> route to the human-review queue (the fallback).")
    else:
        for k, v in result.items():
            print(f"    {k:<22}= {v!r}")
        amount = result["amount_krw"]
        penalty_per_day = amount * result["penalty_rate_per_day"] / 100
        print(f"\n    Usage example: penalty per day of delay = {amount:,} x 0.1% "
              f"= KRW {penalty_per_day:,.0f}")
        print("    -> With types guaranteed, the data goes straight into math and DB inserts.")

    print("\n[4] The pattern, recapped")
    print("    (1) Nail down the format: 'output JSON only' (state the schema in the prompt)")
    print("    (2) It can still drift, so parse defensively (strip fences and chit-chat)")
    print("    (3) Re-ask with the validation failures quoted (usually fixed in 1-2 rounds)")
    print("    (4) Past the retry cap, hand it to a human (never fail silently)")

    # ------------------------------------------------------------------
    # [Reference] With the real Claude API: structured-output support can
    # enforce the schema server-side, making the retry loop mostly unnecessary.
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-opus-5", max_tokens=1024,
    #     messages=[{"role": "user", "content": "Extract from this contract: " + CONTRACT}],
    #     output_config={"format": {"type": "json_schema", "schema": {...}}},
    # )
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
