# Lecture 11 · Level 03 — Structured Output and Parsing: Getting JSON Back

> For a program — not a person — to consume the LLM's answer: JSON output, schema validation, and the re-ask-on-failure pattern.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level02 / **Estimated time** 45 min

## 1. Why Learn This — The Business View

When a human reads the LLM's answer on a chat screen, a slightly messy format is fine. But in real automation — pulling the amount out of a contract into the ERP, assigning tickets from complaint classifications, stacking résumé data into a database — the LLM's output becomes **the next program's input**.

A program cannot understand "the contract amount looks like about 120 million won :)". It needs `{"amount_krw": 120000000}`. The single most common place where an LLM project sails through the demo and collapses in production is this "output format occasionally drifts" problem — and the request-validate-retry pattern you learn today is the standard cure.

## 2. Grasping It Through an Analogy

Think of expense reports. If employees email their receipts in **free-form prose**, the finance team has to read and re-type every one, and line items like "lunch, roughly 30 bucks" cause incidents. So companies use **a fixed form**: a date box, an amount box (digits only), a category box (pick one from a list).

Making the LLM answer in JSON is exactly "issuing the form." And just as finance re-checks the form on receipt (rejecting it if the amount box has words in it), the program validates the received JSON and, if something is wrong, **sends it back with the rejection reasons for resubmission**. Rejected three times? A human handles it. The loop we build today is precisely this process.

## 3. Core Concepts

### 3.1 JSON — the shared form between programs

JSON (JavaScript Object Notation) is the standard data format shaped like `{"field": value}`. It is nearly identical to a Python dictionary — one line of `json.loads()` converts it. The starting point is instructing the LLM: "Output JSON only. The fields are party_a (string), amount_krw (number)..."

### 3.2 The three classic failure modes in the wild

1. **Chit-chat mixed in**: "Sure! Here's the extraction: {...} Hope this helps!" — pleasantries wrapped around the JSON.
2. **Markdown fences**: the JSON comes back wrapped in ` ```json ... ``` `.
3. **Type/field errors**: the amount arrives as `"120 million won"` (a string), or a field is missing.

Failures 1–2 are handled by **defensive parsing** (regex out just the JSON part); failure 3 by **schema validation + re-asking**.

### 3.3 The request-validate-retry loop

```
prompt (schema stated) → response → cut out JSON → parse
  → validation passed? → use it
  → failed? → re-ask with "problems in the previous response: amount must be digits" (max N times)
  → over N? → human-review queue (fallback)
```

The key is to **ship the failure reasons verbatim in the re-ask prompt**. Tell an LLM what was wrong with its own output and it usually fixes it within 1–2 rounds. And instead of retrying forever, set a cap — past it, never drop the task silently; hand it to a human.

### 3.4 Schema validation — three checks at minimum

- **Field presence**: are all required keys there?
- **Type**: do number boxes hold numbers (`isinstance` check)?
- **Format/range**: is the date `YYYY-MM-DD`? Is the rate between 0 and 100?

Note that modern APIs offer structured-output features that enforce the schema server-side (see the comment at the bottom of main.py), making retries mostly unnecessary — but the validation habit remains essential, because the schema cannot guarantee the **content** of values (e.g. an invented amount that isn't in the contract).

### 3.5 Scope in practice — how far to delegate

The same pattern transfers directly to: extracting career history from résumés, amounts/dates/categories from receipts, complaint types from support logs, decisions/owners/deadlines from meeting minutes. What they share: "the source text is given, and the output format is fixed." Conversely, fields that require inferring what is not in the text (say, "the risk level of this contract") are judgment problems, not extraction — do not automate them without a separate verification regime. In production, store a confidence marker with each extraction (validation passed / number of retries); it makes picking samples for human audit much easier later.

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level03_structured_output
python3 main.py
```

Reading the output:

- **[1]** The task definition: a 410-character contract text and the expected 6-field schema (parties, amount, term, penalty rate).
- **[2] Attempt 1**: the mock LLM produces the classic real-world response — chit-chat + fence + `"120 million won"` (string amount) + a missing field. Defensive parsing survives it, but validation catches 2 issues.
- **[2] Attempt 2**: with the errors quoted in the prompt, correct JSON comes back and validation passes.
- **[3]** With the data confirmed, we immediately compute "penalty per day of delay = KRW 120,000" — the power of type-guaranteed data.

Code heart: `extract_json_block()` (strip fences/chit-chat), `validate()` (collect missing fields, type errors, and date-format errors all at once), `extract_with_retry()` (accumulate failure reasons in `feedback` and re-ask). `MockExtractorLLM` reproduces first-pass failure / second-pass success by checking whether the prompt carries error feedback — a simplification of a real LLM's self-correcting tendency.

## 5. Try It Yourself

1. **(Easy)** Add `"vat_included": (bool, "whether VAT is included")` to `SCHEMA`. The attempt-2 response lacks this field — what happens? (Hint: all 3 attempts fail → the human-review fallback path runs for the first time.)
2. **(Medium)** Add range validation to `validate()`: `amount_krw` must be positive, and `end_date` must come after `start_date`. (Hint: for `YYYY-MM-DD` strings, lexicographic comparison equals date comparison.)
3. **(Challenge)** Change `MockExtractorLLM`'s first response to "broken JSON with one missing brace" so the `json.JSONDecodeError` path also runs. Compare how the feedback text differs between a parse failure and a validation failure.

## 6. Common Mistakes

- **Calling `json.loads()` without a try** — LLM output is external input that can break at any moment. Exception handling is not optional; it is the default.
- **Skipping validation because the first response looked fine** — an LLM is right 97 times out of 100. The other 3 become production incidents.
- **Re-asking with the identical prompt** — without the failure reasons, you get the identical failure. Put the error messages in the prompt.
- **Infinite retries** — an uncapped loop is a cost bomb and a failure point. Around 3 attempts + human fallback is the standard.
- **Mistaking "validation passed" for "content correct"** — schema validation only checks the shape. Whether the amount matches the contract takes sample audits and evaluation (level11).

## Next Level Preview

So far we've called the model "like a function," but in reality you call a paid API across the network. Level 04 practices the operational calling patterns — API key management, token billing, retry/backoff/timeout — safely against a fake API server.
