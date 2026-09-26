# Lecture 11 · Level 02 — Prompt Engineering: Writing a Good Instruction Sheet

> Same model, wildly different results depending on how the instruction sheet (prompt) is written — practice the 4 components: role, context, format, example.
**Difficulty** ⭐⭐ / **Prerequisites** level01 / **Estimated time** 40 min

## 1. Why Learn This — The Business View

In an LLM project, the technique with the highest return on investment is not swapping models or adding servers — it is **rewriting the prompt**. The cost is zero and the quality shift is visible. More than half of all "the AI gave a weird answer" complaints actually mean "the instruction was vague."

Prompt engineering is not a separate programming language. It is **the craft of writing an unambiguous work request** — in other words, the communication skill of a good manager. That is why business people with no coding background often out-prompt developers. Every bit of experience you have delegating work to teammates transfers directly.

## 2. Grasping It Through an Analogy

Compare two managers assigning work to a new hire.

**Manager A**: "Summarize this." — The new hire panics. How many lines? Who reads it? What matters? The result is a coin flip.

**Manager B**: "Imagine you're the assistant who preps executive briefings (role); the VP is busy and will read this in 30 seconds (context); give me 3 bullet lines focused on the decisions (format), like last week's report (example)." — Even a first-timer delivers a stable result.

An LLM is an extremely capable new hire whose memory resets every time. **You must hand over Manager B's instruction sheet anew each time** to get consistent quality. Prompt engineering is, in the end, "building Manager B's instruction template."

## 3. Core Concepts

### 3.1 The 4 components of a prompt

| Component | What it does | Sample phrasing |
|---|---|---|
| Role | sets voice and expertise | "You are an executive-briefing assistant" |
| Context | criteria for keep/drop decisions | "a busy executive reads it in 30 seconds" |
| Format | pins down output shape | "3 bullet lines", "as JSON" |
| Example | conveys style via samples | "Example: input → output" |

You don't have to fill in all four every time, but when a result disappoints, use them as **the checklist of what to add**.

### 3.2 Few-shot — the power of examples

Instructing without examples is called zero-shot; showing one or more input/output samples is few-shot. Examples solve two problems at once: (1) conveying a style that is hard to put into words, and (2) **locking the output format** — for classification and extraction, where a program consumes the result, a wobbling format breaks the whole automation, so few-shot is practically mandatory.

### 3.3 System prompt vs user prompt

With a real API, the instruction sheet splits into two layers. The system prompt holds what stays the same every time (role, rules); the user message holds this task's material. In company terms, the system prompt is the "job description" and the user message is "today's work request." That is the `system=` parameter in the Claude API example at the bottom of main.py.

### 3.4 The iterative improvement loop

A prompt is never finished on the first draft. The working sequence: write a draft → test with 5–10 representative inputs → study the failures and strengthen the missing components → test again. That "collection of test inputs" later grows into your evaluation set (level11).

### 3.5 Prompts are assets — templating and version control

A well-honed prompt easily evaporates into personal know-how. As a team: (1) turn working prompts into fill-in-the-blank templates and share them, (2) record the date and reason for every change (version control), and (3) collect the inputs on which a prompt failed. To notice "classification accuracy dropped from prompt v3 to v4," you need the records. This management system meets the evaluation set of level11 — every prompt change gets graded against the same eval set so improvement becomes a number.

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level02_prompt_engineering
python3 main.py
```

Reading the output:

- **[1]** The 4-component summary.
- **[2] Case A (minutes summary)**: the bad prompt ("Summarize this.") flings back a single arbitrary sentence, while the role+context+format prompt returns 3 decision-focused bullet lines. Each prompt is graded `components present N/4` above its output (0/4 vs 3/4 in our run).
- **[3] Case B (complaint classification)**: the bad prompt yields "Hmm, this looks like something about refund/billing." — fine for a human, unusable for a program. The few-shot prompt returns the fixed format `Category: Refund/Billing` / `Evidence keywords: refund, card`.
- **[4]** What each component buys you.

Code heart: `audit_prompt()` is a simple checker that grades the presence of the 4 components by keyword. `MockLLM.complete()` is designed to answer in a structured way when role/format/example are detected and vaguely otherwise, reproducing the tendency observed with real LLMs. A real model is not this mechanical, but the direction is the same: **clear instructions → stable output.**

## 5. Try It Yourself

1. **(Easy)** Delete the phrase "in bullet format" from Case A's good prompt and rerun. How does the output degrade? (Hint: the `audit_prompt` score and the response structure change together.)
2. **(Medium)** Write your own good prompt for a "translate" task and feed it to `llm.complete()`. Fill in all four components and score `4/4`. (Hint: the word "Example:" must appear to earn the example component.)
3. **(Challenge)** Build a prompt template for one task you actually do often (say, the weekly report). Leave the changing parts as blanks like `{this week's numbers}` and add code that tests it with 3 representative inputs.

## 6. Common Mistakes

- **Expecting complex results from a one-line instruction** — the "well" in "summarize this well" gives the model zero information. Write your criteria as sentences.
- **Piling up only prohibitions** — "do X" works better than "don't do Y". Best of all: show the desired shape as an example.
- **Wiring unformatted output into automation** — free-form is fine for human readers, but programmatic consumers need an enforced format (ideally JSON — level03).
- **Testing once and declaring victory** — many prompts collapse the moment the input shifts slightly. Test with at least 5 diverse inputs.

## Next Level Preview

Even a good prompt only makes the output follow the format "mostly," not 100%. To wire it into a program you need to receive JSON, validate the schema, and re-ask on failure. Level 03 builds exactly that safety pattern.
