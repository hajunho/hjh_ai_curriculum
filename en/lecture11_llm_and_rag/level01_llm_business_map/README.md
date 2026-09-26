# Lecture 11 · Level 01 — A Map of LLM Use Cases at Work

> Summarize, draft, classify, extract, translate, code — we draw a map of where an LLM belongs in a company, then automate 3 tasks with a mock LLM.
**Difficulty** ⭐ / **Prerequisites** level00 / **Estimated time** 35 min

## 1. Why Learn This — The Business View

When the order comes down — "let's bring AI into this company" — the usual story goes like this: months are spent building a chatbot, the chatbot says something bizarre, and the project gets shut down. The failure is not the technology; it is **task selection**. The LLM was assigned the most dangerous job in the building (stating facts in front of customers) on day one.

Organizations that succeed go the other way. They start with work where mistakes are cheap — drafts, summaries, classification — build trust and know-how, add verification machinery (the RAG and guardrails of the later levels), and only then expand toward the customer. This level gives you that "where do we start" map.

## 2. Grasping It Through an Analogy

Adopting an LLM is like **hiring a silver-tongued new employee**. This hire writes astonishingly well, never tires, and works cheap. But they know nothing about your company — and they have a habit of bluffing when they don't know something.

Think about the order in which you'd hand this person work.

- **First week**: tidying meeting minutes, drafting emails — if it's wrong, a senior colleague fixes it, no harm done.
- **First month**: sorting incoming inquiries by department — the occasional mistake just gets re-filed.
- **Once trust is earned**: hand them the documents and let them assist with customer support — but with a strict rule: never speak without evidence.

What you must never assign is equally clear: anything that goes out under the company seal without review, and anything that exposes company secrets outside. The LLM use-case map is, in the end, "the sequence in which you delegate to the new hire."

## 3. Core Concepts

### 3.1 The six task types

| Type | Example | Why LLMs are good at it | Risk |
|---|---|---|---|
| Summarize | minutes → 3 lines | source text is right there (little room to hallucinate) | low |
| Draft | emails, notices | a human finalizes by design | low |
| Classify | complaint routing | the choices are fixed | medium |
| Extract | amounts, dates from contracts | can be formalized (level03) | medium |
| Translate | partner emails | the essence of language patterns | medium |
| Code | Excel formulas, SQL | running it verifies it | medium |

The common principle: tasks are safest when **the source text is provided, a human reviews, or a machine can verify**. Conversely, tasks that "state facts from the model's memory" (open-ended advice) are dangerous.

### 3.2 The risk checklist — six questions before adoption

The checklist printed in [5] of main.py is the core deliverable. Item 3 in particular — may personal data or secrets be sent to an external API — must be cleared with IT and Legal in advance. Many companies choose on-premises models or enterprise contracts precisely because of this item.

### 3.3 Three levels of automation

1. **Assist (Copilot)**: the LLM drafts, a human completes — the ideal starting point.
2. **Semi-automatic**: the LLM processes, a human spot-checks samples — classification and extraction.
3. **Fully automatic**: no human in the loop — only after evidence citation and guardrails (level11) are in place.

### 3.4 A feel for cost — compare against an hourly wage

Adoption decisions need a rough sense of cost. LLM APIs charge per token (a word fragment); in English a token is roughly three-quarters of a word. One email draft (about 2,000 tokens in and out combined) costs a fraction of a cent to a few cents depending on the model — compare that with 10 minutes of a person's time and the difference is orders of magnitude. But once volume reaches tens of thousands of calls a day the math changes, so build the habit of computing "cost per call × expected volume" monthly. The exact calculation and cost-tracking code come in level04. (Note: the amounts in our sample data are in Korean won, KRW.)

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level01_llm_business_map
python3 main.py
```

Reading the output:

- **[1]** The use-case map as a table. You can lift it straight into a team meeting.
- **[2]** Give MockLLM just three facts — To / Purpose / Deadline — and it produces a courteous email draft. The point is the division of labor: "the human only reviews and signs."
- **[3]** From 5 sentences of minutes it extracts the 3 key lines. Check that the "lunch menu" sentence dropped out of the summary — a mini exercise in summary QA. (In our run, the revenue, marketing-budget, and order-system lines survived; the meeting date and lunch menu were dropped.)
- **[4]** Four complaints get classified into Shipping / Refund-Billing / Product defect / Customer service, each with its evidence keywords. Making the model show its evidence is the practitioner's trick for catching mislabels.
- **[5]** The six-question risk checklist.

Code heart: this level's MockLLM is the rule-based implementation in `mock_llm.py`. `complete()` finds instruction verbs like "classify/summarize/email" in the prompt and routes the task. A real LLM works differently inside, but from the user's seat the behavior is the same: **it reads the instruction and decides its task mode.** At the bottom of the file, a comment shows the real Claude API call — once you have a key, only the `llm.complete(prompt)` line changes.

## 5. Try It Yourself

1. **(Easy)** Add a new complaint to `COMPLAINTS`: "I cannot log in to the app." How does it get classified? (Hint: types missing from the lexicon fall into "Other" — deciding how to handle inputs outside your taxonomy is a real design point.)
2. **(Medium)** Add an "Account/App" category to `_CATEGORY_LEXICON` in `mock_llm.py` (keywords: login, password, app, error) and make the complaint from exercise 1 classify correctly.
3. **(Challenge)** List 10 tasks from your (or an imaginary) company and sort them safe/risky using the 3.1 criteria (source provided? human review? machine-verifiable?). Pick the one you would "start this month" and answer all six checklist questions from [5] for it.

## 6. Common Mistakes

- **Starting with the hardest task (an open-ended support chatbot)** — the starting point with the highest failure odds. Drafts, summaries, classification first.
- **Automation with no review step** — "the LLM's email goes straight out" is a shortcut to an incident. Start at the Copilot stage.
- **Accepting classifications without evidence** — output evidence keywords alongside, and humans will find mislabels ten times faster.
- **Skipping the security review** — sending text with customer data to an external API violates policy at most companies. Learn masking first (level11).

## Next Level Preview

The very same LLM produces night-and-day results depending on how the instruction sheet (the prompt) is written. In Level 02 we run a bad prompt and a good prompt side by side and see the difference with our own eyes.
