# Lecture 11 · Level 10 — Tool Calling and Agents

> Hand the LLM a calculator and a document search, then build the skeleton of an "AI that gets work done" with a ReAct loop that repeats think → act → observe.
**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** level08 / **Estimated time** 60 min

## 1. Why Learn This — The Business View

Up to now the LLM has only *talked*. Real work does not end with talking — you have to calculate, query a database, send an email, open a ticket. Connect **tools** to an LLM and let it decide for itself which tool to use when, and you have an agent.

The business reason agents matter is the division of labour. An LLM often gets arithmetic wrong and knows nothing about your internal data, but it is genuinely good at **judgement** — which tool does this situation need? A calculator, a search index, a database cannot judge anything, but they execute exactly. Bolt the two together and you get a flexible system that nonetheless executes precisely. The flood of "AI agent" products on the market right now are, inside, exactly the loop we build today — and once you know the mechanism you stop being dazzled by the demo and start evaluating the limits and the risks (infinite loops, wrong tool calls) too.

## 2. Grasping It Through an Analogy

It is like giving a capable new assistant the office equipment. The assistant (the LLM) has good judgement but is shaky at mental arithmetic and has not memorized the whole policy handbook. So you put a calculator and a terminal for searching the handbook on their desk.

Inside the assistant's head, on being asked "what's the total daily allowance for a 4-day overseas trip?":

1. **Thought**: "I don't know the allowance rate. Let me look it up in the handbook first."
2. **Action**: type "daily allowance for business trips" into the search terminal.
3. **Observation**: "The overseas daily allowance is KRW 80,000." — noted.
4. **Thought**: "Now I need arithmetic. Calculator, not my head."
5. **Action**: 80,000 × 4 on the calculator. **Observation**: 320,000.
6. **Thought**: "That's everything." → **Answer**: "Per policy the rate is KRW 80,000 a day, so four days comes to KRW 320,000."

That repetition of think → act → observe is the ReAct (Reason + Act) loop. And just as a good manager tells the assistant "if you're still going in circles after half an hour, come ask me", an agent gets a maximum-iteration guard.

## 3. Core Concepts

### 3.1 The structure of tool use

Give the LLM a tool list in the prompt (name, description, input format) and, instead of an answer, the model emits a **structured request: "please call `calculator` with '80000*4'"**. The program — our code — actually runs the tool and pastes the result back into the prompt. The crucial fact: **our code always runs the tool**; the LLM only asks. Which is why dangerous tools (sending mail, taking payment) can have a human-approval step wedged in before execution.

### 3.2 The ReAct loop

```
repeat (at most N times):
  decision = LLM(question, observations so far)
  if decision is "final answer" -> stop
  else -> run the tool -> append the result to the observations
```

The heart of it is that the observation notebook (the scratchpad) accumulates and becomes the input to the next decision. Because every step is logged, "why did it answer that?" is traceable — a large advantage in work that has to be auditable.

### 3.3 Why the calculator is necessary

An LLM is a next-token predictor (level00), so it can happily invent a plausible number for "127 × 34". **Any task with exactly one right answer — arithmetic, dates, database lookups — should be forced through a tool.** That is the first principle of agent design.

### 3.4 Three guards

1. **A step limit**: an infinite loop means runaway API cost. Cap it, always.
2. **Validating tool input**: today's calculator runs a whitelist check that allows only digits and operators. Treat anything the LLM produced as untrusted external input.
3. **Declaring what it cannot do**: for a question no tool can answer (lunch recommendations), not trying and stating the limit *is* the correct behaviour.

### 3.5 Observability — logs are an agent's lifeline

An agent decides several steps on its own, so when it goes wrong you need to see *where* it started going wrong. At minimum, record the full Thought/Action/Observation of every step, each tool's input, output, and elapsed time, and the total call count and token cost. Today's lab prints every step to the screen partly for the demo — but it also means that in production you log the same content. Those records serve incident analysis and also the harder question: is the agent actually earning its keep (success rate, cost per case)?

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level10_tools_agents
python3 main.py
```

Reading the output:

- **[2]** Pure arithmetic: Thought (use the calculator) → `calculator('127 * 34')` → 4318 → answer. The shortest loop, one tool and done.
- **[3]** "How many days of annual leave do I have left (6 used)": document search finds the "granted 15 days" rule → the calculator does 15 − 6 = 9 → the answer cites both the evidence and the calculation. **Two tools chained in order** is this level's highlight.
- **[4]** "Total daily allowance for a 4-day overseas trip": the rate (KRW 80,000) comes from the document, the day count (4) from the question, and the calculator returns 320000.
- **[5]** The lunch question: no tool can solve it → the agent does not try, and states its limit.

Code heart: `run_agent()` *is* the ReAct loop (decide → execute → record the observation → repeat, at most 4 steps). `MockAgentLLM.decide()` imitates by rule the "what next" judgement a real LLM would make from the prompt, and it shows that even choosing the search query (`search_query`) is part of that judgement. The comment block at the bottom notes how the real Claude API's tool declaration (`tools=[{name, description, input_schema}]`) and its `stop_reason == "tool_use"` loop map one-to-one onto today's structure.

## 5. Try It Yourself

1. **(Easy)** Change "6 days" in [3] to "20 days". The calculation goes negative. Should the agent report a negative leave balance as-is? What validation would you add?
2. **(Medium)** Add a third tool `today()` (returns today's date as a string) and add a rule to `decide()` so that "what year is it?" uses it.
3. **(Challenge)** Put a "human approval" step into `run_agent`: if the tool name is not `calculator`, print `[awaiting approval] run doc_search('...')?` before executing, auto-approve for the demo, but leave an approval log entry. That is the human-in-the-loop pattern from real deployments.

## 6. Common Mistakes

- **Letting the LLM do the arithmetic or the lookup itself** — "but it was right last time" is the most dangerous sentence here. One-right-answer tasks go through tools.
- **A loop with no iteration cap** — an agent that repeats the same search forever inflates your bill in real time. A cap plus a halt log is mandatory.
- **Executing tool input without validation** — SQL, expressions, and commands the LLM produced are untrusted input. Defend with whitelist validation, read-only permissions, and approval steps.
- **Writing sloppy tool descriptions** — with a real API the model reads each tool's `description` to decide whether to use it. A thin manual means the wrong tool gets grabbed.
- **Trying to solve everything with an agent** — work with a fixed sequence of steps is cheaper and more reliable as plain code (a pipeline). Use an agent only where a genuine judgement branch exists.

## Next Level Preview

We have a chatbot and an agent — which leaves one last question: "can we trust this enough to put it in front of the company?" In Level 11 we build an evaluator that auto-scores answer/evidence agreement, plus personal-data masking and banned-phrase guardrails, completing an operable quality-control system.
